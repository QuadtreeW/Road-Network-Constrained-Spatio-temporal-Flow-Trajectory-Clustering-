import geopandas as gpd
import networkx as nx
import pickle
from shapely.geometry import Point, LineString, MultiPoint
import pandas as pd
from shapely.ops import nearest_points
import time
import osmnx as ox
import leuvenmapmatching as mm
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching.matcher.distance import DistanceMatcher
import pyproj
import matplotlib.pyplot as plt
import multiprocessing
from geopy.distance import geodesic
from pyproj import CRS


class Graph:
    def __init__(self):
        self.graph_proj = None
        self.map_con = None

    def Get_map(self, place_name):
        """
        从 OpenStreetMap 获取地图。并且保存。最后转为gdf

        参数:
        place_name -- 地点的名称
        """
        
        graph = ox.graph_from_place(place_name, network_type='drive', simplify=True)
        self.graph_proj = ox.project_graph(graph)

        self.map_con = InMemMap("myosm", use_latlon=False, use_rtree=True, index_edges=True)

        # Create GeoDataFrames (gdfs)
        nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
        for nid, row in nodes[['x', 'y']].iterrows():
            self.map_con.add_node(nid, (row['x'], row['y']))
        for eid, _ in edges.iterrows():
            self.map_con.add_edge(eid[0], eid[1])

        print("地图准备完成！")

        # print(self.map_con.all_edges())

    
    def Read_mapXMl(self, file_path):
        """
        从本地文件获取地图。仅支持经纬度保存的graphml。最后转为gdf

        参数:
        file_path -- 文件的路径
        """


        graph = ox.load_graphml(file_path)

        #直接使用3857
        crs = CRS.from_epsg(3857)
        self.graph_proj = ox.project_graph(graph, to_crs=crs)

        #若不正常则还原
        # self.graph_proj = ox.project_graph(graph)

        self.map_con = InMemMap("myosm", use_latlon=False, use_rtree=True, index_edges=True)

        # Create GeoDataFrames (gdfs)
        nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
        for nid, row in nodes[['x', 'y']].iterrows():
            self.map_con.add_node(nid, (row['x'], row['y']))
        for eid, _ in edges.iterrows():
            self.map_con.add_edge(eid[0], eid[1])

        print("成功读取，地图准备完成！")



    def mm(self, path):
        """
        使用Leuven Map Matching库对给定路径进行地图匹配。

        参数:
        path (list of tuple): 需要匹配的路径。每个元组包含一个点的经度和纬度。

        返回值:
        edges (list of tuple): 匹配路径的边。每个元组包含起始节点和结束节点。
        """
        matcher = DistanceMatcher(self.map_con,
                                 max_dist=100, max_dist_init=25,  # meter
                                 min_prob_norm=0.001,
                                 non_emitting_length_factor=0.75,
                                 obs_noise=10, obs_noise_ne=75,  # meter
                                 dist_noise=20,  # meter
                                 non_emitting_states=True,
                                 max_lattice_width=5)
        states, _ = matcher.match(path)
        nodes = matcher.path_pred_onlynodes

        edges = []
        for i in range(len(nodes) - 1):
            start_node = nodes[i]
            end_node = nodes[i + 1]
            for edge in self.map_con.all_edges():
                if edge[0] == start_node and edge[2] == end_node:
                    edges.append((start_node, end_node))

        # print("States\n------")
        # print(states)
        # print("Nodes\n------")
        # print(nodes)
        # print("Edges\n------")
        # print(edges)
        # print("")
        # matcher.print_lattice_stats()

        return edges


G = None

def init_worker_from_file(map_file):
    """
    每个子进程启动时，通过读取路网文件初始化 Graph 对象，
    并将其赋值给全局变量 G。
    """
    global G
    G = Graph()
    # 通过路网文件初始化（此处假设文件保存了经纬度数据的 graphml）
    G.Read_mapXMl(map_file)
    # 调试：打印路网中边的数量
    edges_list = list(G.map_con.all_edges())
    print(f"子进程初始化完成：路网边数量：{len(edges_list)}")

def process_vehicle(vehicle_id, df):
    """
    子进程中处理单个车辆的地图匹配。
    使用全局变量 G（在 init_worker_from_file 中初始化），
    根据车辆轨迹调用 G.mm(path) 完成匹配。

    参数:
      vehicle_id -- 车辆 ID
      df         -- 整个 DataFrame 数据
    返回:
      (vehicle_id, result) 其中 result 为匹配到的边列表
    """

    global G

    try:
        # 获取该车辆的轨迹数据
        vehicle_data = df[df['VEHICLE_ID'] == vehicle_id]
        if vehicle_data.empty or vehicle_data.isnull().values.all():
            print(f"车 {vehicle_id}: 数据为空或全为 NaN")
            return vehicle_id, None

        # 将经度和纬度组合成轨迹点列表
        path = list(zip(vehicle_data['LNG'], vehicle_data['LAT']))
        # print(f"车 {vehicle_id}: 原始轨迹前5个点：{path[:5]}，共 {len(path)} 个点")

        # 开始计时，并调用匹配函数
        start_time = time.time()
        result = G.mm(path)
        elapsed = time.time() - start_time

        print(f"车 {vehicle_id}: 匹配完成，共 {len(result)} 条边，耗时 {elapsed:.3f} 秒")
        if len(result) == 0:
            print(f"车 {vehicle_id}: 匹配结果为空，请检查匹配参数或路网数据是否正确。")
        return vehicle_id, result

    except Exception as e:
        print(f"车 {vehicle_id}: 匹配过程中出现异常：{e}")
        return vehicle_id, None




class Trajectory:
    def __init__(self):
        """
        创建一个新的 Trajectory 对象。
        以及简化的轨迹对象
        """
        self.df = pd.DataFrame()
        self.simplified_list = {}
        self.current_time = None
        self.window_size = None

    def read_from_csv(self, file_path):
        """
        从 CSV 文件中读取数据。一次性读取所有数据。TRACK_ID,VEHICLE_ID,LNG,LAT,VELOCITY,GPS_TIME

        参数:
        file_path -- CSV 文件的路径
        """
        df = pd.read_csv(file_path)

        

        # 将'GPS_TIME'列转换为datetime类型，并按照时间排序
        df['GPS_TIME'] = pd.to_datetime(df['GPS_TIME'])
        df_new = df.sort_values('GPS_TIME')

        self.df = df_new


    def filter_GPS(self, lat_threshold):
        """
        过滤基本不动的车辆。阈值单位为纬度差。

        参数:
        lat_threshold -- 最小移动纬度差阈值，用于排除基本不动的车辆
        """

        # 按照 VEHICLE_ID 和 GPS_TIME 排序
        self.df['GPS_TIME'] = pd.to_datetime(self.df['GPS_TIME'])
        self.df = self.df.sort_values(by=['VEHICLE_ID', 'GPS_TIME'])

        # 获取所有的车辆ID
        vehicle_ids = self.df['VEHICLE_ID'].unique()
        # 统计排除的车辆数量
        excluded_count = 0
        # 存储排除的车辆ID
        excluded_vehicle_ids = []
        # 遍历每辆车
        for vehicle_id in vehicle_ids:
            # 获取指定车辆的数据
            vehicle_data = self.df[self.df['VEHICLE_ID'] == vehicle_id]

            # 如果没有找到指定车辆的数据，那么跳过这辆车
            if vehicle_data.empty:
                continue
            # 获取第一个和最后一个 GPS 点的纬度
            first_lat = vehicle_data.iloc[0]['LAT']
            last_lat = vehicle_data.iloc[-1]['LAT']

            # 判断纬度差是否在阈值内
            if abs(first_lat - last_lat) <= lat_threshold:
                excluded_count += 1
                excluded_vehicle_ids.append(vehicle_id)
        # 更新 self.df，排除基本不动的车辆
        self.df = self.df[~self.df['VEHICLE_ID'].isin(excluded_vehicle_ids)]
        print(f"排除的车辆数量: {excluded_count}")



    def set_time_window(self, start_time, window_size):
        """
        设置时间窗口的开始时间和大小。

        参数:
        start_time -- 开始时间，是一个 pandas.Timestamp 对象
        window_size -- 时间窗口的大小，是一个 pandas.Timedelta 对象
        """
        self.current_time = start_time
        self.window_size = window_size

    
    def data_generator(self, file_path):
        """
        创建一个生成器，每次请求数据时只处理和返回当前时间窗口的数据。

        参数:
        file_path -- CSV 文件的路径
        """
        while True:
            df = pd.DataFrame()  # 创建一个空的 DataFrame 来存储数据
            end_time = self.current_time + self.window_size  # 计算结束时间

            for chunk in pd.read_csv(file_path, chunksize=10000, parse_dates=['GPS_TIME'], date_parser=self.parse_time):
                # 获取当前块的时间范围
                chunk_time_range = chunk['GPS_TIME'].agg(['min', 'max'])

                # 如果当前块的时间范围在当前时间和结束时间之间，那么开始处理数据
                if chunk_time_range['min'] >= self.current_time and chunk_time_range['max'] <= end_time:
                    # 将数据块添加到 DataFrame 中
                    df = pd.concat([df, chunk])

                # 如果当前块的时间范围在结束时间之后，那么停止读取
                elif chunk_time_range['min'] > end_time:
                    break

            # 更新当前时间为下一个时间窗口的开始时间
            self.current_time += self.window_size / 2

            # 使用 yield 关键字返回当前时间窗口的数据
            yield df


    def get_simpl_traj(self, graph: Graph):
        """
        对每辆车的轨迹进行地图匹配，并返回简化轨迹字典。

        参数:
        graph -- 图对象，用于调用mm函数
        """
        # 获取所有的车辆ID
        vehicle_ids = self.df['VEHICLE_ID'].unique()

        # 对每辆车的轨迹进行地图匹配
        for vehicle_id in vehicle_ids:
            # 从df中获取指定车辆的数据
            vehicle_data = self.df[self.df['VEHICLE_ID'] == vehicle_id]

            # 如果没有找到指定车辆的数据，那么跳过这辆车
            if vehicle_data.isnull().values.all():
                continue

            # 将数据转换为路径
            path = list(zip(vehicle_data['LNG'], vehicle_data['LAT']))

            # 调用graph类的mm函数，将结果存储在字典中
            self.simplified_list[vehicle_id] = graph.mm(path)

            # 打印车辆的 ID 和匹配的边的数量
            print(f'车 {vehicle_id} 匹配完成，共 {len(self.simplified_list[vehicle_id])} 条边')

        # 返回整个简化轨迹字典
        return self.simplified_list
    


    
    


    def get_simpl_traj_curr(self, map_file):
        """
        使用多进程并行对每辆车的轨迹进行地图匹配，
        每个子进程通过 map_file 读取并初始化自己的路网对象。
        
        参数:
          map_file -- 路网文件路径（例如 graphml 文件）
        返回:
          simplified_list -- 字典，键为车辆 ID，值为匹配到的边列表
        """
        vehicle_ids = self.df['VEHICLE_ID'].unique()

        start_time = time.time()


        with multiprocessing.Pool(initializer=init_worker_from_file, initargs=(map_file,)) as pool:
            results = pool.starmap(process_vehicle, [(vehicle_id, self.df) for vehicle_id in vehicle_ids])
        for vehicle_id, result in results:
            if result is not None:
                self.simplified_list[vehicle_id] = result

        endtime = time.time()

        print(f"多进程并行处理完成，共 {len(self.simplified_list)} 辆车，耗时 {endtime - start_time:.3f} 秒")

        return self.simplified_list
    





    
