import geopandas as gpd
import networkx as nx
import pickle
from shapely.geometry import Point, LineString, MultiPoint
import pandas as pd
from shapely.ops import nearest_points
import time

class Graph:
    class Node:
        """表示图中的一个节点。"""

        def __init__(self, id, coords):
            """
            创建一个新的节点。

            参数:
            id -- 节点的 ID
            coords -- 节点的坐标
            """
            self.id = id
            self.coords = coords
            self.adjacent_edges = []

    class Edge:
        """表示图中的一条边。"""

        def __init__(self, id, node1, node2):
            """
            创建一条新的边。

            参数:
            id -- 边的 ID
            node1, node2 -- 连接的两个节点
            """
            self.id = id
            self.nodes = {node1, node2}


    def __init__(self, shp_file):
        """
        创建一个新的图。

        参数:
        shp_file -- 用于创建图的 shapefile 文件的路径
        """
        self.shp_file = shp_file
        self.G = nx.Graph()
        self.adjacent_edges = {}

    def create_graph_from_shp(self):
        """
        从 shapefile 文件创建图。
        """
        gdf = gpd.read_file(self.shp_file)
        edge_id = 0
        for index, row in gdf.iterrows():
            if isinstance(row['geometry'], Point):
                self.G.add_node(index, geometry=row['geometry'])
            elif isinstance(row['geometry'], LineString):
                for seg_start, seg_end in zip(list(row['geometry'].coords)[:-1], list(row['geometry'].coords)[1:]):
                    self.G.add_edge(seg_start, seg_end, id=edge_id)
                    edge_id += 1

    def create_adjacent_edges(self):
        """
        创建每个节点的邻居边的列表。
        """
        for edge in self.G.edges(data=True):
            node1, node2 = edge[0], edge[1]
            edge_id = edge[2]['id']
            if node1 in self.adjacent_edges:
                self.adjacent_edges[node1].append(edge_id)
            else:
                self.adjacent_edges[node1] = [edge_id]
            if node2 in self.adjacent_edges:
                self.adjacent_edges[node2].append(edge_id)
            else:
                self.adjacent_edges[node2] = [edge_id]


    def get_neighbor_edges_nodes(self, edge_id, node_id):
        """
        给定一条边和一个节点的 ID，返回这个节点的所有邻居边的 ID，除了输入的这条边的 ID。

        参数:
        edge_id -- 边的 ID
        node_id -- 节点的 ID
        """
        return [e for e in self.adjacent_edges[node_id] if e != edge_id]
    
    def get_neighbor_edges(self, edge_id):
        """
        给定一条边的 ID，返回这条边连接的所有邻居边的 ID。

        参数:
        edge_id -- 边的 ID
        """

        # 找到给定边连接的两个节点
        node1, node2 = [node for node in self.G.edges if self.G.edges[node]['id'] == edge_id][0]

        # 找到这两个节点连接的所有边
        edges_node1 = self.adjacent_edges[node1]
        edges_node2 = self.adjacent_edges[node2]

        # 合并两个列表并去除重复的边
        neighbor_edges = list(set(edges_node1 + edges_node2))

        # 从列表中移除给定的边
        neighbor_edges.remove(edge_id)

        return neighbor_edges
    
    

    def save_graph(self, filename):
        """
        将图保存到文件。后缀请设置为 .gpickle。

        参数:
        filename -- 文件名
        """
        nx.write_gpickle(self.G, filename)

    def load_graph(self, filename):
        """
        从文件加载图。

        参数:
        filename -- 文件名
        """
        self.G = nx.read_gpickle(filename)

    def save_adjacent_edges(self, filename):
        """
        将邻居边的列表保存到文件。后缀请设置为 .pickle。

        参数:
        filename -- 文件名
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.adjacent_edges, f)

    def load_adjacent_edges(self, filename):
        """
        从文件加载邻居边的列表。

        参数:
        filename -- 文件名
        """
        with open(filename, 'rb') as f:
            self.adjacent_edges = pickle.load(f)





class Trajectory:
    def __init__(self):
        """
        创建一个新的 Trajectory 对象。
        以及简化的轨迹对象
        """
        self.df = pd.DataFrame()
        self.simplified_df = pd.DataFrame()  # 初始化一个空的 DataFrame 来存储简化的轨迹片段

    def read_from_csv(self, file_path):
        """
        从 CSV 文件中读取数据。一次性读取所有数据。TRACK_ID,VEHICLE_ID,LNG,LAT,VELOCITY,GPS_TIME

        参数:
        file_path -- CSV 文件的路径
        """
        df = pd.read_csv(file_path)

        # 创建一个新的DataFrame，并将"LNG"和"LAT列的名称更改为"x"和"y"
        df_new = df.rename(columns={'LNG': 'x', 'LAT': 'y'})

        # 将'GPS_TIME'列转换为datetime类型，并按照时间排序
        df_new['GPS_TIME'] = pd.to_datetime(df_new['GPS_TIME'])
        df_new = df_new.sort_values('GPS_TIME')

        self.df = df_new

    def parse_time(time_str: str) -> pd.Timestamp:
        return pd.to_datetime(time_str, format='%d/%m/%Y %H:%M:%S')
    

    def DataDistribute(self, start_time: pd.Timestamp, read_interval: pd.Timedelta):
        """
        按照指定的开始时间和读取间隔从 CSV 文件中读取数据，并将数据存储到一个 DataFrame 中。

        参数:
        start_time -- 开始读取的时间，是一个 pandas.Timestamp 对象
        read_interval -- 读取间隔，表示每次读取的时间长度，是一个 pandas.Timedelta 对象
        """
        df = pd.DataFrame()  # 创建一个空的 DataFrame 来存储数据
        end_time = start_time + read_interval  # 计算结束时间

        for chunk in pd.read_csv(self.file_path, chunksize=10000, parse_dates=['GPS_TIME'], date_parser=self.parse_time):
            # 获取当前块的时间范围
            chunk_time_range = chunk['GPS_TIME'].agg(['min', 'max'])

            # 如果当前块的时间范围在开始时间和结束时间之间，那么开始处理数据
            if chunk_time_range['min'] >= start_time and chunk_time_range['max'] <= end_time:
                # 在这里处理数据
                # ...

                # 将数据块添加到 DataFrame 中
                df = pd.concat([df, chunk])

            # 如果当前块的时间范围在开始时间之前，那么跳过这个块
            elif chunk_time_range['max'] < start_time:
                continue

            # 如果当前块的时间范围在结束时间之后，那么只处理在开始时间和结束时间之间的数据
            elif chunk_time_range['min'] < end_time:
                chunk = chunk[(chunk['GPS_TIME'] >= start_time) & (chunk['GPS_TIME'] < end_time)]
                # 在这里处理数据
                # ...

                # 将数据块添加到 DataFrame 中
                df = pd.concat([df, chunk])

            # 如果当前块的时间范围完全在结束时间之后，那么停止读取
            else:
                break

        # 返回存储了所有数据的 DataFrame
        self.df = df


    def get_simpl_traj(self, vehicle_id):
        """
        返回一个车辆的简化轨迹片段。

        参数:
        vehicle_id -- 车辆的 ID
        """
        # 从简化的轨迹片段中获取指定车辆的数据
        vehicle_data = self.simplified_df[self.simplified_df['VEHICLE_ID'] == vehicle_id]

        # 如果没有找到指定车辆的数据，那么返回一个空的 DataFrame
        if vehicle_data.empty:
            return pd.DataFrame()

        # 返回指定车辆的简化轨迹片段
        return vehicle_data
    

    def get_closest_node(self, graph, point):
        """
        返回最接近给定点的节点的 ID。

        参数:
        graph -- 一个 Graph 对象
        point -- 一个 Point 对象
        """
        # 创建一个 MultiPoint 对象，包含图中所有的节点
        multipoint = MultiPoint([node for node in graph.nodes()])
        
        # 找到最接近给定点的节点
        nearest = nearest_points(point, multipoint)[1]
        
        # 返回最接近的节点的 ID
        return graph.nodes().index(nearest)

    def simplify_trajectory(self, graph):
        """
        简化轨迹，将连续的 GPS 点合并为一条边。

        参数:
        graph -- 一个 Graph 对象
        """
        simplified_data = []
        traj_id = 0  # 添加一个轨迹 ID
        for vehicle_id in self.df['VEHICLE_ID'].unique():
            vehicle_data = self.df[self.df['VEHICLE_ID'] == vehicle_id]
            start_time = None
            end_time = None
            start_node = None
            end_node = None
            current_edge_id = None
            for _, row in vehicle_data.iterrows():
                if start_time is None:
                    # 这是这条边的第一个点
                    start_time = row['GPS_TIME']
                    start_node = self.get_closest_node(graph, Point(row['x'], row['y']))
                    current_edge_id = row['EDGE_ID']
                elif row['EDGE_ID'] != current_edge_id:
                    # 这是一个新的边，所以我们结束当前的边，并开始一个新的边
                    end_time = row['GPS_TIME']
                    end_node = self.get_closest_node(graph, Point(row['x'], row['y']))
                    simplified_data.append([traj_id, vehicle_id, current_edge_id, start_time, end_time, start_node, end_node])
                    traj_id += 1  # 在每个新的边上递增轨迹 ID
                    start_time = row['GPS_TIME']
                    start_node = self.get_closest_node(graph, Point(row['x'], row['y']))
                    current_edge_id = row['EDGE_ID']
            # 添加最后一条边
            end_time = vehicle_data.iloc[-1]['GPS_TIME']
            end_node = self.get_closest_node(graph, Point(vehicle_data.iloc[-1]['x'], vehicle_data.iloc[-1]['y']))
            simplified_data.append([traj_id, vehicle_id, current_edge_id, start_time, end_time, start_node, end_node])
            traj_id += 1  # 在每个新的边上递增轨迹 ID

        simplified_df = pd.DataFrame(simplified_data, columns=['TRAJ_ID', 'VEHICLE_ID', 'EDGE_ID', 'START_TIME', 'END_TIME', 'START_NODE', 'END_NODE'])
        self.simplified_df = pd.concat([self.simplified_df, simplified_df])


