import numpy as np
import geopandas as gpd
import networkx as nx
import pickle
from shapely.geometry import Point, LineString, MultiPoint
import pandas as pd
from shapely.ops import nearest_points
from datetime import datetime, timedelta
from Gmm import Graph,Trajectory
import matplotlib.pyplot as plt
import osmnx as ox
import time
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform

class ParalClustering:

    def __init__(self):
        """
        使用字典创建一个新的 ParalClustering 对象。
        """
        self.clusters = {}
        self.minpts = 0
        self.graph = None
    

    def ExpandCluster(self, win_traj: pd.DataFrame, edge: tuple, cluster_id: int) -> pd.DataFrame:
        """
        扩展簇。
    
        参数:
        win_traj -- 时间窗口内的所有轨迹，是一个 DataFrame 对象
        edge -- 要处理的边，由两个节点组成的元组
        cluster_id -- 当前簇的 ID
    
        返回:
        更新后的轨迹片段 DataFrame
        """
    
        # 创建一个队列，用于存储待访问的边
        edge_queue = [edge]
    
        while edge_queue:
            current_edge = edge_queue.pop(0)
            current_edge_mask = (win_traj['EDGE'] == current_edge) | (win_traj['EDGE'] == current_edge[::-1])
    
            # 如果当前边的轨迹数大于 minpts，并且它们还没有被分配到其他簇，将它添加到当前簇
            if current_edge_mask.sum() >= self.minpts and (win_traj.loc[current_edge_mask, 'cluster_id'] == -1).all():
                win_traj.loc[current_edge_mask, 'cluster_id'] = cluster_id
    
                # 访问当前边的邻居
                neighbor_edges = self.graph.map_con.edges_nbrto(current_edge)
                for neighbor_edge in neighbor_edges:
                    # 提取邻居边的节点
                    neighbor_edge_nodes = (neighbor_edge[0], neighbor_edge[2])
                    neighbor_edge_mask = (win_traj['EDGE'] == neighbor_edge_nodes) | (win_traj['EDGE'] == neighbor_edge_nodes[::-1])
                    neighbor_trajectories = win_traj[neighbor_edge_mask]
                    # 如果邻居边的轨迹数大于 minpts，并且它还没有被访问过，那么将它添加到队列中
                    if len(neighbor_trajectories) >= self.minpts and neighbor_edge_nodes not in edge_queue and neighbor_edge_nodes[::-1] not in edge_queue:
                        edge_queue.append(neighbor_edge_nodes)
    
        return win_traj

    
    
    def cluster_traj_init(self, simplified_dict: dict, Graph: Graph, minpts: int) -> dict:
        """
        对轨迹进行聚类。
    
        参数:
        simplified_dict -- 简化后的轨迹字典，每个键是车辆ID，值是简化后的轨迹片段
        Graph -- 一个路网 Graph 对象
        minpts -- DBSCAN 的 minpts 参数，最少轨迹数
    
        返回:
        一个字典，其中的每个键值对对应一个簇和它包含的轨迹片段
        """
        self.minpts = minpts
        self.graph = Graph
        start_time = time.time()
    
        # 在每个轨迹片段中添加车辆的 ID，然后将所有轨迹片段合并到一个 DataFrame 中
        win_traj = pd.concat([pd.DataFrame({'EDGE': data, 'VEHICLE_ID': vehicle_id}) 
                              for vehicle_id, data in simplified_dict.items() if isinstance(data, list)])
    
        win_traj['cluster_id'] = -1  # 初始化簇 ID
    
        cluster_id = 1
    
        # 遍历每个边
        for edge in win_traj['EDGE'].unique():
            edge_mask = win_traj['EDGE'] == edge
            if win_traj.loc[edge_mask, 'cluster_id'].iloc[0] == -1:
                same_edge_traj_mask = edge_mask | (win_traj['EDGE'] == edge[::-1])
                if same_edge_traj_mask.sum() >= self.minpts:
                    # 如果同一边的数量大于 minpts，则形成一个簇
                    # print("形成一个簇")
    
                    win_traj = self.ExpandCluster(win_traj, edge, cluster_id)
                    cluster_id += 1  # 创建新的簇 ID
                else:
                    win_traj.loc[same_edge_traj_mask, 'cluster_id'] = 0  # 将轨迹片段标记为已访问但未形成簇
    
        # 创建一个字典，其中的每个键值对对应一个簇和它包含的轨迹片段
        clusters = {cid: {'TRAJECTORIES': win_traj[win_traj['cluster_id'] == cid],
                          'VEHICLE_COUNT': win_traj[win_traj['cluster_id'] == cid]['VEHICLE_ID'].nunique()}
                    for cid in win_traj['cluster_id'].unique() if cid > 0}
    
        if not clusters:
            print("没有聚类出来的簇！")
        else:
            print(f"聚类的个数为：{len(clusters)}")
            for key, value in clusters.items():
                print(f"簇 {key} 的轨迹数为：{len(value['TRAJECTORIES'])}，包含车辆数：{value['VEHICLE_COUNT']}")
    
        self.clusters = clusters  # 将 clusters 设置为对象的属性

        end_time = time.time()
        print(f"聚类用时：{end_time - start_time} 秒")
    
        return clusters


    def init_nbrt(self, simplified_dict: dict, Graph: Graph, minpts: int) -> dict:
        """
        对轨迹进行聚类。
    
        参数:
        simplified_dict -- 简化后的轨迹字典，每个键是车辆ID，值是简化后的轨迹片段
        Graph -- 一个路网 Graph 对象
        minpts -- DBSCAN 的 minpts 参数，最少轨迹数
    
        返回:
        一个字典，其中的每个键值对对应一个簇和它包含的轨迹片段
        """
        self.minpts = minpts
        self.graph = Graph
        start_time = time.time()
    
        # 在每个轨迹片段中添加车辆的 ID，然后将所有轨迹片段合并到一个 DataFrame 中
        win_traj = pd.concat([pd.DataFrame({'EDGE': data, 'VEHICLE_ID': vehicle_id}) 
                              for vehicle_id, data in simplified_dict.items() if isinstance(data, list)])
    
        win_traj['cluster_id'] = -1  # 初始化簇 ID
    
        cluster_id = 1
    
        # 遍历每个边
        for edge in win_traj['EDGE'].unique():
            edge_mask = win_traj['EDGE'] == edge
            if win_traj.loc[edge_mask, 'cluster_id'].iloc[0] == -1:
                same_edge_traj_mask = edge_mask | (win_traj['EDGE'] == edge[::-1])
                if same_edge_traj_mask.sum() >= self.minpts:
                    # 如果同一边的数量大于 minpts，则形成一个簇
                    win_traj = self.ExpandCluster(win_traj, edge, cluster_id)
                    cluster_id += 1  # 创建新的簇 ID
                else:
                    win_traj.loc[same_edge_traj_mask, 'cluster_id'] = 0  # 将轨迹片段标记为已访问但未形成簇
    
        # 创建一个字典，其中的每个键值对对应一个簇和它包含的轨迹片段
        clusters = {}
        for cid in win_traj['cluster_id'].unique():
            if cid > 0:
                cluster_traj = win_traj[win_traj['cluster_id'] == cid]
                neighbor_edges = set()
                for edge in cluster_traj['EDGE'].unique():
                    # 只记录同一簇内的邻接边
                    for neighbor_edge in self.graph.map_con.edges_nbrto(edge):
                        neighbor_edge_nodes = (neighbor_edge[0], neighbor_edge[2])
                        if (cluster_traj['EDGE'] == neighbor_edge_nodes).any() or (cluster_traj['EDGE'] == neighbor_edge_nodes[::-1]).any():
                            neighbor_edges.add(neighbor_edge_nodes)
                clusters[cid] = {
                    'TRAJECTORIES': cluster_traj,
                    'VEHICLE_COUNT': cluster_traj['VEHICLE_ID'].nunique(),
                    'NEIGHBOR_EDGES': neighbor_edges
                }
    
        if not clusters:
            print("没有聚类出来的簇！")
        else:
            print(f"聚类的个数为：{len(clusters)}")
            for key, value in clusters.items():
                print(f"簇 {key} 的轨迹数为：{len(value['TRAJECTORIES'])}，包含车辆数：{value['VEHICLE_COUNT']}")
    
        self.clusters = clusters  # 将 clusters 设置为对象的属性
    
        end_time = time.time()
        print(f"聚类用时：{end_time - start_time} 秒")
    
        return clusters


    def update_cluster(self, simplified_dict: dict, Graph: Graph, minpts: int, prev_clusters: dict) -> dict:
        """
        更新簇。

        参数:
        simplified_dict -- 简化后的轨迹字典，每个键是车辆ID，值是简化后的轨迹片段
        Graph -- 一个路网 Graph 对象
        minpts -- DBSCAN 的 minpts 参数，最少轨迹数
        prev_clusters -- 上一次的簇，是一个字典，其中的每个键值对对应一个簇和它包含的轨迹片段

        返回:
        一个字典，其中的每个键值对对应一个簇和它包含的轨迹片段
        """
        self.minpts = minpts
        self.graph = Graph
        start_time = time.time()

        win_traj = pd.concat([pd.DataFrame({'EDGE': data, 'VEHICLE_ID': vehicle_id}) 
                            for vehicle_id, data in simplified_dict.items() if isinstance(data, list)])

        win_traj['cluster_id'] = -1  # 初始化簇 ID

        cluster_id = 1

        # 创建一个集合来存储已经处理过的边
        processed_edges = set()

        # 首先遍历前一次聚类结果中的边
        for cluster in prev_clusters.values():
            for edge in cluster['TRAJECTORIES']['EDGE'].unique():
                if edge not in processed_edges and edge[::-1] not in processed_edges:
                    edge_mask = win_traj['EDGE'] == edge
                    if edge_mask.any():  # 检查是否存在匹配的边
                        if win_traj.loc[edge_mask, 'cluster_id'].iloc[0] == -1:
                            same_edge_traj_mask = edge_mask | (win_traj['EDGE'] == edge[::-1])
                            if same_edge_traj_mask.sum() >= self.minpts:
                                win_traj = self.ExpandCluster(win_traj, edge, cluster_id)
                                cluster_id += 1
                            else:
                                win_traj.loc[same_edge_traj_mask, 'cluster_id'] = 0  # 将轨迹片段标记为已访问但未形成簇
                    processed_edges.add(edge)

        # 然后遍历剩下的边
        for edge in win_traj['EDGE'].unique():
            if edge not in processed_edges and edge[::-1] not in processed_edges:
                edge_mask = win_traj['EDGE'] == edge
                if win_traj.loc[edge_mask, 'cluster_id'].iloc[0] == -1:
                    same_edge_traj_mask = edge_mask | (win_traj['EDGE'] == edge[::-1])
                    if same_edge_traj_mask.sum() >= self.minpts:
                        win_traj = self.ExpandCluster(win_traj, edge, cluster_id)
                        cluster_id += 1
                    else:
                        win_traj.loc[same_edge_traj_mask, 'cluster_id'] = 0
                processed_edges.add(edge)

        clusters = {cid: {'TRAJECTORIES': win_traj[win_traj['cluster_id'] == cid],
                        'VEHICLE_COUNT': win_traj[win_traj['cluster_id'] == cid]['VEHICLE_ID'].nunique()}
                    for cid in win_traj['cluster_id'].unique() if cid > 0}

        if not clusters:
            print("没有聚类出来的簇！")
        else:
            print(f"聚类的个数为：{len(clusters)}")
            for key, value in clusters.items():
                print(f"簇 {key} 的轨迹数为：{len(value['TRAJECTORIES'])}，包含车辆数：{value['VEHICLE_COUNT']}")

        self.clusters = clusters
        end_time = time.time()
        print(f"聚类更新用时：{end_time - start_time} 秒")
        return clusters


    def update_cluster_with_overlap_optimized(self, curr_simplified_dict: dict, Graph: Graph, minpts: int, prev_clusters: dict) -> dict:
        """
        更新簇，利用前次聚类结果中的边，并优化邻居访问次数。
    
        参数:
        curr_simplified_dict -- 当前时间窗口的简化后的轨迹字典，每个键是车辆ID，值是简化后的轨迹片段
        Graph -- 一个路网 Graph 对象
        minpts -- DBSCAN 的 minpts 参数，最少轨迹数
        prev_clusters -- 上一次的簇，是一个字典，其中的每个键值对对应一个簇和它包含的轨迹片段
    
        返回:
        一个字典，其中的每个键值对对应一个簇和它包含的轨迹片段
        """
        self.minpts = minpts
        self.graph = Graph
        start_time = time.time()
    
        # 将当前时间窗口的轨迹字典转换为 DataFrame
        curr_win_traj = pd.concat([pd.DataFrame({'EDGE': data, 'VEHICLE_ID': vehicle_id}) 
                                   for vehicle_id, data in curr_simplified_dict.items() if isinstance(data, list)])
    
        curr_win_traj['cluster_id'] = -1  # 初始化簇 ID
        cluster_id = max(prev_clusters.keys()) + 1 if prev_clusters else 1
    
        # 创建一个集合来存储已经处理过的边
        processed_edges = set()
    
        # 创建一个字典来临时存储邻接边信息
        temp_neighbor_edges = {}
    
        # 通过前次聚类结果中的边，统计本次窗口内每个边的车辆数量是否大于 minpts
        for cluster in prev_clusters.values():
            for edge in cluster['TRAJECTORIES']['EDGE'].unique():
                edge_mask = curr_win_traj['EDGE'] == edge
                if edge_mask.any():
                    if curr_win_traj.loc[edge_mask, 'cluster_id'].iloc[0] == -1:
                        same_edge_traj_mask = edge_mask | (curr_win_traj['EDGE'] == edge[::-1])
                        if same_edge_traj_mask.sum() >= self.minpts:
                            curr_win_traj.loc[same_edge_traj_mask, 'cluster_id'] = cluster_id
                            processed_edges.update(same_edge_traj_mask[same_edge_traj_mask].index)
                            # 检查并初始化 NEIGHBOR_EDGES
                            if 'NEIGHBOR_EDGES' in cluster:
                                temp_neighbor_edges[cluster_id] = set(cluster['NEIGHBOR_EDGES'])
                            else:
                                temp_neighbor_edges[cluster_id] = set()
                            cluster_id += 1
    
        # 遍历当前窗口剩下的边，扩展并合并到初始簇或形成新的簇
        for edge in curr_win_traj['EDGE'].unique():
            if edge not in processed_edges and edge[::-1] not in processed_edges:
                edge_mask = curr_win_traj['EDGE'] == edge
                if edge_mask.any():
                    if curr_win_traj.loc[edge_mask, 'cluster_id'].iloc[0] == -1:
                        same_edge_traj_mask = edge_mask | (curr_win_traj['EDGE'] == edge[::-1])
                        if same_edge_traj_mask.sum() >= self.minpts:
                            # 尝试扩展当前边
                            edge_queue = [edge]
                            expanded_traj = curr_win_traj.copy()
                            expanded_edge_mask = expanded_traj['cluster_id'] == cluster_id
                            merged = False
    
                            while edge_queue:
                                current_edge = edge_queue.pop(0)
                                current_edge_mask = (expanded_traj['EDGE'] == current_edge) | (expanded_traj['EDGE'] == current_edge[::-1])
    
                                if current_edge_mask.sum() >= self.minpts and (expanded_traj.loc[current_edge_mask, 'cluster_id'] == -1).all():
                                    expanded_traj.loc[current_edge_mask, 'cluster_id'] = cluster_id
    
                                    # 检查是否有邻居边在初始簇中
                                    neighbor_edges = self.graph.map_con.edges_nbrto(current_edge)
                                    for neighbor_edge in neighbor_edges:
                                        neighbor_edge_nodes = (neighbor_edge[0], neighbor_edge[2])
                                        neighbor_edge_mask = (expanded_traj['EDGE'] == neighbor_edge_nodes) | (expanded_traj['EDGE'] == neighbor_edge_nodes[::-1])
                                        if neighbor_edge_mask.any() and expanded_traj.loc[neighbor_edge_mask, 'cluster_id'].iloc[0] > 0:
                                            cluster_id_to_merge = expanded_traj.loc[neighbor_edge_mask, 'cluster_id'].iloc[0]
                                            if cluster_id_to_merge not in temp_neighbor_edges:
                                                temp_neighbor_edges[cluster_id_to_merge] = set()
                                            expanded_traj.loc[expanded_edge_mask, 'cluster_id'] = cluster_id_to_merge
                                            temp_neighbor_edges[cluster_id_to_merge].update(neighbor_edges)
                                            merged = True
                                            break
                                        if neighbor_edge_nodes not in edge_queue and neighbor_edge_nodes[::-1] not in edge_queue:
                                            edge_queue.append(neighbor_edge_nodes)
    
                                if merged:
                                    break
    
                            if not merged:
                                temp_neighbor_edges[cluster_id] = set(neighbor_edges)
                                cluster_id += 1
    
                            curr_win_traj = expanded_traj
                        else:
                            curr_win_traj.loc[same_edge_traj_mask, 'cluster_id'] = 0
                    processed_edges.update(same_edge_traj_mask[same_edge_traj_mask].index)
    
        # 创建一个字典，其中的每个键值对对应一个簇和它包含的轨迹片段
        clusters = {}
        for cid in curr_win_traj['cluster_id'].unique():
            if cid > 0:
                cluster_traj = curr_win_traj[curr_win_traj['cluster_id'] == cid]
                neighbor_edges = temp_neighbor_edges.get(cid, set())
                clusters[cid] = {
                    'TRAJECTORIES': cluster_traj,
                    'VEHICLE_COUNT': cluster_traj['VEHICLE_ID'].nunique(),
                    'NEIGHBOR_EDGES': neighbor_edges
                }
    
        if not clusters:
            print("没有聚类出来的簇！")
        else:
            print(f"聚类的个数为：{len(clusters)}")
            for key, value in clusters.items():
                print(f"簇 {key} 的轨迹数为：{len(value['TRAJECTORIES'])}，包含车辆数：{value['VEHICLE_COUNT']}")
    
        self.clusters = clusters
        end_time = time.time()
        print(f"聚类更新用时：{end_time - start_time} 秒")
        return clusters



    def visualize_clusters(self, title=None):
        """
        可视化聚类结果，将不同的簇的边用不同的颜色标识。
    
        参数:
        title -- 图形的标题，如果不提供，就不显示标题
        """
        # 创建一个颜色列表
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
        # 创建一个颜色字典，其中的每个键值对对应一个簇的 ID 和它的颜色
        color_dict = {cluster_id: colors[i % len(colors)] for i, cluster_id in enumerate(self.clusters.keys())}
    
        # 显示图形，设置节点颜色为透明，背景颜色为白色，路网颜色为透明度为0.4的灰色
        fig, ax = ox.plot_graph(self.graph.graph_proj, node_size=0, node_color='none', edge_color='gray', edge_alpha=0.4, bgcolor='w', show=False, close=False)
    
        # 如果提供了标题，就设置图形的标题
        if title is not None:
            plt.title(title)
    
        # 遍历每个簇
        for cluster_id, cluster in self.clusters.items():
            # 只处理 cluster_id 大于 0 的簇
            if cluster_id > 0:
                # 获取当前簇的颜色
                color = color_dict[cluster_id]
    
                # 获取当前簇的所有边
                for _, row in cluster['TRAJECTORIES'].iterrows():
    
                    nodes = [row['EDGE'][0], row['EDGE'][1]]
    
                    # 使用节点列表绘制边
                    ox.plot_graph_route(self.graph.graph_proj, nodes, route_color=color, route_linewidth=2, route_alpha=1, orig_dest_size=0, orig_dest_node_color='none', orig_dest_node_edgecolor='none', with_nodes=False, ax=ax, show=False, close=False)
    
        # 在图形上添加 minpts 值，颜色设置为黑色
        plt.text(0.5, 0, 'minpts: {}'.format(self.minpts), transform=plt.gca().transAxes, color='black')
    
        # 保存图形，路径和分辨率已经写死
        # plt.savefig("my_figure2.png", dpi=300)
    
        # 显示图形
        plt.show()

    
    def edge_filter(self, distance_threshold):
        """
        根据空间大小筛选簇。注意单位是米。

        参数:
        distance_threshold -- 距离阈值，用于筛选簇
        """
        filtered_clusters = {}

        for cluster_id, cluster in self.clusters.items():
            # 获取当前簇的所有边
            edges = cluster['TRAJECTORIES']['EDGE'].unique()

            # 如果边的数量小于或等于5条，进行进一步筛选
            if len(edges) <= 5:
                # 提取边上的每一个顶点坐标值
                points = []
                for edge in edges:
                    points.append(self.graph.graph_proj.nodes[edge[0]])
                    points.append(self.graph.graph_proj.nodes[edge[1]])

                # 转换为 DataFrame 以便排序和计算
                points_df = pd.DataFrame(points)

                # 按照经度和纬度排序两次
                lng_sorted = points_df.sort_values(by='x')
                lat_sorted = points_df.sort_values(by='y')

                # 比较两次最大与最小值的差
                lng_diff = lng_sorted['x'].max() - lng_sorted['x'].min()
                lat_diff = lat_sorted['y'].max() - lat_sorted['y'].min()

                # 如果有两个小于规定阈值，将该簇从最终结果排除
                if lng_diff < distance_threshold and lat_diff < distance_threshold:
                    continue

            # 保留符合条件的簇
            filtered_clusters[cluster_id] = cluster
        print(f"筛选后的簇数为：{len(filtered_clusters)},排除的簇数为：{len(self.clusters) - len(filtered_clusters)}")
        self.clusters = filtered_clusters


    def merge_check(self):
        """
        检测是否有连续的簇被分开，并将这些簇合并。

        返回:
        更新后的簇字典
        """
        # 创建一个字典，用于存储每个顶点所属的簇
        vertex_to_cluster = {}

        # 创建一个字典，用于存储需要合并的簇
        clusters_to_merge = {}

        # 遍历每个簇
        for cluster_id, cluster in self.clusters.items():
            # 获取当前簇的所有边
            edges = cluster['TRAJECTORIES']['EDGE'].unique()

            # 遍历每条边的顶点
            for edge in edges:
                start_node = edge[0]
                end_node = edge[1]

                # 检查顶点是否已经存在于字典中
                if start_node in vertex_to_cluster:
                    # 如果顶点已经存在于字典中，并且所属的簇不同，则记录需要合并的簇
                    if vertex_to_cluster[start_node] != cluster_id:
                        if vertex_to_cluster[start_node] not in clusters_to_merge:
                            clusters_to_merge[vertex_to_cluster[start_node]] = set()
                        clusters_to_merge[vertex_to_cluster[start_node]].add(cluster_id)
                else:
                    # 如果顶点不存在于字典中，则将其添加到字典中
                    vertex_to_cluster[start_node] = cluster_id

                if end_node in vertex_to_cluster:
                    if vertex_to_cluster[end_node] != cluster_id:
                        if vertex_to_cluster[end_node] not in clusters_to_merge:
                            clusters_to_merge[vertex_to_cluster[end_node]] = set()
                        clusters_to_merge[vertex_to_cluster[end_node]].add(cluster_id)
                else:
                    vertex_to_cluster[end_node] = cluster_id

        # 合并需要合并的簇
        for base_cluster, merge_clusters in clusters_to_merge.items():
            # 检查 base_cluster 是否存在
            if base_cluster not in self.clusters:
                continue
            for merge_cluster in list(merge_clusters):  # 使用 list() 以避免在迭代时修改集合
                # 检查 merge_cluster 是否存在
                if merge_cluster in self.clusters:
                    # 将 merge_cluster 的轨迹合并到 base_cluster
                    self.clusters[base_cluster]['TRAJECTORIES'] = pd.concat(
                        [self.clusters[base_cluster]['TRAJECTORIES'], self.clusters[merge_cluster]['TRAJECTORIES']]
                    )
                    self.clusters[base_cluster]['VEHICLE_COUNT'] += self.clusters[merge_cluster]['VEHICLE_COUNT']
                    # 删除 merge_cluster
                    del self.clusters[merge_cluster]
                else:
                    # 如果 merge_cluster 不存在，则从 clusters_to_merge 中移除
                    clusters_to_merge[base_cluster].remove(merge_cluster)

        print(f"合并后的簇数为：{len(self.clusters)}, 合并的簇数为：{len(clusters_to_merge)}")
        return self.clusters

    def save_pkl(self, output_path):
        """
        将聚类结果保存为 PKL 文件。

        参数:
        output_path -- 输出 PKL 文件的路径
        """
        with open(output_path, 'wb') as f:
            pickle.dump(self.clusters, f)
        print(f"聚类结果已保存到：{output_path}")

    def save_shp(self, output_path):
        """
        将聚类结果保存为 Shapefile 格式。

        参数:
        output_path -- 输出 Shapefile 文件的路径
        """
        # 创建一个列表，用于存储每个簇的几何和属性
        data = []

        # 遍历每个簇
        for cluster_id, cluster in self.clusters.items():
            # 获取当前簇的所有边
            edges = cluster['TRAJECTORIES']['EDGE'].unique()

            # 遍历每条边，创建 LineString 几何
            for edge in edges:
                try:
                    # 获取完整的路线
                    route = self.graph.graph_proj.edges[edge[0], edge[1], 0]['geometry']
                except KeyError:
                    # 如果找不到 'geometry' 键，创建一个简单的 LineString
                    start_node = self.graph.graph_proj.nodes[edge[0]]
                    end_node = self.graph.graph_proj.nodes[edge[1]]
                    route = LineString([(start_node['x'], start_node['y']), (end_node['x'], end_node['y'])])
                data.append({
                    'geometry': route,
                    'cluster_id': cluster_id,
                    'veh_count': cluster['VEHICLE_COUNT'],
                    'edge': str(edge)  # 将 edge 转换为字符串类型
                })

        # 创建 GeoDataFrame
        gdf = gpd.GeoDataFrame(data)

        # 设置坐标系为 3857
        gdf.set_crs(epsg=3857, inplace=True)

        # 保存为 Shapefile
        gdf.to_file(output_path, driver='ESRI Shapefile')
        print(f"Shapefile 文件已保存到：{output_path}")


    def calculate_silhouette_score(self):
        """
        计算全局的轮廓系数。

        返回:
        全局的轮廓系数
        """
        # 提取样本点和簇标签
        samples = []
        labels = []
        for cluster_id, cluster in self.clusters.items():
            trajectories = cluster['TRAJECTORIES']
            for _, row in trajectories.iterrows():
                edge = row['EDGE']
                start_node = self.graph.graph_proj.nodes[edge[0]]
                end_node = self.graph.graph_proj.nodes[edge[1]]
                samples.append((edge[0], start_node['x'], start_node['y'], cluster_id))
                samples.append((edge[1], end_node['x'], end_node['y'], cluster_id))

        # 去重处理，确保每个节点只记录一次
        samples = list(set(samples))

        # 分离样本点和簇标签
        sample_points = [(node, x, y) for node, x, y, _ in samples]
        labels = [cluster_id for _, _, _, cluster_id in samples]


        # 计算样本点之间的路网距离
        def road_network_distance(p1, p2):
            try:
                return nx.shortest_path_length(self.graph.graph_proj, source=p1, target=p2, weight='length')
            except nx.NetworkXNoPath:
                return np.inf

        # 计算距离矩阵
        distance_matrix = np.zeros((len(sample_points), len(sample_points)))
        for i, (node1, _, _) in enumerate(sample_points):
            for j, (node2, _, _) in enumerate(sample_points):
                if i != j:
                    distance_matrix[i, j] = road_network_distance(node1, node2)

        # 将无穷大值替换为距离矩阵中的最大有限值的两倍
        max_finite_distance = np.max(distance_matrix[np.isfinite(distance_matrix)])
        distance_matrix[np.isinf(distance_matrix)] = max_finite_distance * 2
        # 计算轮廓系数
        silhouette_avg = silhouette_score(distance_matrix, labels, metric='precomputed')
        print(f"全局的轮廓系数为：{silhouette_avg}")


    def calculate_dcsi(self):
        """
        计算密度聚类可分离性指数（DCSI）。

        返回:
        DCSI 值
        """
        # 提取样本点和簇标签
        samples = []
        labels = []
        for cluster_id, cluster in self.clusters.items():
            trajectories = cluster['TRAJECTORIES']
            for _, row in trajectories.iterrows():
                edge = row['EDGE']
                start_node = self.graph.graph_proj.nodes[edge[0]]
                end_node = self.graph.graph_proj.nodes[edge[1]]
                samples.append((edge[0], start_node['x'], start_node['y'], cluster_id))
                samples.append((edge[1], end_node['x'], end_node['y'], cluster_id))

        # 去重处理，确保每个节点只记录一次
        samples = list(set(samples))

        # 分离样本点和簇标签
        sample_points = [(node, x, y) for node, x, y, _ in samples]
        labels = [cluster_id for _, _, _, cluster_id in samples]

        # 计算样本点之间的路网距离
        def road_network_distance(p1, p2):
            try:
                return nx.shortest_path_length(self.graph.graph_proj, source=p1, target=p2, weight='length')
            except nx.NetworkXNoPath:
                return np.inf

        # 计算距离矩阵
        distance_matrix = np.zeros((len(sample_points), len(sample_points)))
        for i, (node1, _, _) in enumerate(sample_points):
            for j, (node2, _, _) in enumerate(sample_points):
                if i != j:
                    distance_matrix[i, j] = road_network_distance(node1, node2)

        # 将无穷大值替换为距离矩阵中的最大有限值的两倍
        max_finite_distance = np.max(distance_matrix[np.isfinite(distance_matrix)])
        distance_matrix[np.isinf(distance_matrix)] = max_finite_distance * 2

        # 计算簇内密度（Intra-cluster Density）
        intra_cluster_densities = []
        for cluster_id in set(labels):
            cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
            if len(cluster_indices) > 1:
                intra_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
                intra_cluster_density = np.mean(intra_distances)
                intra_cluster_densities.append(intra_cluster_density)

        # 计算簇间密度（Inter-cluster Density）
        inter_cluster_densities = []
        unique_labels = list(set(labels))
        for i, cluster_id1 in enumerate(unique_labels):
            for cluster_id2 in unique_labels[i+1:]:
                cluster_indices1 = [i for i, label in enumerate(labels) if label == cluster_id1]
                cluster_indices2 = [i for i, label in enumerate(labels) if label == cluster_id2]
                inter_distances = distance_matrix[np.ix_(cluster_indices1, cluster_indices2)]
                inter_cluster_density = np.mean(inter_distances)
                inter_cluster_densities.append(inter_cluster_density)

        # 计算 DCSI
        if intra_cluster_densities and inter_cluster_densities:
            avg_intra_cluster_density = np.mean(intra_cluster_densities)
            avg_inter_cluster_density = np.mean(inter_cluster_densities)
            dcsi = avg_inter_cluster_density / avg_intra_cluster_density
            print(f"DCSI 值为：{dcsi}")
            return dcsi
        else:
            print("无法计算 DCSI 值")
            return None

    def get_cluster_info(self):
        """
        获取聚类结果的统计信息并打印，包括：
        - 簇个数
        - 簇平均路网距离（每个簇内所有路段长度的平均值）
        - 全局簇平均密度（单簇的计算为平均所有路段边内的轨迹片段数，全局的为所有簇的平均）
        - 簇平均包含车辆数
        """
        if not self.clusters:
            print("当前没有聚类结果！")
            return
    
        # 簇个数
        num_clusters = len(self.clusters)
        print(f"簇个数：{num_clusters}")
    
        # 簇平均路网距离
        total_cluster_distance = 0
        for cluster in self.clusters.values():
            cluster_distance = 0
            edges = cluster['TRAJECTORIES']['EDGE'].unique()
            for edge in edges:
                try:
                    # 获取路段长度
                    length = self.graph.graph_proj.edges[edge[0], edge[1], 0]['length']
                except KeyError:
                    # 如果没有长度信息，跳过该边
                    continue
                cluster_distance += length
            total_cluster_distance += cluster_distance
        avg_cluster_distance = total_cluster_distance / num_clusters if num_clusters > 0 else 0
        print(f"簇平均路网距离：{avg_cluster_distance:.2f} 米")
    
        # 全局簇平均密度
        total_density = 0
        for cluster in self.clusters.values():
            edges = cluster['TRAJECTORIES']['EDGE'].unique()
            edge_density = []
            for edge in edges:
                edge_mask = cluster['TRAJECTORIES']['EDGE'] == edge
                edge_density.append(edge_mask.sum())  # 每条边的轨迹片段数
            cluster_density = np.mean(edge_density) if edge_density else 0
            total_density += cluster_density
        avg_density = total_density / num_clusters if num_clusters > 0 else 0
        print(f"全局簇平均密度：{avg_density:.2f} 轨迹片段/路段")
    
        # 簇平均包含车辆数
        total_vehicle_count = sum(cluster['VEHICLE_COUNT'] for cluster in self.clusters.values())
        avg_vehicle_count = total_vehicle_count / num_clusters if num_clusters > 0 else 0
        print(f"簇平均包含车辆数：{avg_vehicle_count:.2f} 辆")


    def plot_cluster_distance_distribution(self):
        """
        绘制聚类结果中簇的路网距离分布直方图。
        展示每个簇包含的所有路段长度总和的分布情况。
        """
        if not self.clusters:
            print("当前没有聚类结果！")
            return
    
        import matplotlib.pyplot as plt
        import numpy as np
    
        # 计算每个簇的路网总距离
        cluster_distances = []
        for cluster in self.clusters.values():
            cluster_distance = 0
            edges = cluster['TRAJECTORIES']['EDGE'].unique()
            for edge in edges:
                try:
                    # 获取路段长度
                    length = self.graph.graph_proj.edges[edge[0], edge[1], 0]['length']
                    cluster_distance += length
                except KeyError:
                    # 如果没有长度信息，跳过该边
                    continue
            cluster_distances.append(cluster_distance)
    
        # 创建直方图
        plt.figure(figsize=(10, 6))
        
        # 计算合适的区间数（使用 Sturges 规则）
        # n_bins = int(np.log2(len(cluster_distances))) + 1
        n_bins = 20
        # 绘制直方图
        plt.hist(cluster_distances, bins=n_bins, edgecolor='black')
        
        # 添加垂直线表示平均值
        mean_distance = np.mean(cluster_distances)
        plt.axvline(x=mean_distance, color='r', linestyle='--', 
                    label=f'Mean: {mean_distance:.2f}m')
        
        # Set chart title and labels
        plt.title('Network Distance Distribution of Clusters')
        plt.xlabel('Network Distance (meters)')
        plt.ylabel('Number of Clusters')
    
        
        # 添加网格
        plt.grid(True, alpha=0.3)
        
        # 添加图例
        plt.legend()
        
        # 优化布局
        plt.tight_layout()
        
        # 显示图表
        plt.show()
    
        # 打印一些基本统计信息
        print(f"路网距离统计信息：")
        print(f"最小值：{min(cluster_distances):.2f} 米")
        print(f"最大值：{max(cluster_distances):.2f} 米")
        print(f"平均值：{mean_distance:.2f} 米")
        print(f"中位数：{np.median(cluster_distances):.2f} 米")
        print(f"标准差：{np.std(cluster_distances):.2f} 米")

    def plot_minpts_distance_boxplot(self, simplified_traj, graph, start_minpts=3, end_minpts=13, step=2, edge_filter_threshold=500):
        """
        Plot boxplot of network distances distribution for different minpts values.
        不显示离群值版本。
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from copy import deepcopy
    
        # Store distances for each minpts value
        all_distances = []
        minpts_values = []
    
        # Iterate through different minpts values
        for minpts in range(start_minpts, end_minpts + 1, step):
            print(f"\nProcessing minpts = {minpts}")
            
            # Create new clustering instance
            cluster_instance = ParalClustering()
            
            # Perform clustering
            cluster_instance.cluster_traj_init(simplified_traj, graph, minpts)
            cluster_instance.edge_filter(edge_filter_threshold)
            cluster_instance.merge_check()
    
            # Calculate distances for current minpts
            cluster_distances = []
            for cluster in cluster_instance.clusters.values():
                cluster_distance = 0
                edges = cluster['TRAJECTORIES']['EDGE'].unique()
                for edge in edges:
                    try:
                        length = graph.graph_proj.edges[edge[0], edge[1], 0]['length']
                        cluster_distance += length
                    except KeyError:
                        continue
                cluster_distances.append(cluster_distance)
    
            if cluster_distances:  # Only add if we have clusters
                all_distances.append(cluster_distances)
                minpts_values.append(minpts)
    
        # Create boxplot
        plt.figure(figsize=(12, 6))
        
        # Create boxplot with custom style and no outliers
        bp = plt.boxplot(all_distances, 
                        patch_artist=True,
                        showfliers=False,  # 不显示离群点
                        whis=[0, 100])     # 将须的范围设置为最小值到最大值
        
        # Customize boxplot colors
        for box in bp['boxes']:
            box.set(facecolor='lightblue', alpha=0.7)
        plt.setp(bp['medians'], color='red')
        
        # Set labels and title
        plt.xlabel('MinPts Value')
        plt.ylabel('Network Distance (meters)')
        plt.title('Network Distance Distribution vs MinPts')
        
        # Set x-axis labels to minpts values
        plt.xticks(range(1, len(minpts_values) + 1), minpts_values)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Optimize layout
        plt.tight_layout()
        
        # Show plot
        plt.show()
    
        # Print statistics for each minpts value
        for i, minpts in enumerate(minpts_values):
            distances = all_distances[i]
            print(f"\nStatistics for MinPts = {minpts}:")
            print(f"Number of clusters: {len(distances)}")
            print(f"Mean distance: {np.mean(distances):.2f} meters")
            print(f"Median distance: {np.median(distances):.2f} meters")
            print(f"Std deviation: {np.std(distances):.2f} meters")