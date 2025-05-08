from FlowClustering import ParalClustering
from Gmm import Graph, Trajectory
import pickle
import geopandas as gpd
from shapely.geometry import LineString
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta


class PatternAnalysis:
    def __init__(self, pkl_files, graph:Graph):
        """
        初始化模式分析类
        :param pkl_files: 聚类结果的 PKL 文件列表
        :param graph: 图对象
        """
        self.clustering_results = self.load_clustering_results(pkl_files)
        self.graph = graph
        self.states = []

    def load_clustering_results(self, pkl_files):
        """
        从 PKL 文件中加载聚类结果
        :param pkl_files: 聚类结果的 PKL 文件列表
        :return: 聚类结果列表
        """
        clustering_results = []
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as f:
                result = pickle.load(f)
                if result is not None:
                    clustering_results.append(result)
        return clustering_results


    def analyze_patterns(self, target_edge):
        """
        分析所有前后聚类结果的演化模式
        :param target_edge: 指定的边，形态如 (11133300555, 1169609832),(2671929619, 3166459123)
        :return: 演化模式的状态列表
        """
        if len(self.clustering_results) < 2:
            print("聚类结果不足，无法进行模式分析")
            return []
    
        for i in range(1, len(self.clustering_results)):
            prev_result = self.clustering_results[i - 1]
            curr_result = self.clustering_results[i]
    
            # 找出前簇和后簇的所有边
            prev_edges, prev_cluster_ids = self.find_clusters_with_edge_in_result(prev_result, target_edge)
            curr_edges, curr_cluster_ids = self.find_clusters_with_edge_in_result(curr_result, target_edge)
    
            # 状态：出现或消失
            if not prev_cluster_ids and curr_cluster_ids:
                print(f"第{i}阶段：状态：出现")
                self.states.append("create")
                continue
            elif prev_cluster_ids and not curr_cluster_ids:
                print(f"第{i}阶段：状态：消失")
                self.states.append("disappear")
                continue
            elif not prev_cluster_ids and not curr_cluster_ids:
                print(f"第{i}阶段：前结果或后结果中未找到包含输入边的簇")
                self.states.append("no_change")
                continue
    
            # 获取前簇的所有边
            prev_all_edges = self.get_all_edges_from_clusters(prev_result, prev_cluster_ids)
    
            # 检查前簇的边在当前阶段的簇分布
            edge_to_cluster_map = {}
            for edge in prev_all_edges:
                for cluster_id, cluster in curr_result.items():
                    edges = cluster['TRAJECTORIES']['EDGE'].values
                    if any(np.array_equal(edge, e) for e in edges) or any(np.array_equal((edge[1], edge[0]), e) for e in edges):
                        if edge not in edge_to_cluster_map:
                            edge_to_cluster_map[edge] = set()
                        edge_to_cluster_map[edge].add(cluster_id)
    
            # 获取所有后簇的 ID
            involved_clusters = set()
            for clusters in edge_to_cluster_map.values():
                involved_clusters.update(clusters)
    
            # print(f"第{i}阶段：")
            # print(f"前簇的所有边：{prev_all_edges}")
            # print(f"边到后簇的映射：{edge_to_cluster_map}")
            # print(f"涉及的后簇：{involved_clusters}")
    
            # 判断分裂和合并
            if len(prev_cluster_ids) == 1 and len(involved_clusters) > 1:
                print(f"第{i}阶段：状态：分裂")
                self.states.append("split")
            elif len(involved_clusters) == 1:  # 当前阶段只有一个簇
                # 检查当前簇的边是否来自前一个阶段的多个簇
                curr_cluster_id = list(involved_clusters)[0]
                curr_cluster_edges = self.get_all_edges_from_clusters(curr_result, [curr_cluster_id])
    
                # 找到当前簇的边在前一个阶段的簇分布
                edge_to_prev_cluster_map = {}
                for edge in curr_cluster_edges:
                    for cluster_id, cluster in prev_result.items():
                        edges = cluster['TRAJECTORIES']['EDGE'].values
                        if any(np.array_equal(edge, e) for e in edges) or any(np.array_equal((edge[1], edge[0]), e) for e in edges):
                            if edge not in edge_to_prev_cluster_map:
                                edge_to_prev_cluster_map[edge] = set()
                            edge_to_prev_cluster_map[edge].add(cluster_id)
    
                # 获取前一个阶段涉及的簇数量
                involved_prev_clusters = set()
                for clusters in edge_to_prev_cluster_map.values():
                    involved_prev_clusters.update(clusters)
    
                if len(involved_prev_clusters) > 1:
                    print(f"第{i}阶段：状态：合并")
                    self.states.append("merge")

                elif len(prev_cluster_ids) == 1 and len(involved_clusters) == 1:
                    # 判断收缩和扩展状态
                    curr_all_edges = self.get_all_edges_from_clusters(curr_result, list(involved_clusters))
                    if prev_all_edges.issuperset(curr_all_edges):
                        print(f"第{i}阶段：状态：收缩")
                        self.states.append("shrink")
                    elif curr_all_edges.issuperset(prev_all_edges):
                        print(f"第{i}阶段：状态：扩展")
                        self.states.append("expand")
                    else:
                        # 计算前后簇的密度
                        prev_density = self.calculate_cluster_density(prev_result, prev_cluster_ids[0])
                        curr_density = self.calculate_cluster_density(curr_result, list(involved_clusters)[0])
                        if curr_density > prev_density:
                            print(f"第{i}阶段：状态：扩展")
                            self.states.append("expand")
                        else:
                            print(f"第{i}阶段：状态：收缩")
                            self.states.append("shrink")
                else:
                    print(f"第{i}阶段：状态：无变化")
                    self.states.append("no_change")
        

    def find_clusters_with_edge_in_result(self, result, target_edge):
        """
        找出指定聚类结果中包含指定边的簇，并返回这些簇的所有边和簇的 ID 列表
        :param result: 聚类结果
        :param target_edge: 指定的边，形态如 (11133300555, 1169609832),(277049983, 1768207605),(2671929619, 3166459123)
        :return: 包含指定边的簇的所有边的集合和簇的 ID 列表
        """
        all_edges = set()
        cluster_ids = []
        for cluster_id, cluster in result.items():
            edges = cluster['TRAJECTORIES']['EDGE'].values
            if any(np.array_equal(target_edge, edge) for edge in edges) or any(np.array_equal((target_edge[1], target_edge[0]), edge) for edge in edges):
                unique_edges = cluster['TRAJECTORIES']['EDGE'].unique()
                all_edges.update(unique_edges)
                cluster_ids.append(cluster_id)
        return all_edges, cluster_ids
    

    def get_all_edges_from_clusters(self, result, cluster_ids):
        """
        获取指定聚类结果中指定簇的所有边
        :param result: 聚类结果
        :param cluster_ids: 簇的 ID 列表
        :return: 所有边的集合
        """
        all_edges = set()
        for cluster_id in cluster_ids:
            cluster = result.get(cluster_id)
            if cluster:
                edges = cluster['TRAJECTORIES']['EDGE'].unique()
                all_edges.update(edges)
        return all_edges

    def check_edges_belong_to_clusters(self, all_edges, prev_cluster_ids, curr_cluster_ids):
        """
        检查所有边是否只属于前簇和后簇的两种簇 ID
        :param all_edges: 所有边的集合
        :param prev_cluster_ids: 前簇的 ID 列表
        :param curr_cluster_ids: 后簇的 ID 列表
        :return: 是否只包含前簇和后簇的两种簇 ID
        """
        valid_cluster_ids = set(prev_cluster_ids).union(set(curr_cluster_ids))
        for edge in all_edges:
            edge_clusters = set()
            for result in self.clustering_results:
                for cluster_id, cluster in result.items():
                    edges = cluster['TRAJECTORIES']['EDGE'].values
                    if any(np.array_equal(edge, e) for e in edges) or any(np.array_equal((edge[1], edge[0]), e) for e in edges):
                        edge_clusters.add(cluster_id)
            if not edge_clusters.issubset(valid_cluster_ids):
                return False
        return True

    def calculate_cluster_density(self, clustering_result, cluster_id):
        """
        计算指定簇的密度（所有边的车辆数量和除以所有簇边的路网长度）
        :param clustering_result: 聚类结果
        :param cluster_id: 簇的 ID
        :return: 簇的密度
        """
        cluster = clustering_result.get(cluster_id)
        if not cluster:
            return 0
    
        total_length = 0
        total_vehicle_count = 0
        for edge in cluster['TRAJECTORIES']['EDGE'].unique():
            try:
                # 获取边的长度
                length = self.graph.graph_proj.edges[edge[0], edge[1], 0]['length']
                # 计算该边的车辆数量
                vehicle_count = cluster['TRAJECTORIES'][cluster['TRAJECTORIES']['EDGE'] == edge]['VEHICLE_ID'].nunique()
            except KeyError:
                # 如果找不到 'length' 键，计算简单的欧几里得距离
                start_node = self.graph.graph_proj.nodes[edge[0]]
                end_node = self.graph.graph_proj.nodes[edge[1]]
                length = ((start_node['x'] - end_node['x'])**2 + (start_node['y'] - end_node['y'])**2)**0.5
                vehicle_count = cluster['TRAJECTORIES'][cluster['TRAJECTORIES']['EDGE'] == edge]['VEHICLE_ID'].nunique()
            total_length += length
            total_vehicle_count += vehicle_count
    
        if total_length == 0:
            return 0
    
        return total_vehicle_count / total_length




#------------------------------------------------------------------------------------------------------------

    def visualize_patterns(self):
        """
        可视化聚类结果中的模式
        :param self.states: 演化模式的状态列表
        """
        # 定义状态的顺序和对应的数值
        state_order = ["disappear", "split", "shrink", "merge", "expand", "create"]
        state_values = {state: i for i, state in enumerate(state_order)}
    
        # 将状态转换为数值，跳过 "no_change" 状态
        y_values = [state_values[state] for state in self.states if state != "no_change"]
        x_values = [i + 1 for i, state in enumerate(self.states) if state != "no_change"]
    
        # 创建折线图
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
    
        # 设置横坐标和纵坐标标签
        plt.xticks(range(1, len(self.states) + 1))
        plt.yticks(range(len(state_order)), state_order)
    
        # 设置标题和轴标签
        plt.title("Evolution Patterns of Clustering Results")
        plt.xlabel("Stage")
        plt.ylabel("State")
    
        # 显示图表
        plt.grid(True)
        plt.show()


    def summarize_clusters(self):
        """
        统计每个聚类结果中的簇的个数以及每个簇中包含的车辆数量
        """
        summary = []
        for result in self.clustering_results:
            if result is not None:
                num_clusters = len(result)  # 簇的个数
                cluster_sizes = [cluster['VEHICLE_COUNT'] for cluster in result.values()]  # 每个簇中包含的车辆数量
                summary.append({
                    'num_clusters': num_clusters,
                    'cluster_sizes': cluster_sizes
                })
                print(f"聚类结果: {result}")
                print(f"簇的个数: {num_clusters}")
                print(f"每个簇中包含的车辆数量: {cluster_sizes}\n")
        return summary 

    def find_clusters_with_edge(self, target_edge):
        """
        找出所有聚类结果中包含指定边的簇，并返回这些簇的所有边和簇的 ID 列表
        :param target_edge: 指定的边，形态如 (11133300555, 1169609832)
        :return: 包含指定边的簇的所有边的集合和簇的 ID 列表
        """
        all_edges = set()
        cluster_ids = []
        for result in self.clustering_results:
            for cluster_id, cluster in result.items():
                if target_edge in cluster['TRAJECTORIES']['EDGE'].values:
                    edges = cluster['TRAJECTORIES']['EDGE'].unique()
                    print(f"聚类结果中包含边 {target_edge} 的簇 ID: {cluster_id}")
                    print(f"簇 {cluster_id} 的所有边: {cluster['TRAJECTORIES']['EDGE'].unique()}\n")
                    all_edges.update(edges)
                    cluster_ids.append(cluster_id)
        return all_edges, cluster_ids

    def save_edges_to_shp(self, edges, output_path):
            """
            将边集合保存为 Shapefile 格式
            :param edges: 边的集合
            :param output_path: 输出 Shapefile 文件的路径
            """
            data = []
            for edge in edges:
                try:
                    # 获取完整的路线
                    route = self.graph.graph_proj.edges[edge[0], edge[1], 0]['geometry']
                except KeyError:
                    # 如果找不到 'geometry' 键，创建一个简单的 LineString
                    start_node = self.graph.graph_proj.nodes[edge[0]]
                    end_node = self.graph.graph_proj.nodes[edge[1]]
                    route = LineString([(start_node['x'], start_node['y']), (end_node['x'], end_node['y'])])
                data.append({'geometry': route, 'edge': str(edge)})

            gdf = gpd.GeoDataFrame(data)
            gdf.set_crs(epsg=3857, inplace=True)
            gdf.to_file(output_path, driver='ESRI Shapefile')
            print(f"Shapefile 文件已保存到：{output_path}")

    def calculate_cluster_size(self, clustering_result, cluster_id):
        """
        计算指定簇的大小（所有边的路网长度之和）
        :param clustering_result: 聚类结果
        :param cluster_id: 簇的 ID
        :return: 簇的大小（路网长度之和）
        """
        cluster = clustering_result.get(cluster_id)
        if not cluster:
            return 0

        total_length = 0
        for edge in cluster['TRAJECTORIES']['EDGE'].unique():
            try:
                # 获取边的长度
                length = self.graph.graph_proj.edges[edge[0], edge[1], 0]['length']
            except KeyError:
                # 如果找不到 'length' 键，计算简单的欧几里得距离
                start_node = self.graph.graph_proj.nodes[edge[0]]
                end_node = self.graph.graph_proj.nodes[edge[1]]
                length = ((start_node['x'] - end_node['x'])**2 + (start_node['y'] - end_node['y'])**2)**0.5
            total_length += length

        return total_length



    def generate_sankey_json(self, target_edge, start_time, interval):
        """
        生成用于桑基图的 JSON 数据，使用时间作为每阶段的命名
        :param target_edge: 指定的边，形态如 (11133300555, 1169609832)，(277049983, 1768207605)
        :param start_time: 开始时间，格式为 "HH:MM"
        :param interval: 每个阶段的时间间隔，单位为分钟
        :return: JSON 数据
        """

    
        # 将输入的时间字符串转换为 datetime 对象
        current_time = datetime.strptime(start_time, "%H:%M")
    
        data = []
        links = []
        cluster_map = {}  # 用于存储每个簇的名称和出现次数
        cluster_edges = {}  # 用于存储每个簇的边
        all_previous_clusters = []  # 存储所有过去阶段的簇
    
        # 遍历每个聚类结果，找到包含指定边的簇
        for i, result in enumerate(self.clustering_results):
            # 使用时间作为阶段命名
            stage_time = current_time.strftime("%H:%M")
            current_time += timedelta(minutes=interval)  # 增加时间间隔
    
            edges, cluster_ids = self.find_clusters_with_edge_in_result(result, target_edge)
    
            for cluster_id in cluster_ids:
                cluster_name = f"{stage_time}"  # 仅使用时间作为簇的名称
                if cluster_name not in cluster_map:
                    cluster_map[cluster_name] = 0
                    cluster_edges[cluster_name] = set()
                cluster_map[cluster_name] += 1
                cluster_edges[cluster_name].update(edges)
    
        # 生成 data 部分
        for cluster_name in cluster_map.keys():
            data.append({"name": cluster_name})
    
        # 生成 links 部分
        current_time = datetime.strptime(start_time, "%H:%M")  # 重置时间
        for i in range(len(self.clustering_results)):
            # 使用时间作为阶段命名
            stage_time = current_time.strftime("%H:%M")
            current_time += timedelta(minutes=interval)  # 增加时间间隔
    
            current_stage_clusters = [name for name in cluster_map.keys() if name == stage_time]
    
            for prev_cluster in all_previous_clusters:  # 遍历所有之前的簇
                for curr_cluster in current_stage_clusters:
                    common_edges = self.get_common_edges(cluster_edges[prev_cluster], cluster_edges[curr_cluster])
                    if common_edges:
                        links.append({
                            "source": prev_cluster,
                            "target": curr_cluster,
                            "value": len(common_edges)
                        })
    
            all_previous_clusters.extend(current_stage_clusters)  # 记录所有过去的簇
    
        # 返回 JSON 数据
        print(json.dumps({"data": data, "links": links}, indent=2))
        return json.dumps({"data": data, "links": links}, indent=2)
    


    def sankey_json(self, target_edge, start_time, interval):
        """
        生成用于桑基图的 JSON 数据，使用时间作为每阶段的命名。
        [修改后逻辑] 对当前阶段的每条边，向前查找最近的源阶段，并统计来源。
        值为转移的路段的路网长度总和。
    
        :param target_edge: 指定的边，形态如 (11133300555, 1169609832)，(277049983, 1768207605)
        :param start_time: 开始时间，格式为 "HH:MM"
        :param interval: 每个阶段的时间间隔，单位为分钟
        :return: JSON 数据
        """
        from datetime import datetime, timedelta
        import json
        import numpy as np # 确保导入
    
        # --- 第一阶段：识别节点并聚合边集 (保持不变) ---
        current_time = datetime.strptime(start_time, "%H:%M")
    
        data = []
        # links 列表将在第二阶段重新生成
        cluster_map = {}
        cluster_edges = {} # { "HH:MM": set_of_edges }
        # 记录实际包含目标边的阶段名称列表，按时间顺序
        valid_stage_node_names = []
    
        for i, result in enumerate(self.clustering_results):
            stage_time = current_time.strftime("%H:%M")
            current_time += timedelta(minutes=interval)
    
            edges_in_stage_for_target, cluster_ids = self.find_clusters_with_edge_in_result(result, target_edge)
    
            if cluster_ids:
                cluster_name = f"{stage_time}"
                valid_stage_node_names.append(cluster_name) # 记录有效阶段名称
                if cluster_name not in cluster_map:
                    cluster_map[cluster_name] = 0
                    cluster_edges[cluster_name] = set()
                cluster_map[cluster_name] += len(cluster_ids)
                hashable_edges = set(tuple(e) if isinstance(e, (list, np.ndarray)) else e for e in edges_in_stage_for_target)
                cluster_edges[cluster_name].update(hashable_edges)
    
        # 生成 data 部分 (基于找到的有效阶段节点)
        for cluster_name in valid_stage_node_names: # 仅为有效阶段创建节点
            data.append({"name": cluster_name})
    
        # --- 第二阶段：生成 links 部分 (修改后逻辑 - 查找最近来源) ---
        links = [] # 重置 links 列表
        print("\n--- Sankey Pass 2: Creating Links (Finding Nearest Source for each edge) ---")
    
        # 遍历所有有效的时间阶段节点，从第二个开始 (索引 i=1)
        for i in range(1, len(valid_stage_node_names)):
            curr_node_name = valid_stage_node_names[i]
            curr_edges = cluster_edges[curr_node_name]
    
            # 存储从哪个前序节点流向当前节点的边的路网长度 {prev_node_name: total_length}
            source_stage_lengths = {}
    
            # 对当前阶段的每一条边，向前查找最近的来源
            for edge in curr_edges:
                edge_tuple = tuple(edge) if isinstance(edge, (list, np.ndarray)) else edge
                # 检查反向边，如果 get_common_edges 考虑了无向，这里也应该考虑
                reverse_edge_tuple = tuple(reversed(edge_tuple))
                found_source = False
    
                # 从紧邻的前一个有效阶段开始向前搜索 (j 从 i-1 到 0)
                for j in range(i - 1, -1, -1):
                    prev_node_name = valid_stage_node_names[j]
                    prev_edges = cluster_edges[prev_node_name]
    
                    # 检查边或其反向是否存在于前一个阶段的边集合中
                    if edge_tuple in prev_edges or reverse_edge_tuple in prev_edges:
                        # 计算该边的路网长度
                        try:
                            # 获取边的长度
                            length = self.graph.graph_proj.edges[edge_tuple[0], edge_tuple[1], 0]['length']
                        except KeyError:
                            try:
                                # 尝试反向边
                                length = self.graph.graph_proj.edges[reverse_edge_tuple[0], reverse_edge_tuple[1], 0]['length']
                            except KeyError:
                                # 如果找不到长度信息，计算简单的欧几里得距离
                                start_node = self.graph.graph_proj.nodes[edge_tuple[0]]
                                end_node = self.graph.graph_proj.nodes[edge_tuple[1]]
                                length = ((start_node['x'] - end_node['x'])**2 + (start_node['y'] - end_node['y'])**2)**0.5
    
                        # 累加到对应源节点的总路网长度
                        source_stage_lengths[prev_node_name] = source_stage_lengths.get(prev_node_name, 0) + length
                        found_source = True
                        # 找到最近的就停止向前搜索该边
                        break
    
            # 根据统计的来源路网长度，生成链接
            print(f"Linking to Stage {i} ({curr_node_name}): Sources found: {source_stage_lengths}")
            for source_node, length in source_stage_lengths.items():
                if length > 0: # 只有当确实有边从该来源过来时才创建链接
                    links.append({
                        "source": source_node,
                        "target": curr_node_name,
                        "value": length # 值是来自该特定最近来源的边的路网长度总和
                    })
                    print(f"  Link: {source_node} -> {curr_node_name} (Value: {length})")
    
    
        # --- 返回 JSON 数据 (保持不变) ---
        print(json.dumps({"data": data, "links": links}, indent=2))
        return json.dumps({"data": data, "links": links}, indent=2)

    def get_common_edges(self, edges1, edges2):
        """
        获取两个边集合之间的共有边，考虑无向匹配。
        :param edges1: 第一个簇的边集合
        :param edges2: 第二个簇的边集合
        :return: 共有边的集合
        """
        common_edges = set()
        for edge in edges1:
            if edge in edges2 or (edge[1], edge[0]) in edges2:  # 考虑无向匹配
                common_edges.add(edge)
        return common_edges

    


        
# 示例用法
if __name__ == "__main__":
    # 假设已经有聚类结果的 PKL 文件列表
    pkl_files = [
        "25tr/eps5/c1000_2005.pkl",
        "25tr/eps5/c1000_2015.pkl",
        "25tr/eps5/c1000_2025.pkl",
        "25tr/eps5/c1000_2035.pkl",
        "25tr/eps5/c1000_2045.pkl",
        "25tr/eps5/c1000_2055.pkl",
        # "25tr/eps5/c1000_1505.pkl",
        # "25tr/eps5/c1000_1515.pkl",
        # "25tr/eps5/c1000_1525.pkl",
        # "25tr/eps5/c1000_1535.pkl",
        # "25tr/eps5/c1000_1545.pkl",
        # "25tr/eps5/c1000_1555.pkl",
        

    ]

    # pkl_files = [
    #     "25tr/clusters1000_2005_curr.pkl",
    #     "25tr/clusters1000_2015_curr.pkl",
    #     "25tr/clusters1000_2025_curr.pkl",
    #     "25tr/clusters1000_2035_curr.pkl",
    #     "25tr/clusters1000_2045_curr.pkl",
    #     "25tr/clusters1000_2055_curr.pkl",
    # ]
    
    graph = Graph() 
    graph.Read_mapXMl("Shenzhen_LL.graphml") 

    # 创建模式分析对象
    pattern_analysis = PatternAnalysis(pkl_files, graph)

    # 统计簇的个数以及每个簇中包含的车辆数量
    # summary = pattern_analysis.summarize_clusters()

    # 找出所有聚类结果中包含指定边的簇，并打印这些簇的所有边
    target_edge = (9722598873, 9722598874)
    # pattern_analysis.find_clusters_with_edge(target_edge)
    # pattern_analysis.analyze_patterns(target_edge)
    # pattern_analysis.visualize_patterns()
    pattern_analysis.sankey_json(target_edge,"15:05", 10)
