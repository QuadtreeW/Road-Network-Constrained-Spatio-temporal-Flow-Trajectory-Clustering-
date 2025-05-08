from Gmm import Graph, Trajectory
from FlowClustering import ParalClustering
from mode import PatternAnalysis
import pandas as pd
import osmnx as ox
import pickle





if __name__ == '__main__':

    graph = Graph()
    graph.Read_mapXMl("Shenzhen_LL.graphml") 

    # ------------------------------------路网匹配初始化-----------------------------------------------


    # traj = Trajectory()
    # traj.read_from_csv('25tr/0418_1555_10_1000.csv') 
    # traj.filter_GPS(0.0009556)

    # ----------------------------------------这个没有用-------------------------------------------

    # 设置时间窗口
    # start_time = pd.Timestamp('2022-01-01 00:00:00')  # 请将这个时间替换为你需要的开始时间
    # window_size = pd.Timedelta(hours=1)  # 这里设置的时间窗口大小为1小时
    # traj.set_time_window(start_time, window_size)

    # ------------------------------------路网匹配并保存-----------------------------------------------

    # simplified_traj = traj.get_simpl_traj_curr("Shenzhen_LL.graphml")
    # with open('25tr/simplified_traj1000_1555_curr.pkl', 'wb') as f:
    #     pickle.dump(simplified_traj, f)

    # ------------------------------------------加载匹配结果-----------------------------------------

    # 加载文件
    # with open('25tr/simplified_traj1000_2015_curr.pkl', 'rb') as f:
    # #     curr_simplified_dict = pickle.load(f)
    with open('25tr/simplified_traj1000_1505_curr.pkl', 'rb') as f:
        simplified_traj = pickle.load(f)

    # ------------------------------------------聚类开始与保存-----------------------------------------



    cluster = ParalClustering()

    # cluster.cluster_traj_init(simplified_traj, graph,5 )
    # cluster.edge_filter(500)
    # cluster.merge_check()
    # cluster.save_pkl("25tr/eps5/c1000_1555_5.pkl")
    # cluster.save_shp("shp/eps5/c1000_1555_5.shp")

    
    # cluster.calculate_silhouette_score()
    # cluster.calculate_dcsi()
    # cluster.get_cluster_info()
    # cluster.plot_cluster_distance_distribution()
    # 调用箱型图绘制函数
    cluster.plot_minpts_distance_boxplot(
        simplified_traj=simplified_traj,
        graph=graph,
        start_minpts=3,
        end_minpts=13,
        step=2,
        edge_filter_threshold=500
    )
     
    # ---------------------------------------加载前次聚类并更新--------------------------------------------


    # 加载已经保存的聚类结果
    # with open('25tr/eps5/c1000_2035.pkl', 'rb') as f:
    #     clusters = pickle.load(f)

    # cluster.update_cluster(simplified_traj, graph,  5,clusters)
    # cluster.edge_filter(500)
    # cluster.merge_check()
    # cluster.save_shp("shp/eps5/wtf.shp")
    # cluster.update_cluster_with_overlap_optimized(simplified_traj, graph,  10,clusters)
    # cluster.edge_filter(500)
    # clusters2 = cluster.merge_check()
    # with open('25tr/nbrt/clusters1000_2015_nbrt.pkl', 'wb') as f:
    #     pickle.dump(clusters2, f)

    # -------------------------------------结果可视化----------------------------------------------

    #cluster.visualize_clusters(title='20:05')






















# import osmnx as ox
# import pandas as pd
# import matplotlib.pyplot as plt

# # 读取地图
# graph = ox.load_graphml('Shenzhen_LL.graphml')
# graph_proj = ox.project_graph(graph)

# clusters = {
#     4: {
#         'TRAJECTORIES': pd.DataFrame({
#             'EDGE': [(10933736833, 10933736834), (3118474661, 3118474705), (2537806944, 1116458648), (1116417986, 5281881064), (2363414851, 2363414883)],
#             'VEHICLE_ID': [2472724, 2472724, 2471798, 2471798, 2471798],
#             'cluster_id': [4, 4, 4, 4, 4]
#         }),
#         'VEHICLE_COUNT': 50
#     }
# }

# # 创建一个颜色列表
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# # 显示图形，设置节点颜色为透明
# fig, ax = ox.plot_graph(graph_proj, node_color='none', edge_color='gray', show=False, close=False)

# for cluster_id, cluster in clusters.items():
#     # 获取当前簇的颜色
#     color = colors[cluster_id % len(colors)]

#     # 获取当前簇的所有边
#     edges = set()
#     for _, row in cluster['TRAJECTORIES'].iterrows():
#         edges.add(row['EDGE'])

#     # 在图形中绘制当前簇的边
#     for edge in edges:
#         # 确保边存在于图形中
#         if edge in graph_proj.edges:
#             # 添加默认的键
#             edge = (*edge, 0)
#             # 获取边的两个节点的坐标
#             x1, y1 = graph_proj.nodes[edge[0]]['x'], graph_proj.nodes[edge[0]]['y']
#             x2, y2 = graph_proj.nodes[edge[1]]['x'], graph_proj.nodes[edge[1]]['y']
#             ax.plot([x1, x2], [y1, y2], color=color)

# # 在图形上添加 minpts 值
# plt.text(0.5, 0, 'minpts: {}'.format(4), transform=plt.gca().transAxes, color='white')

# plt.show()