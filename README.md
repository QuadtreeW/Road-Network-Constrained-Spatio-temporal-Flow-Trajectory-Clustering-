# Road Network Constraine -Spatio-temporal Flow Trajectory Clustering

# RoadTimeStream - 路网时空流模式分析系统

## 项目概述
RoadTimeStream 是一个基于Python的路网时空流模式分析系统,主要用于分析城市路网中车辆轨迹的时空演化模式。系统集成了地图匹配、轨迹聚类和模式识别等功能模块。

## 系统架构
系统由以下核心模块构成:

```
RoadTimeStream/
├── Gmm.py              # 地图匹配模块
├── FlowClustering.py   # 轨迹聚类模块  
├── mode.py             # 模式分析模块
├── test.py             # 测试用例
```

## 主要功能

### 1. 地图匹配 (Map Matching)
- GPS轨迹点与路网匹配
- 轨迹简化与过滤
- 支持时间窗口设置

### 2. 轨迹聚类 (Trajectory Clustering)
- 基于密度的并行聚类算法
- 边过滤机制
- 聚类合并优化
- 聚类质量评估(轮廓系数等)

### 3. 模式分析 (Pattern Analysis) 
支持识别以下6种基本演化模式:
- Create (出现)
- Disappear (消失) 
- Split (分裂)
- Merge (合并)
- Shrink (收缩)
- Expand (扩展)

### 4. 可视化 (Visualization)
- 聚类结果可视化
- 演化模式可视化 
- Sankey图展示时序演化



主要依赖包:
- pandas
- numpy
- networkx
- osmnx
- geopandas
- matplotlib
- leuvenmapmatching

### 数据要求
- 轨迹数据(CSV格式):
  - TRACK_ID: 轨迹ID
  - VEHICLE_ID: 车辆ID
  - LNG/LAT: 经纬度
  - GPS_TIME: 时间戳
- 路网数据: GraphML格式

## 快速开始

### 1. 地图匹配
```python
from Gmm import Graph, Trajectory

# 初始化路网
graph = Graph()
graph.Read_mapXMl("Shenzhen_LL.graphml")

# 轨迹匹配
traj = Trajectory()
traj.read_from_csv('data/trajectory.csv')
traj.filter_GPS(0.0009556)
simplified_traj = traj.get_simpl_traj_curr("Shenzhen_LL.graphml")
```

### 2. 轨迹聚类
```python
from FlowClustering import ParalClustering

cluster = ParalClustering()
cluster.cluster_traj_init(simplified_traj, graph, eps=5)
cluster.edge_filter(500)
cluster.merge_check()
```
![m5](https://github.com/user-attachments/assets/8d65e330-054f-4a2e-b850-96e2f65f4609)



### 3. 模式分析
```python
from mode import PatternAnalysis

pattern_analysis = PatternAnalysis(pkl_files, graph)
pattern_analysis.analyze_patterns(target_edge)
pattern_analysis.visualize_patterns()
```
![dcw_n_5_5](https://github.com/user-attachments/assets/941d7fae-c930-4270-ab1e-59911536b98e)
![image](https://github.com/user-attachments/assets/32f7cfa9-daac-4f72-be01-3df8c0382bbd)



## 性能优化
- 并行聚类算法提高计算效率
- 边过滤减少噪声影响
- 聚类合并优化结果质量






Copyright © 2025 QuadtreeW

本代码仅供学术研究和学习使用。禁止任何形式的商业用途，包括但不限于将本代码用于商业项目、产品、服务或收费系统。
This code is provided for academic research and educational purposes only.  
Any form of commercial use is strictly prohibited, including but not limited to use in commercial projects, products, services, or any system involving payment.


