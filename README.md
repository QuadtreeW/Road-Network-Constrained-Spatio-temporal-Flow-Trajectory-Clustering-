# Road Network Constraine -Spatio-temporal Flow Trajectory Clustering

# RoadTimeStream - 路网时空流模式分析系统


RoadTimeStream — a Python toolkit for analyzing spatio‑temporal flow patterns on urban road networks. It integrates map matching, edge‑based density clustering, basic evolution pattern analysis, and simple visualization.

Key idea: snap raw GPS tracks to the road graph, convert tracks to sequences of edges, then cluster frequent flows on adjacent edges.

## Features
- Map matching: Leuven Map Matching (HMM) over an in‑memory OSM graph
- Edge‑based density clustering: DBSCAN‑style expansion on neighboring edges with MinPts
- Post‑processing:
  - Small‑area filtering
  - Merge check for split clusters that share nodes
  - Basic quality metrics (silhouette score, DCSI)
- Pattern analysis (create/disappear/split/merge/shrink/expand) and Sankey JSON export for ECharts
- Quick visualization on top of OSMnx

## Dependencies
- pandas, numpy
- networkx, osmnx, geopandas, shapely, pyproj
- leuvenmapmatching
- matplotlib
- scikit‑learn, scipy
- geopy (used in utilities)

Install (example):
```bash
pip install pandas numpy networkx osmnx geopandas shapely pyproj leuvenmapmatching matplotlib scikit-learn scipy geopy
```

## Data
- Trajectories (CSV):
  - TRACK_ID, VEHICLE_ID, LNG, LAT, GPS_TIME
- Road network (GraphML):
  - WGS84 lon/lat GraphML (e.g., exported via OSMnx). Internally re‑projected to EPSG:3857.

## Quickstart
Python API aligned with the repository code (Gmm.py, FlowClustering.py, mode.py).

Map matching and simplified trajectories:
```python
from Gmm import Graph, Trajectory

# 1) Load road network (GraphML with lon/lat)
graph = Graph()
graph.Read_mapXMl("Shenzhen_LL.graphml")

# 2) Load trajectories
traj = Trajectory()
traj.read_from_csv("data/trajectory.csv")

# Optional: filter near‑stationary vehicles by latitude delta
traj.filter_GPS(0.0009556)

# 3) Parallel map matching (each worker loads the graph file)
simplified_traj = traj.get_simpl_traj_curr("Shenzhen_LL.graphml")
```

Clustering and post‑processing:
```python
from FlowClustering import ParalClustering

cluster = ParalClustering()
# minpts = minimum number of trajectory segments on the same edge
clusters = cluster.cluster_traj_init(simplified_traj, graph, minpts=5)

# Optional filters and merge check
cluster.edge_filter(distance_threshold=500)  # approx meters in EPSG:3857
cluster.merge_check()

# Metrics / summary
cluster.calculate_silhouette_score()  # optional, can be slow on large graphs
cluster.calculate_dcsi()              # optional
cluster.get_cluster_info()

# Save results
cluster.save_shp("outputs/clusters.shp")
# cluster.save_pkl("outputs/clusters.pkl")
```

Basic pattern analysis and Sankey JSON:
```python
from mode import PatternAnalysis
from Gmm import Graph

# Suppose you have multiple clustering PKLs over time windows
pkl_files = [

    # ...
]

g = Graph()
g.Read_mapXMl("Shenzhen_LL.graphml")

pa = PatternAnalysis(pkl_files, g)
target_edge = (9722598873, 9722598874)  # example edge (u, v)
# pa.analyze_patterns(target_edge)
# pa.visualize_patterns()

# Export Sankey JSON (nodes named by time)
json_str = pa.sankey_json(target_edge, start_time="15:05", interval=10)
# Use the output in ECharts Sankey
```

Quick visualization (optional):
```python
# After clustering
cluster.visualize_clusters(title="Clustered Flows")
```

## Screenshots
![Cluster example](https://github.com/user-attachments/assets/8d65e330-054f-4a2e-b850-96e2f65f4609)
![Pattern example](https://github.com/user-attachments/assets/941d7fae-c930-4270-ab1e-59911536b98e)
![Sankey example](https://github.com/user-attachments/assets/32f7cfa9-daac-4f72-be01-3df8c0382bbd)

## Repository Layout
```
Gmm.py              # Map loading (OSMnx), in‑memory map for matching, Trajectory I/O and parallel matching
FlowClustering.py   # Edge‑based clustering, filters, merge check, metrics, export
mode.py             # Pattern analysis (create/merge/.../disappear), Sankey JSON
test.py             # Simple tests/examples
```

## Notes and tips
- Read_mapXMl expects a GraphML stored in lon/lat; the code re‑projects to EPSG:3857 internally.
- edge_filter uses bounding‑box spread in projected units (≈ meters in EPSG:3857).
- get_simpl_traj_curr runs parallel map matching by letting each worker load the GraphML; ensure the path is accessible.
- For larger data, start with higher minpts (e.g., 7–9) to reduce noise.

## License and Usage
Copyright © 2025 QuadtreeW

This code is provided for academic research and educational purposes only.
Any form of commercial use is strictly prohibited, including but not limited to use in commercial projects, products, services, or any paid system.

## Acknowledgments
- OpenStreetMap contributors and OSMnx
- Leuven Map Matching library
