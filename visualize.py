import open3d as o3d
import numpy as np

# Load the point cloud bin file

pc_path = "/mnt/e/PCSeg/dataset/sequences/08/velodyne/000020.bin"
raw_data = np.fromfile(pc_path, dtype=np.float32).reshape((-1, 4))
# pcd = o3d.io.read_point_cloud(raw_data)

# print(pcd.shape)

xyz = raw_data[:, : 3]

# print(pcd.shape)
# # Visualize the point cloud
# o3d.visualization.draw_geometries([pcd])


# Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
o3d.io.write_point_cloud("sync.ply", pcd)


 # Load saved point cloud and visualize it
pcd_load = o3d.io.read_point_cloud("sync.ply")
o3d.visualization.draw_geometries([pcd_load])