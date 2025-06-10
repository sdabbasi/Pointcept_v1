import open3d as o3d
import numpy as np

merged_pcd = o3d.geometry.PointCloud()
grid_size = 4  # 4x4 grid
spacing = 60.0  # distance between views in the grid (adjust as needed)

for idx in range(16):
    row = idx // grid_size
    col = idx % grid_size

    # Load the point cloud
    filename = f"teacher_pca_up{idx + 1}.ply"
    pcd = o3d.io.read_point_cloud(f'tools/ply_vis/{filename}')

    # Compute translation vector
    translation = np.array([col * spacing, -row * spacing, 0])  # arrange on XY grid
    pcd.translate(translation)

    # Merge into one cloud
    merged_pcd += pcd

# Save merged point cloud
o3d.io.write_point_cloud("tools/ply_vis/merged_4x4.ply", merged_pcd)
print("Saved merged_4x4.ply")