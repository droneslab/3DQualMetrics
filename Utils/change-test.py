import open3d as o3d
import numpy as np
from sklearn.neighbors import KDTree
import sys
import argparse
import tqdm
import IROSMetric as iros

def compute_change(pcdA, pcdB, eps):
    # pcdA: open3d.geometry.PointCloud
    # pcdB: open3d.geometry.PointCloud
    # eps: float

    # Convert point clouds to numpy arrays
    pcdA_np = np.asarray(pcdA.points) # Reference : scan with chair
    pcdB_np = np.asarray(pcdB.points) # Candidate : scan without chair

    # Build KDTree for pcdA
    treeA = KDTree(pcdA_np)
    treeB = KDTree(pcdB_np)

    # Find points in pcdA that are not in pcdB based on the threshold eps
    distA, indA = treeB.query(pcdA_np, k=1)
    validAind = indA[distA<eps]
    invalidAind = indA[distA>=eps]
    print (f"Number of valid points in pcdA: {len(validAind)}")
    print (f"Number of invalid points in pcdA: {len(invalidAind)}")
    print (f"Total number of points in pcdA: {len(pcdA_np)}")
    # Color point cloud
    pcdA_colors = np.zeros((len(pcdA.points), 3))
    pcdA_colors[validAind] = [0, 1, 0] # Green
    # pcdA_colors[invalidAind] = [1, 0, 0] # Red
    pcdA.colors = o3d.utility.Vector3dVector(pcdA_colors)

    return pcdA


def sector_split(pcd, horiz_sectors=8, vert_sectors=1):
    # pcd: open3d.geometry.PointCloud
    
    # Convert point cloud to numpy array
    pcd_np = np.asarray(pcd.points)
    
    # Convert to spherical coordinates
    r = np.linalg.norm(pcd_np, axis=1)
    theta = np.arccos(pcd_np[:,2] / r)
    phi = np.arctan2(pcd_np[:,1], pcd_np[:,0])
    
    # Handle negative angles for full 360 degree coverage
    phi[phi < 0] += 2 * np.pi
    
    # Define sector boundaries
    horiz_angle_step = 2 * np.pi / horiz_sectors
    vert_angle_step = np.pi / vert_sectors
    
    # Assign points to sectors
    sector_pcds = []
    for v in range(vert_sectors):
        vert_lower = v * vert_angle_step
        vert_upper = (v + 1) * vert_angle_step
        
        for h in range(horiz_sectors):
            horiz_lower = h * horiz_angle_step
            horiz_upper = (h + 1) * horiz_angle_step
            
            # Find points within the current sector
            indices = np.where((theta >= vert_lower) & (theta < vert_upper) &
                               (phi >= horiz_lower) & (phi < horiz_upper))[0]
            
            # Extract points by indices
            sector_pcd = pcd.select_by_index(indices)
            sector_pcds.append(sector_pcd)
    
    return sector_pcds
def calculate_bounds(pcd):
    # Convert point cloud to numpy array
    pcd_np = np.asarray(pcd.points)
    
    # Compute radius as the maximum distance from the origin in the XY plane
    radius = np.max(np.linalg.norm(pcd_np[:, :2], axis=1))
    
    # Compute height as the difference between the max and min z-coordinate
    z_min = np.min(pcd_np[:, 2])
    z_max = np.max(pcd_np[:, 2])
    height = z_max - z_min

    return radius, height

def create_sector_wireframe(radius, height, horiz_sectors=8, vert_sectors=1):
    # Create an Open3D line set object for the wireframe
    lines = o3d.geometry.LineSet()
    
    # Points list
    points = []
    # Lines list
    line_indices = []
    
    # Create horizontal (circular) sector lines
    for v in range(vert_sectors + 1):
        z = -height / 2 + v * (height / vert_sectors)
        for h in range(horiz_sectors):
            angle = h * 2 * np.pi / horiz_sectors
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y, z])
            
            # Connect each point to the next in the circle
            if h < horiz_sectors - 1:
                line_indices.append([v * horiz_sectors + h, v * horiz_sectors + h + 1])
            else:
                # Connect last point back to the first in the same level
                line_indices.append([v * horiz_sectors + h, v * horiz_sectors])
    
    # Connect vertical lines between levels
    if vert_sectors > 1:
        for h in range(horiz_sectors):
            for v in range(vert_sectors):
                line_indices.append([v * horiz_sectors + h, (v + 1) * horiz_sectors + h])

    # Set points and lines to the line set
    lines.points = o3d.utility.Vector3dVector(points)
    lines.lines = o3d.utility.Vector2iVector(line_indices)

    # Define the color (e.g., yellow) for the lines
    colors = [[1, 1, 0] for i in range(len(line_indices))]  # Yellow color
    lines.colors = o3d.utility.Vector3dVector(colors)
    
    return lines

def create_sector_wireframe_with_radials(radius, height, horiz_sectors=8, vert_sectors=1):
    # Create an Open3D line set object for the wireframe
    lines = o3d.geometry.LineSet()
    
    # Points list
    points = []
    # Lines list
    line_indices = []
    
    # Add center bottom point
    center_bottom_index = 0
    points.append([0, 0, -height / 2])
    
    # Add center top point (unused for radial lines but included for completeness)
    center_top_index = 1
    points.append([0, 0, height / 2])
    
    # Create horizontal (circular) sector lines and radial lines
    for v in range(vert_sectors + 1):
        z = -height / 2 + v * (height / vert_sectors)
        for h in range(horiz_sectors):
            angle = h * 2 * np.pi / horiz_sectors
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            current_index = len(points)
            points.append([x, y, z])
            
            # Connect each point to the next in the circle
            next_index = current_index + 1 if (h + 1) < horiz_sectors else 2 + v * horiz_sectors
            line_indices.append([current_index, next_index])
            
            # For the bottom circle, add radial lines from center to the circumference point
            if v == 0:
                line_indices.append([center_bottom_index, current_index])
    
    # Connect vertical lines between levels
    for h in range(horiz_sectors):
        for v in range(vert_sectors):
            bottom_index = 2 + v * horiz_sectors + h
            top_index = 2 + (v + 1) * horiz_sectors + h
            line_indices.append([bottom_index, top_index])

    # Set points and lines to the line set
    lines.points = o3d.utility.Vector3dVector(points)
    lines.lines = o3d.utility.Vector2iVector(line_indices)

    # Define the color (e.g., black) for the lines
    colors = [[0, 0, 0] for i in range(len(line_indices))]  # Black color
    lines.colors = o3d.utility.Vector3dVector(colors)
    
    return lines

def sector_split_color(pcd, horiz_sectors=24, vert_sectors=1):
    # pcd: open3d.geometry.PointCloud
    
    # Convert point cloud to numpy array
    pcd_np = np.asarray(pcd.points)
    
    # Convert to spherical coordinates
    r = np.linalg.norm(pcd_np, axis=1)
    theta = np.arccos(pcd_np[:,2] / r)
    phi = np.arctan2(pcd_np[:,1], pcd_np[:,0])
    
    # Handle negative angles for full 360 degree coverage
    phi[phi < 0] += 2 * np.pi

    # # Define sector boundaries
    horiz_angle_step = 2 * np.pi / horiz_sectors
    vert_angle_step = (np.pi/2) / vert_sectors
    
    # Prepare a color palette
    colors = np.array([
        [1, 0, 0],    # Red
        [0, 1, 0],    # Green
        [0, 0, 1],    # Blue
        [1, 1, 0],    # Yellow
        [0, 1, 1],    # Cyan
        [1, 0, 1],    # Magenta
        [0.5, 0.5, 0],# Olive
        [0, 0.5, 0.5] # Teal
    ])

    # Assign points to sectors and color them
    sector_pcds = []
    for v in range(vert_sectors):
        vert_lower = v * vert_angle_step
        vert_upper = (v + 1) * vert_angle_step
        
        for h in range(horiz_sectors):
            horiz_lower = h * horiz_angle_step
            horiz_upper = (h + 1) * horiz_angle_step
            
            # Find points within the current sector
            indices = np.where((theta >= vert_lower) & (theta < vert_upper) &
                               (phi >= horiz_lower) & (phi < horiz_upper))[0]
            
            # Extract points by indices
            sector_pcd = pcd.select_by_index(indices)
            
            # Color points in this sector
            # Assign a color from the palette, cycling through the palette
            color = colors[(h + v * horiz_sectors) % len(colors)]
            sector_pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(indices), 1)))
            
            sector_pcds.append(sector_pcd)
    
    return sector_pcds

def cluster_pointcloud_dbscan(pcd, eps, min_points):
    """
    Cluster a point cloud using the DBSCAN algorithm and return the clusters as separate point clouds.

    Parameters:
    - pcd: open3d.geometry.PointCloud object, the input point cloud
    - eps: float, the maximum distance between two samples for one to be considered as in the neighborhood of the other
    - min_points: int, the number of samples in a neighborhood for a point to be considered as a core point

    Returns:
    - cluster_pcds: list of open3d.geometry.PointCloud objects, each representing a cluster
    """
    # If the point cloud doesn't have normals, estimate them
    if not pcd.has_normals():
        pcd.estimate_normals()

    # Apply DBSCAN clustering
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    
    # Number of clusters
    max_label = labels.max()

    # Extract individual clusters
    cluster_pcds = []
    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_pcd = pcd.select_by_index(cluster_indices)
        cluster_pcds.append(cluster_pcd)

    return cluster_pcds

def iros_change(pcdA, pcdB, eps, threshold=0.95, horiz_sectors=8):
    sectored_A = sector_split_color(pcdA, horiz_sectors=horiz_sectors)
    sectored_B = sector_split_color(pcdB, horiz_sectors=horiz_sectors)

    # Compute change for each sector between pcdA and pcdB
    change_comp = []
    change_art = []
    for i in range(len(sectored_A)):
        normComp, normArt = iros.calculate_complete_quality_metric(sectored_A[i], sectored_B[i], eps)
        change_comp.append(normComp)
        change_art.append(normArt)

    # Find the sectors with change below the threshold
    # change_selected = [c for c in change if c < threshold]

    # color the sectors based on the change
    # for i in range(len(sectored_B)):
    #     if change_comp[i] < threshold and change_art[i] < threshold:
    #         sectored_B[i].paint_uniform_color([1, 0, 0]) # Red
    #     elif change_comp[i] >= threshold and change_art[i] >= threshold:
    #         sectored_B[i].paint_uniform_color([0, 1, 0]) # Green
    #     else:
    #         print (f"Change in sector {i}: {change_comp[i]}, {change_art[i]}")
    #         sectored_B[i].paint_uniform_color([0, 0, 1]) # Blue

    for i in range(len(sectored_B)):
        if (change_comp[i] + change_art[i])/2 < threshold:
            sectored_B[i].paint_uniform_color([1, 0, 0]) # Red
        elif (change_comp[i] + change_art[i])/2 >= threshold:
            sectored_B[i].paint_uniform_color([0, 1, 0]) # Green
        else:
            print (f"Change in sector {i}: {change_comp[i]}, {change_art[i]}")
            sectored_B[i].paint_uniform_color([0, 0, 1]) # Blue

    return sectored_B, (change_comp, change_art)

def cluster_change(pcdA, pcdB, eps, threshold=0.95):
    sectored_A = sector_split_color(pcdA)
    sectored_B = sector_split_color(pcdB)

    # Compute change for each sector between pcdA and pcdB
    change = []
    for i in range(len(sectored_A)):
        change.append(iros.calculate_complete_quality_metric(sectored_A[i], sectored_B[i], eps))

    # Find the sectors with change below the threshold
    change_selected = [c for c in change if c < threshold]

    # For each selected sector, cluster the points in pcdB and color them based on the cluster
    clustered_pcds = []
    for i in range(len(sectored_B)):
        if change[i] < 0.95:
            # Cluster the points in the sector
            clusters = cluster_pointcloud_dbscan(sectored_B[i], eps=0.1, min_points=10)
            
            # Color the clusters
            for j in range(len(clusters)):
                # Assign a color to the cluster
                color = [np.random.rand(), np.random.rand(), np.random.rand()]
                clusters[j].paint_uniform_color(color)
            
            # Add the clustered point clouds to the list
            clustered_pcds.extend(clusters)
        else:
            # Add the sector to the list without clustering
            clustered_pcds.append(sectored_B[i])

    return clustered_pcds, change




def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Compute change between two point clouds')
    parser.add_argument('--pcdA', type=str, help='Path to the first point cloud') # Reference
    parser.add_argument('--pcdB', type=str, help='Path to the second point cloud') # Candidate
    parser.add_argument('--eps', type=float, help='Threshold for change detection')
    # parser.add_argument('--output', type=str, help='Path to the output point cloud', default='output.ply')
    parser.add_argument('--viz',type=str, help='Visualize the output point cloud', default='False')
    parser.add_argument('--horiz_sectors', type=int, help='Number of horizontal sectors', default=8)
    parser.add_argument('--vert_sectors', type=int, help='Number of vertical sectors', default=1)
    args = parser.parse_args()

    # Load point clouds
    pcdA = o3d.io.read_point_cloud(args.pcdA)
    pcdB = o3d.io.read_point_cloud(args.pcdB)

    # Compute change
    # pcdB = compute_change(pcdA, pcdB, args.eps)

    # Save output point cloud
    # o3d.io.write_point_cloud(args.output, pcdB)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

    # sectored = sector_split_color(pcdB)
    # print (f"Number of sectors: {len(sectored)}")
    # print sizwe of each sector
    # for i in range(len(sectored)):
        # print(f"Number of points in sector {i}: {len(sectored[i].points)}")

    sec_B, change = iros_change(pcdA, pcdB, args.eps, threshold=0.99, horiz_sectors=args.horiz_sectors)

    # clustered_pcds, change = cluster_change(pcdA, pcdB, args.eps)

    print (f"Change: {change}")


    radius, height = calculate_bounds(pcdB)
    wireframe = create_sector_wireframe_with_radials(radius, height, horiz_sectors=args.horiz_sectors, vert_sectors=args.vert_sectors)

    sec_B.append(mesh_frame)
    sec_B.append(wireframe)
    o3d.visualization.draw_geometries(sec_B)
    # o3d.visualization.draw_geometries(clustered_pcds)

    # Visualize point cloud
    # if args.viz == 'True':
        # o3d.visualization.draw_geometries([pcdB, mesh_frame])

if __name__ == "__main__":
    main()