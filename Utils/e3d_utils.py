

import numpy as np
from sklearn.neighbors import KDTree

def build_KDTree(pcd):
    # Build KDTree for point cloud pcd
    points = np.asarray(pcd.points)
    tree = KDTree(points)
    return tree

def calculate_average_distance(pcdA, treeA):
    # pcdA -> ref
    # Calculate the average distance between points in pcdA
    
    pointsA = np.asarray(pcdA.points)
    #print("shape of pointsA: ", pointsA.shape)
    if len(pointsA) < 2:
        return 0
    # treeA = KDTree(pointsA)
    # Find the nearest neighbor in pcdA for each point in pcdA
    distA, indA = treeA.query(pointsA, k=2)
    # Calculate the average distance between points in pcdA
    avgDistA = np.mean(distA[:,1])
    return avgDistA


def calculate_Bvalid_dist(pcdA,pcdB,e, treeA):
    # pcdA -> ref
    # pcdB -> cand
    # distances of valid points in pcdB
    pcdA = np.asarray(pcdA.points)
    pcdB = np.asarray(pcdB.points)
    # treeA = KDTree(pcdA)
    #treeB = KDTree(pcdB)
    # Find the nearest neighbor in pcdA for each point in pcdB
    distB, indB = treeA.query(pcdB, k=1)
    # Find valid points based on threshold e
    validB = distB[distB<e]
    return validB




### Voxelize point cloud
import open3d as o3d

def get_cropping_bound(min_bound, chunk_size, min_cell_index, max_cell_index):
    min_bound = [min_bound[0]+ chunk_size*min_cell_index[0], min_bound[1]+chunk_size*min_cell_index[1], min_bound[2]+chunk_size*min_cell_index[2]]
    max_bound = [min_bound[0]+ chunk_size*max_cell_index[0], min_bound[1]+chunk_size*max_cell_index[1], min_bound[2]+chunk_size*max_cell_index[2]]
    return min_bound, max_bound

def get_cropped_point_cloud(pcd, min_bound_source, chunk_size, min_cell_index, max_cell_index):
    min_bound, max_bound = get_cropping_bound(min_bound_source, chunk_size, min_cell_index, max_cell_index)
    #print(min_bound, max_bound, min_cell_index, max_cell_index)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    pcd_cropped = pcd.crop(bbox)
    return pcd_cropped, bbox


def calculate_voxel_grid_for_point_clouds(pcdA,pcdB,cell_size):
    # pcdA -> ref
    # pcdB -> cand
    # Calculate the voxel grid for pointclouds

    #compute the min bound of the pointcloud
    bb1 = pcdA.get_axis_aligned_bounding_box()
    bb2 = pcdB.get_axis_aligned_bounding_box()   

    min_bound = np.minimum(bb1.min_bound, bb2.min_bound)
    max_bound = np.maximum(bb1.max_bound, bb2.max_bound)

    cell_dim = (np.ceil((max_bound - min_bound) / cell_size)).astype(int)
    max_bound = min_bound + cell_dim * cell_size


    return min_bound, cell_dim


def generate_3dcube_points(min_bound, max_bound, e):

    from itertools import product, combinations
    # Generate the points for the cube
    points=[]
    x_range = np.arange(min_bound[0], max_bound[0], e)
    y_range = np.arange(min_bound[1], max_bound[1], e)
    z_range = np.arange(min_bound[2], max_bound[2], e)

    for point in product(x_range, y_range, z_range):
        points.append(point)
    points = np.array(points)
    return points

def generate_3dcube_points_centers(min_bound, max_bound, e):
    # Create coordinate arrays
    x_range = np.arange(min_bound[0]+e/2, max_bound[0]-e/2, e)
    y_range = np.arange(min_bound[1]+e/2, max_bound[1]-e/2, e)
    z_range = np.arange(min_bound[2]+e/2, max_bound[2]-e/2, e)
    
    # Create a mesh grid and reshape it to get the points
    x, y, z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T

    return points


def iterate_voxels(pointsA, pointsB, cell_size):

    pcdA = o3d.geometry.PointCloud()
    pcdA.points = o3d.utility.Vector3dVector(pointsA)
    pcdB = o3d.geometry.PointCloud()
    pcdB.points = o3d.utility.Vector3dVector(pointsB)

    min_bound, cell_dim = calculate_voxel_grid_for_point_clouds(pcdA,pcdB,cell_size)
    print("min_bound, cell_dim", min_bound, cell_dim)
    # def iterate_cells(cell_dim):
    # #iterate through all the Cells
    #     for i in range(int(cell_dim[0])):
    #         for j in range(int(cell_dim[1])):
    #             for k in range(int(cell_dim[2])):
    #                 min_cell_index = np.array([i, j, k])
    #                 max_cell_index = np.array([i+1, j+1, k+1])
    #                 #print(min_cell_index, max_cell_index)
    #                 yield min_cell_index, max_cell_index


    for i in range(int(cell_dim[0])):
        for j in range(int(cell_dim[1])):
            for k in range(int(cell_dim[2])):
                min_cell_idx = np.array([i, j, k])
                max_cell_idx = np.array([i+1, j+1, k+1])

                #for min_cell_idx, max_cell_idx in iterate_cells(cell_dim):
                curr_pcdA, bboxA = get_cropped_point_cloud(pcdA, min_bound, cell_size, min_cell_idx, max_cell_idx)
                curr_pcdB, bboxB = get_cropped_point_cloud(pcdB, min_bound, cell_size, min_cell_idx, max_cell_idx)

                yield curr_pcdA, curr_pcdB, bboxA, bboxB, min_cell_idx, max_cell_idx
