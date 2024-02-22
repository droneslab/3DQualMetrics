
import sys
import os

print ("Using QualityMetric_v3.py")

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

sys.path.append(current)
# Getting the parent directory name
# where the current directory is present.
#parent = os.path.dirname(current)

#print("Parent Path: ", parent)
# adding the parent directory to
# the sys.path.
#sys.path.append(parent)



#
# import open3d as o3d
import numpy as np
from sklearn.neighbors import KDTree


# from PQM.utils import calculate_average_distance, calculate_Bvalid_dist, build_KDTree
# from PQM.utils import generate_3dcube_points_centers

# from PQM.QualityMetric_v1 import calculate_artifacts_metric

# from PQM.QualityMetric_v1 import calculate_accuracy_metric

# from PQM.QualityMetric_v2 import calculate_resolution_metric

def calculate_Bvalid_index(pcdA_np,pcdB_np,e):
    # pcdA -> ref
    # pcdB -> cand
    # distances of valid points in pcdB
    # pcdA = np.asarray(pcdA.points)
    # pcdB = np.asarray(pcdB.points)
    treeA = KDTree(pcdA_np)
    #treeB = KDTree(pcdB)
    # Find the nearest neighbor in pcdA for each point in pcdB
    distB, indB = treeA.query(pcdB_np, k=1)
    # Find valid points based on threshold e
    validBind = indB[distB<e]
    return validBind

import time 


def calculate_subvoxel_indices(point_cloud, min_bound, e):
    return np.floor((point_cloud - min_bound) / e).astype(int)

def calculate_coverage(point_cloud_1, point_cloud_2, min_bound, e):

    # Calculate sub-voxel indices for both point clouds
    subvoxel_indices_pc1 = calculate_subvoxel_indices(point_cloud_1, min_bound, e)
    subvoxel_indices_pc2 = calculate_subvoxel_indices(point_cloud_2, min_bound, e)
    
    # Convert sub-voxel indices to sets
    subvoxel_indices_set_pc1 = set(map(tuple, subvoxel_indices_pc1))
    subvoxel_indices_set_pc2 = set(map(tuple, subvoxel_indices_pc2))
    
    # Find common and unique sub-voxel indices
    covered_index = subvoxel_indices_set_pc1.intersection(subvoxel_indices_set_pc2)
    #ref_only_indices = subvoxel_indices_set_pc1.difference(subvoxel_indices_set_pc2)
    cand_only_indices = subvoxel_indices_set_pc2.difference(subvoxel_indices_set_pc1)

    #calculate coverage
    coverage = 1.0 if len(subvoxel_indices_set_pc1) == 0 else len(covered_index) / len(subvoxel_indices_set_pc1)
    count_ref, count_cand, count_both = len(subvoxel_indices_set_pc1), len(subvoxel_indices_set_pc2), len(covered_index)

    return coverage, count_ref, count_cand, count_both, len(cand_only_indices)
    # Return the counts
    # return len(covered_subvoxel_indices), len(unique_subvoxel_indices_pc1), len(unique_subvoxel_indices_pc2)






# ## New implementation for resolution metric based on avg distance between points
# def calculate_resolution_metric(pcdA,pcdB):
#     # pcdA -> ref
#     # pcdB -> cand
#     # Normalized resolution score
#     # Calculate the average distance between points in pcdA
#     avgDistA = calculate_average_distance(pcdA)
#     avgDistB = calculate_average_distance(pcdB)
    
#     if avgDistB == 0:
#         #Pranay: TODO : Please check, This is a hack to avoid division by zero
#         avgDistB = 1e-6 

#     resRatio = avgDistA/avgDistB

#     # print ("Res ratio: ", resRatio)

#     return resRatio

from sklearn.neighbors import KDTree

def build_KDTree_np(pcd_np):
    # Build KDTree for point cloud pcd
    #points = np.asarray(pcd.points)
    tree = KDTree(pcd_np)
    return tree


def calculate_Bvalid_dist_np(pcdB,e, treeA):
    # pcdA -> ref
    # pcdB -> cand
    # distances of valid points in pcdB

    # treeA = KDTree(pcdA)
    #treeB = KDTree(pcdB)
    # Find the nearest neighbor in pcdA for each point in pcdB
    distB, indB = treeA.query(pcdB, k=1)
    # Find valid points based on threshold e
    validB = distB[distB<e]
    return validB
def calculate_artifacts_metric_np(pcdB,validB_count):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized artifact score
    # validB_count = calculate_Bvalid_count(pcdA,pcdB,e)
    return (validB_count/len(pcdB))

def calculate_average_distance_np(pointsA, treeA):
    # pcdA -> ref
    # Calculate the average distance between points in pcdA
    
    #print("shape of pointsA: ", pointsA.shape)
    if len(pointsA) < 2:
        return 0
    # treeA = KDTree(pointsA)
    # Find the nearest neighbor in pcdA for each point in pcdA
    distA, indA = treeA.query(pointsA, k=2)
    # Calculate the average distance between points in pcdA
    avgDistA = np.mean(distA[:,1])
    return avgDistA

## New implementation for resolution metric based on avg distance between points
def calculate_resolution_metric_np(pcdA,pcdB,treeA,treeB, e):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized resolution score
    # Calculate the average distance between points in pcdA
    avgDistA = calculate_average_distance_np(pcdA,treeA)
    # avgDistA = calculate_average_distance(pcdA)
    avgDistB = calculate_average_distance_np(pcdB,treeB)
    # print ("Avg dist A: ", avgDistA)
    # print ("Avg dist A: ", avgDistA)
    # Return ratio between e and minDistB
    # resRatio = 0
    # if e/avgDistB < 1:
    #     # resRatio = e / minDistB
    #     resRatio = avgDistB / avgDistA
    # else:
    #     resRatio = 1
    if avgDistB == 0:
        avgDistB = (np.sqrt(3) *e)/2
    if avgDistA > avgDistB:
        return 1.0

    resRatio = avgDistA/avgDistB

    # print ("Res ratio: ", resRatio)

    return resRatio




def cluster_by_voxel(point_cloud, e, min_bound):
    # point_cloud = np.asarray(pcd.points)

    # if min_bound is None:
    #     bbox = pcd.get_axis_aligned_bounding_box()
    #     min_bound = np.asarray(bbox.min_bound)
    
    ind_voxel = np.floor((point_cloud - min_bound) / e).astype(int)
    
    # Convert to a structured array for unique operation
    structured_arr = np.ascontiguousarray(ind_voxel).view(np.dtype((np.void, ind_voxel.dtype.itemsize * ind_voxel.shape[1])))
    
    # Find unique rows (i.e., unique voxels) and their indices
    unique_structs, unique_indices = np.unique(structured_arr, return_inverse=True)
    

    # Convert back to original array shape
    unique_tuples = ind_voxel[np.unique(unique_indices, return_index=True)[1]]

    voxel_to_index = {tuple(unique_tuple): index for index, unique_tuple in enumerate(unique_tuples)}

    #print("tuple_to_index", voxel_to_index)
    

    # Convert back to original array shape
    unique_voxels = unique_structs.view(ind_voxel.dtype).reshape(-1, ind_voxel.shape[1])
    #print("unique_voxels", unique_voxels)
    # Compute the minimum bound for each unique voxel
    voxel_min_bounds = unique_voxels.astype(float) * e + min_bound

    # Convert unique_indices to a Python list
    indices_list = unique_indices.tolist()
    
    return indices_list, voxel_to_index, voxel_min_bounds



## New implementation for resolution metric based on avg distance between points
def get_covered_points(pcd_gt_np, pcd_cand_np,e, min_bound):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized resolution score
    # Calculate the average distance between points in pcdA

    ind_gt, ind_voxel_gt, voxel_min_bounds_gt = cluster_by_voxel(pcd_gt_np, e, min_bound)
    ind_cand, ind_voxel_cand, voxel_min_bounds_cand = cluster_by_voxel(pcd_cand_np, e, min_bound)
    ind_gt = np.asarray(ind_gt)
    ind_cand = np.asarray(ind_cand)
    ind_voxel_gt_key = set(ind_voxel_gt.keys())
    ind_voxel_cand_key = set(ind_voxel_cand.keys())
    ind_all_covered_voxel = ind_voxel_gt_key.intersection(ind_voxel_cand_key)

    all_cov_gt_points = []
    all_cov_cand_points = []
    for voxel in ind_all_covered_voxel:

        curr_index = ind_voxel_gt[tuple(voxel)]
        gt_points = pcd_gt_np[ind_gt ==curr_index]
        all_cov_gt_points.extend(gt_points)

        curr_index_candidate = ind_voxel_cand[tuple(voxel)]
        cand_points = pcd_cand_np[ind_cand ==curr_index_candidate]

        all_cov_cand_points.extend(cand_points)
    
    all_cov_gt_points = np.asarray(all_cov_gt_points)
    all_cov_cand_points = np.asarray(all_cov_cand_points)
    return all_cov_gt_points, all_cov_cand_points



def calculate_accuracy_metric_np(pcdB,e,validB_dist):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized accuracy score
    # Find the nearest neighbor in pcdB for each point in pcdA
    # validB_dist = calculate_Bvalid_dist(pcdA,pcdB,e)
    normAcc = 1-(validB_dist.sum() / (len(pcdB)* e))
    return normAcc

def calculate_quality_metric(pcdA_np, pcdB_np, e, cell_size, min_bound):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized complete quality score
    #calculate the accuracy metric
    count_ref, count_both, count_cand_only  = 0,0,0

    #print("len(pcdA_np): ", len(pcdA_np))
    #print("len(pcdB_np): ", len(pcdB_np))
    #if bboxA is None:
    #    bboxA = pcdA_np.get_axis_aligned_bounding_box()
    if len(pcdA_np) == 0 and len(pcdB_np) == 0: #case 1: if both the pointclouds are empty
        normAcc = -1
        normArt = 1
        normRes = -1
        normCov = 1

    elif len(pcdA_np) != 0 and len(pcdB_np) != 0: # case 2: if both the pointclouds are not empty

        # Build KDTree for both the pointclouds
        treeA = build_KDTree_np(pcdA_np)
        treeB = build_KDTree_np(pcdB_np)


        # Testing coverage metric
        #normCov, count_ref, count_cand = calculate_coverage(treeA,treeB, bboxA, e=e)
        normCov, count_ref, count_cand, count_both, count_cand_only = calculate_coverage(pcdA_np, pcdB_np, min_bound, e=e)
        # Calculate the artifacts metric
        #normArt = calculate_artifacts_metric_np(pcdB_np, validB_count)

        #Using subvoxel indices 
        # normArt = count_both / count_cand

        if count_cand_only == 0:
            normArt = 1
        else:
            normArt = 1- count_cand_only/count_cand
        # Calculate the resolution metric

        pcdA_np_cov , pcdB_np_cov = get_covered_points(pcdA_np, pcdB_np,cell_size, min_bound)

        normRes = calculate_resolution_metric_np(pcdA_np_cov , pcdB_np_cov, treeA, treeB, e)
        # Calculate the accuracy metric

        validB_dist = calculate_Bvalid_dist_np(pcdB_np_cov,e, treeA)
        normAcc = calculate_accuracy_metric_np(pcdB_np_cov, e, validB_dist)
    elif len(pcdA_np) == 0 and len(pcdB_np) != 0: # case 3: if ref pointcloud is empty
        normAcc = -1
        normArt = 0
        normRes = -1
        normCov = 1
        count_cand_only = len(pcdB_np)
        # Calculate sub-voxel indices for both point clouds
        subvoxel_indices_pc2 = calculate_subvoxel_indices(pcdB_np, min_bound, e)
        count_cand_only = len(set(map(tuple, subvoxel_indices_pc2)))

    else: # case 4: if cand pointcloud is empty
        normAcc = -1
        normArt = 1
        normRes = -1
        normCov = 0
        #count_ref = len(pcdA_np)

        # Calculate sub-voxel indices for both point clouds
        subvoxel_indices_pc1 = calculate_subvoxel_indices(pcdA_np, min_bound, e)
        count_ref = len(set(map(tuple, subvoxel_indices_pc1)))



    return normCov, normArt, normRes, normAcc, int(count_ref), int(count_both), int(count_cand_only)


