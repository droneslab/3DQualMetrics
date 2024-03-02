
import sys
import os
from tqdm import tqdm
current = os.path.dirname(os.path.realpath(__file__))
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append(current)


#
# import open3d as o3d
import numpy as np
from sklearn.neighbors import KDTree


def calculate_Bvalid_index(pcdA_np,pcdB_np,e):
    treeA = KDTree(pcdA_np)
    distB, indB = treeA.query(pcdB_np, k=1)
    # Find valid points based on threshold e
    validBind = indB[distB<e]
    return validBind

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


from sklearn.neighbors import KDTree

def build_KDTree_np(pcd_np):
    tree = KDTree(pcd_np)
    return tree


def calculate_Bvalid_dist_np(pcdB,e, treeA):
    distB, indB = treeA.query(pcdB, k=1)
    # Find valid points based on threshold e
    validB = distB[distB<e]
    return validB
def calculate_artifacts_metric_np(pcdB,validB_count):
    return (validB_count/len(pcdB))

def calculate_average_distance_np(pointsA, treeA):
    if len(pointsA) < 2:
        return 0
    # Find the nearest neighbor in pcdA for each point in pcdA
    distA, indA = treeA.query(pointsA, k=2)
    # Calculate the average distance between points in pcdA
    avgDistA = np.mean(distA[:,1])
    return avgDistA

## New implementation for resolution metric based on avg distance between points
def calculate_resolution_metric_np(pcdA,pcdB,treeA,treeB, e):
    # Calculate the average distance between points in pcdA
    avgDistA = calculate_average_distance_np(pcdA,treeA)

    avgDistB = calculate_average_distance_np(pcdB,treeB)

    if avgDistB == 0:
        avgDistB = (np.sqrt(3) *e)/2
    if avgDistA > avgDistB:
        return 1.0

    resRatio = avgDistA/avgDistB
    return resRatio




def cluster_by_voxel(point_cloud, e, min_bound):
    
    ind_voxel = np.floor((point_cloud - min_bound) / e).astype(int)
    
    # Convert to a structured array for unique operation
    structured_arr = np.ascontiguousarray(ind_voxel).view(np.dtype((np.void, ind_voxel.dtype.itemsize * ind_voxel.shape[1])))
    
    # Find unique rows (i.e., unique voxels) and their indices
    unique_structs, unique_indices = np.unique(structured_arr, return_inverse=True)
    

    # Convert back to original array shape
    unique_tuples = ind_voxel[np.unique(unique_indices, return_index=True)[1]]

    voxel_to_index = {tuple(unique_tuple): index for index, unique_tuple in enumerate(unique_tuples)}

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


# def calculate_Bvalid(pcdA,pcdB,e):
#     # pcdA -> ref
#     # pcdB -> cand
#     # distances of valid points in pcdB
#     pcdA = np.asarray(pcdA.points)
#     pcdB = np.asarray(pcdB.points)
#     treeA = KDTree(pcdA)
#     #treeB = KDTree(pcdB)
#     # Find the nearest neighbor in pcdA for each point in pcdB
#     distB, indB = treeA.query(pcdB, k=1)
#     # Find valid points based on threshold e
#     validB = distB[distB<e]
#     return validB

def calculate_accuracy_metric_np(pcdB,e,validB_dist):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized accuracy score
    # Find the nearest neighbor in pcdB for each point in pcdA
    normAcc = 1-(validB_dist.sum() / (len(pcdB)* e))
    return normAcc

# def calculate_color_metric(pcdA,pcdB, e):
#     # Compute the color difference between the two point clouds
#     # pcdA -> ref
#     # pcdB -> cand
#     # Color score based on valid points, valid points are the points in pcdB that are within e distance of pcdA
    
#     validB_points = calculate_Bvalid(pcdA,pcdB,e)

#     # Calculate the color difference between the valid points
#     color_diff = np.linalg.norm(pcdA.colors[validB_points] - pcdB.colors[validB_points], axis=1)
#     # Calculate the color score
#     color_score = 1 - (color_diff.sum() / (len(validB_points) * 255 * np.sqrt(3)))
#     return color_score

def calculate_Bvalid(pointsA, pointsB, e):
    """
    Calculate valid points in pointsB that are within a distance 'e' from pointsA.
    
    :param pointsA: NumPy array of points for pcdA.
    :param pointsB: NumPy array of points for pcdB.
    :param e: Distance threshold.
    :return: Boolean array indicating valid points in pointsB, and indices of those points in pointsA.
    """
    treeA = KDTree(pointsA)
    distB, indB = treeA.query(pointsB, k=1)
    valid_indices = distB < e
    return valid_indices, indB[valid_indices]


def calculate_color_metric_cos(pointsA, colorsA, pointsB, colorsB, e):
    cos_values = []

    # Loop over each point in pointsB
    for i, pointB in enumerate(tqdm(pointsB, desc="Computing color metric")):
        # Calculate the Euclidean distance from pointB to all points in pointsA
        distances = np.sqrt(np.sum((pointsA - pointB)**2, axis=1))
        # Create a boolean mask where distances are less than the threshold 'e'
        within_e = distances < e
        # If there are any points within 'e' distance
        if np.any(within_e):
            closest_point_idx = np.argmin(distances)
            # Calculate cosine similarity between the colors of pointB and its closest point in pointsA
            cos_sim = cosine_similarity([colorsB[i]], [colorsA[closest_point_idx]])
            # print ("cos_sim", cos_sim)
            cos_values.append(cos_sim[0][0])
            
    # Calculate the average cosine similarity across all valid points
    average_cos = np.mean(cos_values)
    return average_cos


def calculate_color_metric_hsv(pointsA, colorsA, pointsB, colorsB, e):
    from skimage import data
    from skimage.color import rgb2hsv
    cos_values = []


    for i, pointB in enumerate(tqdm(pointsB, desc="Computing color metric")):
        # Calculate the Euclidean distance from pointB to all points in pointsA
        distances = np.sqrt(np.sum((pointsA - pointB)**2, axis=1))
        # Create a boolean mask where distances are less than the threshold 'e'
        within_e = distances < e
        # If there are any points within 'e' distance
        if np.any(within_e):
            # Find the index of the closest point in pointsA to pointB
            closest_point_idx = np.argmin(distances)

            # Convert the colors to HSV
            hsvA = rgb2hsv(colorsA[closest_point_idx])
            hsvB = rgb2hsv(colorsB[i])  

            # Cosine similarity between the colors of pointB and its closest point in pointsA
            cos_sim = cosine_similarity([hsvB], [hsvA])

            cos_values.append(cos_sim[0][0])
            


    # Calculate the average cosine similarity across all valid points
    average_cos = np.mean(cos_values)
    return average_cos

def calculate_color_metric(pointsA, colorsA, pointsB, colorsB, e):
    mse_values = []

    for i, pointB in enumerate(tqdm(pointsB, desc="Computing color metric")):
        # Calculate the Euclidean distance from pointB to all points in pointsA
        distances = np.sqrt(np.sum((pointsA - pointB)**2, axis=1))
        # Create a boolean mask where distances are less than the threshold 'e'
        within_e = distances < e
        # If there are any points within 'e' distance
        if np.any(within_e):
            # Find the index of the closest point in pointsA to pointB
            closest_point_idx = np.argmin(distances)

            color_diff_r = colorsB[i][0] - colorsA[closest_point_idx][0]
            color_diff_g = colorsB[i][1] - colorsA[closest_point_idx][1]
            color_diff_b = colorsB[i][2] - colorsA[closest_point_idx][2]

            mse_r = np.square(color_diff_r)
            mse_g = np.square(color_diff_g)
            mse_b = np.square(color_diff_b)
            mse = (mse_r + mse_g + mse_b)/3

    average_rmse = np.sqrt(np.mean(mse_values))

    return average_rmse

# def calculate_color_metric(pointsA, colorsA, pointsB, colorsB, e):
#     """
#     Calculate the color metric (average PSNR) for valid points in pcdB relative to pcdA.
    
#     :param pointsA: NumPy array of points for pcdA.
#     :param colorsA: NumPy array of colors for pcdA.
#     :param pointsB: NumPy array of points for pcdB.
#     :param colorsB: NumPy array of colors for pcdB.
#     :param e: Distance threshold for valid points.
#     :return: PSNR value for valid points.
#     """
#     valid_indices, valid_indices_A = calculate_Bvalid(pointsA, pointsB, e)
#     valid_colors_B = colorsB[valid_indices]
#     valid_colors_A = colorsA[valid_indices_A]
    
#     # Calculate the color difference (Euclidean distance) for valid points
#     color_diff = np.linalg.norm(valid_colors_B - valid_colors_A, axis=1)
    
#     # Assuming the maximum color value is 255
#     MAX_I = 255.0
#     MSE = np.mean(np.square(color_diff))
    
#     if MSE == 0:
#         # Perfect match case
#         return float('inf')
#     PSNR = 20 * np.log10(MAX_I / np.sqrt(MSE))
    
#     return PSNR

def calculate_quality_metric(pcdA_np, pcdB_np, e, cell_size, min_bound):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized complete quality score
    #calculate the accuracy metric
    count_ref, count_both, count_cand_only  = 0,0,0
    if len(pcdA_np) == 0 and len(pcdB_np) == 0: #case 1: if both the pointclouds are empty
        normAcc = -1
        normArt = 1
        normRes = -1
        normCov = 1

    elif len(pcdA_np) != 0 and len(pcdB_np) != 0: # case 2: if both the pointclouds are not empty

        # Build KDTree for both the pointclouds
        treeA = build_KDTree_np(pcdA_np)
        treeB = build_KDTree_np(pcdB_np)

        normCov, count_ref, count_cand, count_both, count_cand_only = calculate_coverage(pcdA_np, pcdB_np, min_bound, e=e)
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


