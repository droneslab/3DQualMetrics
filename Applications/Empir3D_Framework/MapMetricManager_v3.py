import sys,os
sys.path.append(os.getcwd())

from Applications.Empir3D_Framework.system_constants import *

from Applications.Empir3D_Framework.MapMetricManager_base import MapMetricManager_base, MapCell


from QualityMetric_v3 import  calculate_quality_metric




from tqdm import tqdm

from multiprocess import Process, Manager
import copy

import time

def cluster_by_voxel(pcd, e, min_bound=None):
    point_cloud = np.asarray(pcd.points)

    if min_bound is None:
        bbox = pcd.get_axis_aligned_bounding_box()
        min_bound = np.asarray(bbox.min_bound)
    
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


def compute_parallel_metric(pcd_gt, pcd_cand, options,cell_size, min_bound, metric_fn):

    ind_gt, ind_voxel_gt, voxel_min_bounds_gt = cluster_by_voxel(pcd_gt, cell_size, min_bound)
    ind_cand, ind_voxel_cand, voxel_min_bounds_cand = cluster_by_voxel(pcd_cand, cell_size, min_bound)
    ind_gt = np.asarray(ind_gt)
    ind_cand = np.asarray(ind_cand)
    pcd_gt_np = np.asarray(pcd_gt.points)
    pcd_cand_np = np.asarray(pcd_cand.points)


    def parallel_compute(d, min_cell_index,gt_points, cand_points, options, cell_size, min_bound, metric_fn):

        #d[tuple(min_cell_index)] = calculate_quality_metric(gt_points, cand_points, e, cell_size, min_bound)
        d[str(min_cell_index)] = MapCell(min_cell_index,gt_points, cand_points, options, False, metric_fn=metric_fn, cell_size=cell_size, min_bound=min_bound)


    manager = Manager()
    d = manager.dict()

    job = []

    # union of all the voxels
    ind_voxel_gt_key = set(ind_voxel_gt.keys())
    ind_voxel_cand_key = set(ind_voxel_cand.keys())
    ind_all_voxel = ind_voxel_gt_key.union(ind_voxel_cand_key)

    
    print("Number of Iterations: ", len(ind_all_voxel))
    # print("Number of GT Voxels: ", len(ind_voxel_gt_key))
    # print("Number of Cand Voxels: ", len(ind_voxel_cand_key))
    for i, voxel in tqdm(enumerate(ind_all_voxel), total=len(ind_all_voxel), desc="Creating Jobs"):
        voxel_min_bound =None
        if tuple(voxel) not in ind_voxel_gt:
            gt_points = np.zeros((0,3))
        else:
            curr_index = ind_voxel_gt[tuple(voxel)]
            gt_points = pcd_gt_np[ind_gt ==curr_index]
            voxel_min_bound = voxel_min_bounds_gt[curr_index]

        if tuple(voxel) not in ind_voxel_cand:
            cand_points = np.zeros((0,3))
        else:
            curr_index_candidate = ind_voxel_cand[tuple(voxel)]
            cand_points = pcd_cand_np[ind_cand ==curr_index_candidate]
            if voxel_min_bound is None:
                voxel_min_bound = voxel_min_bounds_cand[curr_index_candidate]    

        job.append(Process(target=parallel_compute, args=(d, voxel,gt_points, cand_points, options, cell_size, voxel_min_bound, metric_fn)))
    print ("\n")
    num_cores = (os.cpu_count()-1)
    print("Number of Jobs: ", len(job))
    print("Number of Cores: ", num_cores)
    print ("\n")
    for i, _ in tqdm(enumerate(range(0, len(job), num_cores)), total=int(len(job)/num_cores), desc="Running Jobs"):
        _ = [p.start() for p in job[i:i+num_cores]]
        _ = [p.join() for p in job[i:i+num_cores]]
        # Free up memory
        job[i:i+num_cores] = []

    # run the remaining jobs
    _ = [p.start() for p in job]
    _ = [p.join() for p in job]
    metriccells= copy.deepcopy(d)

    # print("Number of Cells: ", len(metriccells))

    return metriccells, ind_gt, ind_voxel_gt



import open3d as o3d



class MapMetricManager_v3(MapMetricManager_base):

    def __init__(self, gt_file, cnd_file, cell_size, metric_options = {EPSILON_STR: 0.1, WEIGHT_coverage_STR: 0.1, WEIGHT_ARTIFACTS_STR: 0.1, WEIGHT_RESOLUTION_STR: 0.4, WEIGHT_ACCURACY_STR: 0.4}, compute_flag=1):
        super(MapMetricManager_v3, self).__init__(gt_file, cnd_file, cell_size, metric_options, compute_flag, calculate_quality_metric)

        self.voxels = None
        self.voxel_to_index = None

        self.metric_results = None



    def compute_metric(self, filename ="results/test.json"):

        # print("Computing Metric V3...")
        st_time = time.time()
        self.metriccells, self.voxel_to_index, self.voxels= compute_parallel_metric(self.pointcloud_GT, self.pointcloud_Cnd, self.options, self.cell_size, self.min_bound, self.metric_fn)

        print ("Time Taken: ", time.time() - st_time)
        self.save_metric(filename)


    def visualize(self):
        #color based on quality dict results
        colors = np.zeros((len(self.pointcloud_GT.points), 3))
        for voxel in self.voxels:
            curr_index = self.voxels[tuple(voxel)]

            current_quality = self.metriccells[str(voxel)].metrics[COVERAGE_STR]
            colors[self.voxel_to_index ==curr_index] = [1-current_quality,current_quality, 0]

        self.pointcloud_GT.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([self.pointcloud_GT])


    def compute_metric_average(self, metric_results):
        #iterate through all the Cells
        metric_results[AVERAGE_STR] = {}
        metric_results[VARIANCE_STR] = {}
        for metric in METRICS_LIST:
            metric_list = [metric_results[METRICS_STR][key][metric] for key in metric_results[METRICS_STR].keys()]
            if metric == RESOLUTION_STR:
                # Remove all 0 from metric_list
                metric_list = [x for x in metric_list if x !=-1]
                average_metric = np.mean(metric_list)
            elif metric == ACCURACY_STR:
                # Remove all -1 from metric_list
                metric_list = [x for x in metric_list if x !=-1]
                #metric_list = [x for x in metric_list if x != 0]
                average_metric = np.mean(metric_list)
            elif metric == COVERAGE_STR:
                metric_list_ref = [metric_results[METRICS_STR][key][COMPELTENESS_REF_STR] for key in metric_results[METRICS_STR].keys()]
                metric_list_ref = sum(metric_list_ref)
                metric_list_cand =  [metric_results[METRICS_STR][key][COMPELTENESS_BOTH_INTERSECTION_STR] for key in metric_results[METRICS_STR].keys()]
                metric_list_cand = sum(metric_list_cand)
                if metric_list_ref == 0:
                    average_metric = 0
                else:
                    average_metric = metric_list_cand/metric_list_ref
            elif metric == ARTIFACTS_STR:
                metric_list_both_intersection = [metric_results[METRICS_STR][key][COMPELTENESS_BOTH_INTERSECTION_STR] for key in metric_results[METRICS_STR].keys()]
                metric_list_both_intersection = sum(metric_list_both_intersection)
                metric_list_cand =  [metric_results[METRICS_STR][key][COMPELTENESS_CAND_ONLY_STR] for key in metric_results[METRICS_STR].keys()]
                metric_list_cand = sum(metric_list_cand)
                if (metric_list_cand) == 0:
                    average_metric = 1
                else:
                    average_metric =1 - (metric_list_cand/(metric_list_cand+metric_list_both_intersection)) 
            # else:
            #     average_metric = np.mean(metric_list)
            print("Average ", metric, ": ", average_metric)
            metric_var = np.var(metric_list)
            # print("Variance for ", metric, ": ", metric_var)
            metric_results[AVERAGE_STR][metric] = average_metric
            metric_results[VARIANCE_STR][metric] = metric_var

        self.metric_results = metric_results
                                                                 
        return metric_results

