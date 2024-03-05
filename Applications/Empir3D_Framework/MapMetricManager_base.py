import sys,os
sys.path.append(os.getcwd())

from QualityMetric_v4 import  calculate_4d_metric as calculate_quality_metric
import numpy as np
import open3d as o3d
from tqdm import tqdm
import copy
import json
import pickle as pkl
from Utils.framework_util import get_cropping_bound, get_cropped_point_cloud, generate_grid_lines, get_cropped_box

from Applications.Empir3D_Framework.system_constants import *

#import matplotlib.pyplot as plt

def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(10.0, 0.0)
    return False

	
def change_background_to_black(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    return False

def change_background_to_white(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    return False

# def load_render_option(vis):
#     vis.get_render_option().load_from_json(render_option_path)
#     return False

def capture_depth(vis):
    depth = vis.capture_depth_float_buffer()
    plt.imshow(np.asarray(depth))
    plt.show()
    return False

def capture_image(vis):
    image = vis.capture_screen_float_buffer()
    plt.imshow(np.asarray(image))
    plt.show()
    return False

import time

class MapCell:
    def __init__(self, cell_index, pcd_gt,pcd_cand,pointcloud_gt, pointcloud_cnd, options, compute_flag,fill_metrics = True, metric_fn = calculate_quality_metric,  cell_size=None, min_bound=None):
        self.cell_index = cell_index
        self.metrics = {}
        self.options = options
        self.compute_flag = compute_flag
        self.metric_fn = metric_fn
        print ("Test")
        print (type(pointcloud_gt)) 
        if fill_metrics:
            self.metrics[COVERAGE_STR], self.metrics[ARTIFACTS_STR], self.metrics[RESOLUTION_STR], self.metrics[ACCURACY_STR],self.metrics[COLOR_STR], self.metrics[COMPELTENESS_REF_STR], self.metrics[COMPELTENESS_BOTH_INTERSECTION_STR], self.metrics[COMPELTENESS_CAND_ONLY_STR] = self.compute_metric_v3(pcd_gt,pcd_cand,pointcloud_gt, pointcloud_cnd, options[EPSILON_STR],  cell_size=cell_size, min_bound=min_bound)  
            self.metrics[QUALITY_STR]  = self.metrics[COVERAGE_STR] * options[WEIGHT_coverage_STR] + self.metrics[ARTIFACTS_STR] * options[WEIGHT_ARTIFACTS_STR] + self.metrics[RESOLUTION_STR] * options[WEIGHT_RESOLUTION_STR] + self.metrics[ACCURACY_STR] * options[WEIGHT_ACCURACY_STR]     

            # self.metrics[CHAMFER_STR] = -1 # calculate_chamfer_distance_metric(pointcloud_gt, pointcloud_cnd)
            # self.metrics[NORMALIZED_CHAMFER_STR] = -1 #calculate_normalized_chamfer_distance_metric(pointcloud_gt, pointcloud_cnd)
            # self.metrics[HOUSDORFF_STR] = -1 #calculate_hausdorff_distance_metric(pointcloud_gt, pointcloud_cnd)

    def compute_metric_v3(self, pcd_gt,pcd_cand,pointcloud_gt, pointcloud_cnd, eps,  cell_size, min_bound):
        # cell_size = bbox_gt.get_extent()[0]
        # min_bound = bbox_gt.min_bound
        return self.metric_fn(pcd_gt,pcd_cand,pointcloud_gt, pointcloud_cnd, eps, cell_size, min_bound)
    # def compute_metric(self, pointcloud_gt, pointcloud_cnd, eps):

    #     return self.metric_fn(pointcloud_gt, pointcloud_cnd, eps)
                                                                         

def parse_mapmetric_cells(cell_index, options, cell_metrics):
    cell = MapCell(cell_index, None, None, options, fill_metrics = False, compute_flag = 2)
    cell.metrics = cell_metrics
    return cell

def get_list_from_string(cell_index):
    cell_indx_val_tmp = cell_index.replace("[", "").replace("]", "").split(" ")
    cell_indx_val = np.array([int(x) for x in cell_indx_val_tmp if x.strip() != ""])
    return cell_indx_val



def get_list_from_string_tuple(cell_index):
    cell_indx_val_tmp = cell_index.replace("(", "").replace(")", "").replace(",", "").split(" ")
    #print(cell_indx_val_tmp)
    cell_indx_val = np.array([int(x) for x in cell_indx_val_tmp if x.strip() != ""])
    return cell_indx_val

def parse_mapmetric_config(config_file):
    with open(config_file) as f:
        config = json.load(f)
        map_metric = MapMetricManager_base(config[CONFIG_GT_FILE_STR], config[CONFIG_CND_FILE_STR], config[CELL_SIZE_STR], config[CONFIG_OPTIONS_STR])
        for cell_index in config[METRICS_STR]:
            # cell_indx_val_tmp = cell_index.replace("[", "").replace("]", "").split(" ")
            # cell_indx_val = np.array([int(x) for x in cell_indx_val_tmp if x.strip() != ""])
            map_metric.metriccells[cell_index] = parse_mapmetric_cells(get_list_from_string(cell_index), config[CONFIG_OPTIONS_STR], config[METRICS_STR][cell_index])
        return map_metric
    


class MapMetricManager_base:
    def __init__(self, gt_file, cnd_file, cell_size, metric_options = {EPSILON_STR: 0.1, WEIGHT_coverage_STR: 0.1, WEIGHT_ARTIFACTS_STR: 0.1, WEIGHT_RESOLUTION_STR: 0.4, WEIGHT_ACCURACY_STR: 0.4}, compute_flag=1, metric_fn = calculate_quality_metric):

        self.gt_file = gt_file
        self.cnd_file = cnd_file

        self.compute_flag = compute_flag

        self.pointcloud_GT = o3d.io.read_point_cloud(gt_file)
        self.pointcloud_Cnd = o3d.io.read_point_cloud(cnd_file)

        # self.pointcloud_GT.paint_uniform_color(GT_COLOR)
        # self.pointcloud_Cnd.paint_uniform_color(CND_COLOR)

        self.cell_size = cell_size
        #compute the min bound of the pointcloud
        bb1 = self.pointcloud_Cnd.get_axis_aligned_bounding_box()
        bb2 = self.pointcloud_GT.get_axis_aligned_bounding_box()

        self.min_bound = np.minimum(bb1.min_bound, bb2.min_bound)
        self.max_bound = np.maximum(bb1.max_bound, bb2.max_bound)

        #print(self.min_bound, self.max_bound)

        self.cell_dim = (np.ceil((self.max_bound - self.min_bound) / self.cell_size)).astype(int)
        self.max_bound = self.min_bound + self.cell_dim * self.cell_size

        self.metriccells = {}

        self.options = metric_options

        self.metric_fn = metric_fn

    def reset(self, cell_size, metric_options = {EPSILON_STR: 0.1, WEIGHT_coverage_STR: 0.1, WEIGHT_ARTIFACTS_STR: 0.1, WEIGHT_RESOLUTION_STR: 0.4, WEIGHT_ACCURACY_STR: 0.4}):
        self.metriccells = {}
        self.cell_size = cell_size
        self.cell_dim = (np.ceil((self.max_bound - self.min_bound) / self.cell_size)).astype(int)
        self.max_bound = self.min_bound + self.cell_dim * self.cell_size
        self.options = metric_options

    #visualize pointcloud
    def visualize(self,  show = True, show_grid = True):
	
        #visualize the pointcloud
        self.pointcloud_GT.paint_uniform_color(GREEN_COLOR)
        self.pointcloud_Cnd.paint_uniform_color(RED_COLOR)

        #o3d.visualization.draw_geometries([self.pointcloud_GT, self.pointcloud_Cnd])
        pcd_list = [self.pointcloud_GT, self.pointcloud_Cnd]

        if show_grid:
            grid_lines = generate_grid_lines(self.min_bound, self.max_bound, self.cell_dim)        
            for line in grid_lines:
                pcd_list.append(line)
        if show:
            o3d.visualization.draw_geometries(pcd_list)

        return pcd_list
        # o3d.visualization.draw_geometries_with_animation_callback(pcd_list, rotate_view)

    def get_heatmap(self, metric_name):
        heatmap = copy.deepcopy(self.pointcloud_GT) #o3d.geometry.PointCloud()
        heatmap.paint_uniform_color([0, 0, 0])
        colors = np.zeros((len(heatmap.points), 3))
        metric_val_list = []
        for cell_index in self.metriccells:
            cell = self.metriccells[cell_index]
            metric_val_list.append(cell.metrics[metric_name])
        metric_val_list = np.array(metric_val_list)
        if not np.all(metric_val_list == 0):
            metric_val_list = (metric_val_list - np.min(metric_val_list)) / (np.max(metric_val_list) - np.min(metric_val_list))
        #print("max and min",metric_val_list )
        i=0
        # print(np.mean(metric_val_list))
        # print(np.max(metric_val_list))

        red = np.array([1,0,0])
        green = np.array([0,1,0])

        qualities = np.linspace(0,1,20)

        color_map = np.zeros((20,3))

        color_dict = {
        }

        for k in range(20):
            color_map[k] = red * (1-qualities[k]) + green * qualities[k]
            color_dict[qualities[k]] = color_map[k]

        
        for cell_index in self.metriccells:
            cell = self.metriccells[cell_index]
            cell_index_vals = get_list_from_string_tuple(cell_index)
            cell_index_next = cell_index_vals + np.array([1, 1, 1])

            min_bound, max_bound = get_cropping_bound(self.min_bound, self.cell_size, cell_index_vals, cell_index_next)
            #print(min_bound, max_bound)
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)

            col_indx = bbox.get_point_indices_within_bounding_box(heatmap.points)
            

            #color = cell.metrics[metric_name] * GREEN_COLOR + (1-cell.metrics[metric_name]) * RED_COLOR
            curr_val = float(metric_val_list[i])


            closest_color = min(color_dict.keys(), key=lambda x:abs(x-curr_val))
            color = color_dict[closest_color]

            # print(curr_val,closest_color)

            # color = curr_val * GREEN_COLOR.astype(np.float32) + (1-curr_val) * RED_COLOR.astype(np.float32)
            #colors[col_indx] = [1-cell.metrics[metric_name], cell.metrics[metric_name], 0]
            
            colors[col_indx] = color
            #cropped_gt.paint_uniform_color([cell.metrics[metric_name], 0, 0])
            i+=1


            #cell_center = self.min_bound + (cell.cell_index + 0.5) * self.cell_size
            #heatmap.points.append(cell_center)
        heatmap.colors = o3d.utility.Vector3dVector(colors) # append([cell.metrics[metric_name], 0, 0])
        return heatmap

    
    #visualize pointcloud heatmap
    def visualize_heatmap(self, metric_name, show = True, show_grid = True):
        #visualize the pointcloud
        heatmap = self.get_heatmap(metric_name)
        #heatmap.paint_uniform_color(RED_COLOR)
        # if save:
        #     o3d.io.write_point_cloud(filename, heatmap)

        pcd_list = [heatmap]

        if show_grid:
            grid_lines = generate_grid_lines(self.min_bound, self.max_bound, self.cell_dim)        
            for line in grid_lines:
                pcd_list.append(line)

        #o3d.visualization.draw_geometries(pcd_list)
        if show:
            key_to_callback = {}
            key_to_callback[ord("K")] = change_background_to_black
            #key_to_callback[ord("k")] = change_background_to_white
            # key_to_callback[ord("R")] = load_render_option
            key_to_callback[ord(",")] = capture_depth
            key_to_callback[ord(".")] = capture_image
            o3d.visualization.draw_geometries_with_key_callbacks(pcd_list, key_to_callback)

        return pcd_list
        # o3d.visualization.draw_geometries_with_animation_callback(pcd_list, rotate_view)


    def save_heatmap(self, metric_name, filename):
        heatmap = self.get_heatmap(metric_name)
        o3d.io.write_point_cloud(filename, heatmap)

    #visualize pointcloud with grid
    def visualize_cropped_point_cloud(self, cell_size, min_cell_index, max_cell_index, save=False, filename="test.pcd"):
        #visualize the pointcloud

        cropped_gt, _ = get_cropped_point_cloud(self.pointcloud_GT, self.min_bound, cell_size, min_cell_index, max_cell_index)
        cropped_gt.paint_uniform_color(RED_COLOR)
        cropped_candidate, _ = get_cropped_point_cloud(self.pointcloud_Cnd, self.min_bound, cell_size, min_cell_index, max_cell_index)
        #cropped_candidate = self.pointcloud_Cnd.crop(bbox)
        cropped_candidate.paint_uniform_color(GREEN_COLOR)
        
        if save:
            o3d.io.write_point_cloud(filename+"_gt.pcd", cropped_gt)
            o3d.io.write_point_cloud(filename+"_cnd.pcd", cropped_candidate)
        o3d.visualization.draw_geometries([cropped_gt, cropped_candidate])


    def iterate_cells(self):
        #iterate through all the Cells
        for i in range(int(self.cell_dim[0])):
            for j in range(int(self.cell_dim[1])):
                for k in range(int(self.cell_dim[2])):
                    min_cell_index = np.array([i, j, k])
                    max_cell_index = np.array([i+1, j+1, k+1])
                    #print(min_cell_index, max_cell_index)
                    yield min_cell_index, max_cell_index
  
    def visualize_points_per_cell(self):
        pcd_list = []
        for min_cell_index, max_cell_index in self.iterate_cells():
            
            cropped_gt, _ = get_cropped_point_cloud(self.pointcloud_GT, self.min_bound, self.cell_size, min_cell_index, max_cell_index)
            cropped_candidate, _ = get_cropped_point_cloud(self.pointcloud_Cnd, self.min_bound, self.cell_size, min_cell_index, max_cell_index)
            if cropped_gt.is_empty() or cropped_candidate.is_empty():
                #print("CELL: ", min_cell_index, max_cell_index, end="\t" )
                #print("EMPTY")
                pass
            else:
                #print()                
                pcd_list.append(cropped_gt)
                pcd_list.append(cropped_candidate)


        grid_lines = generate_grid_lines(self.min_bound, self.max_bound, self.cell_dim)

        for line in grid_lines:
            pcd_list.append(line)

        o3d.visualization.draw_geometries(pcd_list)

    # def compute_metric_prev(self, filename="test.json"):
    #     st_time = time.time()
    #     #iterate through all the Cells
    #     print ("Using this version of compute_metric")
    #     numIter = self.cell_dim[0] * self.cell_dim[1] * self.cell_dim[2]
    #     print("Number of Iterations: ", numIter)
    #     for i,(min_cell_index, max_cell_index) in tqdm(enumerate(self.iterate_cells()), total=numIter, desc="Computing Metrics"):
    #     # for min_cell_index, max_cell_index in self.iterate_cells():
            
    #         cropped_gt,gt_box = get_cropped_point_cloud(self.pointcloud_GT, self.min_bound, self.cell_size, min_cell_index, max_cell_index)
    #         cropped_candidate,cand_box = get_cropped_point_cloud(self.pointcloud_Cnd, self.min_bound, self.cell_size, min_cell_index, max_cell_index)
    #         if cropped_gt.is_empty() and cropped_candidate.is_empty():
    #             #print("EMPTY CELL: ", min_cell_index, max_cell_index, end="\t" )
    #             pass
    #         else:
    #             print ("Test2")
    #             print (type(cropped_gt))
    #             self.metriccells[str(min_cell_index)] =  MapCell(min_cell_index,cropped_gt, cropped_candidate, self.options,self.compute_flag, metric_fn=self.metric_fn)                
    #             # print(self.metriccells[str(min_cell_index)].metrics)

    #     print ("Time Taken: ", time.time() - st_time)                                                         
    #     self.save_metric(filename)


                                          


    def compute_metric_average(self, metric_results):
        #iterate through all the Cells
        metric_results[AVERAGE_STR] = {}
        metric_results[VARIANCE_STR] = {}
        for metric in METRICS_LIST:
            metric_list = [metric_results[METRICS_STR][key][metric] for key in metric_results[METRICS_STR].keys()]
            average_metric = np.mean(metric_list)
            print("Average ", metric, ": ", average_metric)
            metric_var = np.var(metric_list)
            print("Variance for ", metric, ": ", metric_var)
            metric_results[AVERAGE_STR][metric] = average_metric
            metric_results[VARIANCE_STR][metric] = metric_var
                                                                 
        return metric_results


    def save_metric(self, filename="test.json"):
        metric_results = {}   
        metric_results[METRICS_STR]={}
        for key in self.metriccells.keys():
           metric_results[METRICS_STR][key] = self.metriccells[key].metrics

        metric_results[CELL_SIZE_STR] = self.cell_size
        metric_results[CELL_DIM_STR] = self.cell_dim.flatten().tolist()
        metric_results[MIN_BOUND_STR] = self.min_bound.flatten().tolist()
        metric_results[MAX_BOUND_STR] = self.max_bound.flatten().tolist()
        metric_results[CONFIG_OPTIONS_STR] = self.options
        metric_results[CONFIG_GT_FILE_STR] = self.gt_file
        metric_results[CONFIG_CND_FILE_STR] = self.cnd_file

        metric_results = self.compute_metric_average(metric_results)


        # metric_results[DENSITY_GT_STR] = calculate_density(self.pointcloud_GT)
        # metric_results[DENSITY_CND_STR] = calculate_density(self.pointcloud_Cnd)

        # metric_results[CHAMFER_STR]=  -1 # calculate_chamfer_distance_metric(self.pointcloud_GT, self.pointcloud_Cnd)
        # metric_results[NORMALIZED_CHAMFER_STR]=  -1 #calculate_normalized_chamfer_distance_metric(self.pointcloud_GT, self.pointcloud_Cnd)
        # metric_results[HOUSDORFF_STR]=  -1 #calculate_hausdorff_distance_metric(self.pointcloud_GT, self.pointcloud_Cnd)
        # print("================Summary======================")
        # print("Our Metric: ", metric_results[AVERAGE_STR][QUALITY_STR])
        # print("Chamfer Metric: ", metric_results[CHAMFER_STR])
        # print("Normalized Chamfer Metric: ", metric_results[NORMALIZED_CHAMFER_STR])
        # print("Hausdorff Metric: ", metric_results[HOUSDORFF_STR])

        if filename is not None:
            with open(filename, 'w+') as fp:
                json.dump(metric_results, fp, indent=4)


    # def compute_metric(self, filename ="results/test.json"):

    #     st_time = time.time()
    #     from multiprocess import Process, Manager
    #     def f(d, min_cell_index,cropped_gt, cropped_candidate, bbox_gt, bbox_cnd):
    #         print ("Test3")
    #         print (type(cropped_gt))
    #         d[str(min_cell_index)] = MapCell(min_cell_index,cropped_gt, cropped_candidate, self.options,self.compute_flag, metric_fn=self.metric_fn, bbox_gt=bbox_gt, bbox_cnd=bbox_cnd)
    #         #print(d[str(min_cell_index)].metrics)

    #     manager = Manager()
    #     d = manager.dict()

    #     job = []
    #     numIter = self.cell_dim[0] * self.cell_dim[1] * self.cell_dim[2]
    #     print("Number of Iterations: ", numIter)
    #     for i,(min_cell_index, max_cell_index) in tqdm(enumerate(self.iterate_cells()), total=numIter, desc="Generating Jobs"):
    #         cropped_gt, bbox_gt = get_cropped_point_cloud(self.pointcloud_GT, self.min_bound, self.cell_size, min_cell_index, max_cell_index)
    #         cropped_candidate, bbox_cnd = get_cropped_point_cloud(self.pointcloud_Cnd, self.min_bound, self.cell_size, min_cell_index, max_cell_index)
    #         gt_is_empty = cropped_gt.is_empty()
    #         cand_is_empty = cropped_candidate.is_empty()
    #         if gt_is_empty and cand_is_empty:
    #             pass
    #         else:

    #             cropped_gt = np.asarray(cropped_gt.points)
    #             cropped_candidate = np.asarray(cropped_candidate.points)
    #             job.append(Process(target=f, args=(d, min_cell_index,cropped_gt, cropped_candidate, bbox_gt, bbox_cnd)))
    #             #job[-1].start()
                
    #     # Set number of cores to use here
    #     num_cores = 8
    #     for i, _ in tqdm(enumerate(range(0, len(job), num_cores)), total=int(len(job)/num_cores), desc="Running Jobs"):
    #         _ = [p.start() for p in job[i:i+num_cores]]
    #         _ = [p.join() for p in job[i:i+num_cores]]
    #         # Free up memory
    #         job[i:i+num_cores] = []
    #     print ("Starting Jobs ...")
    #     #_ = [p.start() for p in job]
    #     #_ = [p.join() for p in job]

    #     self.metriccells= copy.deepcopy(d)

    #     print ("Time Taken: ", time.time() - st_time)
    #     self.save_metric(filename)
            

def get_cropped_point_cloud_np(pcd, min_bound_source, chunk_size, min_cell_index, max_cell_index):
    min_bound, max_bound = get_cropping_bound(min_bound_source, chunk_size, min_cell_index, max_cell_index)
    #print(min_bound, max_bound, min_cell_index, max_cell_index)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    pcd_cropped = pcd.crop(bbox)
    pcd_cropped_np = np.asarray(pcd_cropped.points)
    return pcd_cropped_np, bbox  