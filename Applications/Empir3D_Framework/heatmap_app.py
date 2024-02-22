import sys,os
sys.path.append(os.getcwd())

import numpy as np

#from MapMetricManager import parse_mapmetric_config

import argparse
import json

from system_constants import *

from Applications.Empir3D_Framework.MapMetricManager_v3 import MapMetricManager_v3
from Applications.Empir3D_Framework.MapMetricManager_base import parse_mapmetric_cells

def get_list_from_string_tuple(cell_index):
    cell_indx_val_tmp = cell_index.replace("(", "").replace(")", "").replace(",", "").split(" ")
    #print(cell_indx_val_tmp)
    cell_indx_val = np.array([int(x) for x in cell_indx_val_tmp if x.strip() != ""])
    return cell_indx_val

def parse_mapmetric_config(config_file):
    with open(config_file) as f:
        config = json.load(f)
        map_metric = MapMetricManager_v3(config[CONFIG_GT_FILE_STR], config[CONFIG_CND_FILE_STR], config[CELL_SIZE_STR], config[CONFIG_OPTIONS_STR])
        for cell_index in config[METRICS_STR]:
            # cell_indx_val_tmp = cell_index.replace("[", "").replace("]", "").split(" ")
            # cell_indx_val = np.array([int(x) for x in cell_indx_val_tmp if x.strip() != ""])
            map_metric.metriccells[cell_index] = parse_mapmetric_cells(get_list_from_string_tuple(cell_index), config[CONFIG_OPTIONS_STR], config[METRICS_STR][cell_index])
        return map_metric



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config json file")
    parser.add_argument("--metric", help="sub-metric to visualize; default is 'quality'", default="quality")

    args = parser.parse_args()

    mapmetric = parse_mapmetric_config(args.config)

    #Load json and get gt and cnd file paths
    with open(args.config, mode='r') as fp:
        config = json.load(fp)
        gt_path = config["gt_file"]
        cnd_path = config["cnd_file"]
        gt_name = gt_path.split("/")[-1].split(".")[0]
        cnd_name = cnd_path.split("/")[-1].split(".")[0]

    json_name = "gt_"+gt_name + "_" + "cnd_" +cnd_name

    metric = None
    if args.metric == "quality":
        metric = QUALITY_STR
    elif args.metric == "coverage":
        metric = COVERAGE_STR
    elif args.metric == "accuracy":
        metric = ACCURACY_STR
    elif args.metric == "resolution":
        metric = RESOLUTION_STR
    elif args.metric == "artifacts":
        metric = ARTIFACTS_STR
    else:
        print ("Invalid metric specified, computing for quality")
        metric = QUALITY_STR

    mapmetric.visualize_heatmap(metric)

    print ("Saving heatmap to heatmap.ply")

    mapmetric.save_heatmap(metric, "Evaluation/Results/Heatmaps/heatmap_"+args.metric+"_"+json_name+".ply")

if __name__ == "__main__":
    main()