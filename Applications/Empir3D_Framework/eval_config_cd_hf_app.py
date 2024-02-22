import sys
sys.path.append('../../PQM')

import argparse
import json
import os
from tqdm import tqdm
import os

from system_constants import *

# Example usage:
# python eval_config_app.py --config config.json

# config["gt_path"] = "gt.pcd"
# config["cnd_paths"] = ["fast.pcd", "lego.pcd"]
# config["save_path"] = "results/"
# config["e"] = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
# config["MPD"] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# config["size"] = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


# Wi -> Incompteness
# Wt -> Artifacts
# Wr -> Resolution
# Wa -> Accuracy

import time


import wandb


import subprocess as sp

def calcChamferDistance(gt, cnd):
    cd = sp.check_output(["pdal", "chamfer", gt, cnd])
    cd = cd.decode("utf-8")
    cd = cd.splitlines()
    cd = cd[1].split(":")
    cd = cd[1]
    cd = cd.strip()
    cd = float(cd[:-1])
    return cd

def calcHausdorffDistance(gt, cnd):
    hd = sp.check_output(["pdal", "hausdorff", gt, cnd])
    hd = hd.decode("utf-8")
    hd = hd.splitlines()
    hd = hd[6].split(":")
    hd = hd[1]
    hd = float(hd[:-1])
    return hd


metric_to_func = {
    CHAMFER_STR: calcChamferDistance,
    HOUSDORFF_STR: calcHausdorffDistance
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config json file")
    # Add 4 options for compute mode, gpu, cpu, gpu-batch, cpu-multi
    compute_mode = parser.add_mutually_exclusive_group()
    compute_mode.add_argument("--gpu", action="store_true", help="compute mode gpu")
    compute_mode.add_argument("--cpu", action="store_true", help="compute mode cpu")
    compute_mode.add_argument("--gpu_batch", action="store_true", help="compute mode gpu-batch")
    compute_mode.add_argument("--cpu_multi", action="store_true", help="compute mode cpu-multi")
    args = parser.parse_args()

    # Read environment variables for version
    version = os.environ.get('MMM_VERSION')

    print ("MapMetricManager Version = " + version)

    # version = args.version

    if version == "1.0":
        from MapMetricManager_v1 import MapMetricManager_v1 as MapMetricManager
    elif version == "2.0":
        from MapMetricManager_v2 import MapMetricManager_v2 as MapMetricManager
    elif version == "3.0":
        from MapMetricManager_v3 import MapMetricManager_v3 as MapMetricManager

    # Set compute flag 1 for gpu, 2 for cpu, 3 for gpu-batch, 4 for cpu-multi
    compute_flag = 0
    if parser.parse_args().gpu:
        compute_flag = 1
    elif parser.parse_args().cpu:
        compute_flag = 2
    elif parser.parse_args().gpu_batch:
        compute_flag = 3
    elif parser.parse_args().cpu_multi:
        compute_flag = 4

    
    config = {}
    metric_list =[CHAMFER_STR, HOUSDORFF_STR]

    with open(args.config, mode='r') as fp:
        config = json.load(fp)

        gt_path = config[CONFIG_GT_FILE_STR]
        cnd_paths = config[CONFIG_CND_FILE_STR]
        save_path = config[CONFIG_SAVE_PATH_STR]

        size = config[CELL_SIZE_STR]
        weights = config[CONFIG_WEIGHTS_STR]
        eps = config[CONFIG_EPS_STR]

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(gt_path):
            print("gt path not exist")
            return
        
        for cnd_path in cnd_paths:
            if not os.path.exists(cnd_path):
                print("cnd path not exist")
                continue

            print ("")


            for metric_name in metric_list:

                map_name = cnd_path.split("/")[-1].split(".")[0]
                pqm_run_name = f"{metric_name}_{map_name}"
                run_1 = wandb.init(
                    # set the wandb project where this run will be logged
                    project="pqm-icra23-compute-m2",
                    name = pqm_run_name,
                    # track hyperparameters and run metadata
                    config={
                    "metric": metric_name, # Change this for chamfer and hausdorff
                    "eps": "NA", # chamfer and hausdorff N/A
                    "cell_size": "NA", # chamfer and hausdorff N/A
                    "map_name": map_name,
                    }
                )
                
                
                st_time = time.time()

                metric_val = metric_to_func[metric_name](gt_path, cnd_path)

                print ("============Total Time Taken: ", time.time() - st_time)
                #mapManager.visualize()

                #===wandb logging========================================================
                log_dict ={}
                log_dict["time_taken"] = time.time() - st_time

                log_dict["metric_value"] = metric_val

                run_1.log(log_dict)
                run_1.finish()

                
                #=========================================================================


        #mapManager = MapMetricManager(args.gt, args.cnd, args.size, metric_options=metric_options)



if __name__ == "__main__":
    main()

