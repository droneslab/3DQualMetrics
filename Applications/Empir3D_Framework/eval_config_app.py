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


# import wandb

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
    # version = os.environ.get('MMM_VERSION')

    # print ("MapMetricManager Version = " + version)

    # version = args.version

    # if version == "1.0":
    #     from MapMetricManager_v1 import MapMetricManager_v1 as MapMetricManager
    # elif version == "2.0":
    #     from MapMetricManager_v2 import MapMetricManager_v2 as MapMetricManager
    # elif version == "3.0":
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
            mapManager = None
            print ("Starting Computation for ", cnd_path, "...")
            for size_ in tqdm(size, desc="Trials",total=(len(size)*len(eps))):
                for w1_ in weights:
                    for eps_ in eps:
                        metric_options = {"wc":w1_[0], "wt":w1_[1], "wa":w1_[2],"wr":w1_[3], "e": eps_}
                        # pqm_version = os.environ.get('PQM_VERSION')
                        # e = eps_
                        # cell_size = size_
                        # map_name = cnd_path.split("/")[-1].split(".")[0]
                        # pqm_run_name = f"pqm_v{pqm_version}_eps{e}_cell{cell_size}_{map_name}"
                        # run_1 = wandb.init(
                        #     # set the wandb project where this run will be logged
                        #     project="pqm-icra23-compute-m2",
                        #     name = pqm_run_name,
                        #     # track hyperparameters and run metadata
                        #     config={
                        #     "metric": f"pqm_v{pqm_version}", # Change this for chamfer and hausdorff
                        #     "eps": e, # chamfer and hausdorff N/A
                        #     "cell_size": cell_size, # chamfer and hausdorff N/A
                        #     "map_name": map_name,
                        #     }
                        # )
                        
                        
                        st_time = time.time()
                        if mapManager is None:
                            mapManager = MapMetricManager(gt_path, cnd_path, size_, metric_options=metric_options,compute_flag=compute_flag)
                        cmd_file_name = cnd_path.split("/")[-1].split(".")[0]
                        mapManager.reset(size_, metric_options=metric_options)
                        curr_json_file = os.path.join(save_path, "{}_size_{}_wc_{}_wt_{}_wa_{}_wr_{}_eps_{}.json".format(cmd_file_name, size_, w1_[0],w1_[1],w1_[2],w1_[3], eps_))
                        mapManager.compute_metric(curr_json_file)

                        print ("============Total Time Taken: ", time.time() - st_time)
                        #mapManager.visualize()

                        #===wandb logging========================================================
                        # log_dict ={}
                        # log_dict["time_taken"] = time.time() - st_time

                        # metric_results = mapManager.metric_results


                        
                        # for metric in METRICS_LIST:
                        #     log_dict[AVERAGE_STR+"_"+metric] =  metric_results[AVERAGE_STR][metric]
                        #     log_dict[VARIANCE_STR+"_"+metric] =  metric_results[VARIANCE_STR][metric]
                        
                        # log_dict["metric_value"] = metric_results[AVERAGE_STR][QUALITY_STR]

                        # run_1.log(log_dict)
                        # artifact_name = pqm_run_name#+".json"
                        # artifact = wandb.Artifact(artifact_name, type="json_results")
                        # artifact.add_file(curr_json_file)
                        # run_1.log_artifact(artifact)
                        # run_1.finish()
                        
                        #=========================================================================


        #mapManager = MapMetricManager(args.gt, args.cnd, args.size, metric_options=metric_options)



if __name__ == "__main__":
    main()

