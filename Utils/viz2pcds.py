# Script to view 2 pointclouds side by side
# Read json, gt_file and cnd_file, former is gt, latter is cnd

import numpy as np
import json
import os
import sys
import argparse
import open3d as o3d

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config json file")
    parser.add_argument("--type", type=str, default="overlay")
    args = parser.parse_args()
    config = {}
    with open(args.config, mode='r') as fp:
        config = json.load(fp)
        gt_path = config["gt_file"]
        cnd_path = config["cnd_file"][0]

    print ("gt_path: ", gt_path)
    print ("cnd_path: ", cnd_path)

    gt = o3d.io.read_point_cloud(gt_path)
    cnd  = o3d.io.read_point_cloud(cnd_path)

    if args.type == "2windows":
        # View 2 pointclouds side by side
        viz_gt = o3d.visualization.Visualizer()
        viz_gt.create_window()
        viz_gt.add_geometry(gt)
        # viz_gt.get_render_option().point_size = 1
        viz_cnd = o3d.visualization.Visualizer()
        viz_cnd.create_window()
        viz_cnd.add_geometry(cnd)
        # viz_cnd.get_render_option().point_size = 1
        viz_gt.run()
        viz_cnd.run()
    elif args.type == "overlay":
        # View 2 pointclouds overlaid
        viz = o3d.visualization.Visualizer()
        viz.create_window()
        # Color GT to green and CND to red
        gt.paint_uniform_color([0,1,0]) # Green
        cnd.paint_uniform_color([1,0,0]) # Red
        xyz_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0, origin=[0,0,0])
        viz.add_geometry(xyz_axis)
        viz.add_geometry(gt)
        viz.add_geometry(cnd)
        viz.run()


        

if __name__ == "__main__":
    main()

