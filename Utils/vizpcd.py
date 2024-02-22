# Script to visualize a single pointcloud, use argeparse to specify the path to the pointcloud

import numpy as np
import open3d as o3d
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to pointcloud")
    parser.add_argument("--type", type=str, default="heightramp")
    args = parser.parse_args()
    pcd = o3d.io.read_point_cloud(args.path)
    if args.type == "heightramp":
        # Visualize heightramp
        viz = o3d.visualization.Visualizer()
        viz.create_window()
        pcd.colors = o3d.utility.Vector3dVector(np.array([[0,0,1] for i in range(len(pcd.points))]))
        viz.add_geometry(pcd)
        viz.run()
    elif args.type == "rgb":
        pass

if __name__ == "__main__":
    main()