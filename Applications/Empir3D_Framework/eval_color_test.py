import numpy as np
import open3d as o3d
import sys,os
import argparse
import json

from QualityMetric_v4 import  calculate_quality_metric, calculate_color_metric


def test_color_metric(pcdA,pcdB,e):
    # Take two pointclouds and split into 10 subpointclouds
    pcdA_np = np.asarray(pcdA.points)
    pcdB_np = np.asarray(pcdB.points)
    pcdA_colors = np.asarray(pcdA.colors)
    pcdB_colors = np.asarray(pcdB.colors)

    # Check if colors are present
    if len(pcdA_colors) == 0 or len(pcdB_colors) == 0:
        print("No colors present in pointclouds")
        return 0, 0
    else :
        print("Colors present in pointclouds")
        # return calculate_color_metric(pcdA, pcdB, e), calculate_quality_metric(pcdA, pcdB, e, 1, 1)
        color_metric = calculate_color_metric(pcdA_np, pcdA_colors, pcdB_np, pcdB_colors, e)
        # pqm = calculate_quality_metric(pcdA, pcdB, e, 1, 1)
        return color_metric, 1

    # # Split the pointclouds into 10 subpointclouds
    # pcdA_np_split = np.array_split(pcdA_np, 10)
    # pcdB_np_split = np.array_split(pcdB_np, 10)
    # pcdA_colors_split = np.array_split(pcdA_colors, 10)
    # pcdB_colors_split = np.array_split(pcdB_colors, 10)

    # # Calculate the color metric for each subpointcloud
    # color_metric = []
    # for i in range(10):
    #     pcdA_split = o3d.geometry.PointCloud()
    #     pcdA_split.points = o3d.utility.Vector3dVector(pcdA_np_split[i])
    #     pcdA_split.colors = o3d.utility.Vector3dVector(pcdA_colors_split[i])
    #     pcdB_split = o3d.geometry.PointCloud()
    #     pcdB_split.points = o3d.utility.Vector3dVector(pcdB_np_split[i])
    #     pcdB_split.colors = o3d.utility.Vector3dVector(pcdB_colors_split[i])
    #     color_metric.append(calculate_color_metric(pcdA_split, pcdB_split, e))

    # # Calculate the average color metric 
    # average_color_metric = sum(color_metric) / len(color_metric)

    # # Calculate quality metric for each subpointcloud
    # quality_metric = []
    # for i in range(10):
    #     pcdA_split = o3d.geometry.PointCloud()
    #     pcdA_split.points = o3d.utility.Vector3dVector(pcdA_np_split[i])
    #     pcdA_split.colors = o3d.utility.Vector3dVector(pcdA_colors_split[i])
    #     pcdB_split = o3d.geometry.PointCloud()
    #     pcdB_split.points = o3d.utility.Vector3dVector(pcdB_np_split[i])
    #     pcdB_split.colors = o3d.utility.Vector3dVector(pcdB_colors_split[i])
    #     quality_metric.append(calculate_quality_metric(pcdA_split, pcdB_split, e, 1, 1))

    # # Calculate the average quality metric
    # average_quality_metric = sum(quality_metric) / len(quality_metric)

    # return average_color_metric, average_quality_metric

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test color metric')
    parser.add_argument('--pcdA', type=str, help='Path to reference pointcloud')
    parser.add_argument('--pcdB', type=str, help='Path to candidate pointcloud')
    parser.add_argument('-e', type=float, help='Threshold for color metric')
    args = parser.parse_args()

    # Load pointclouds
    pcdA = o3d.io.read_point_cloud(args.pcdA)
    pcdB = o3d.io.read_point_cloud(args.pcdB)

    # Calculate color metric
    color_metric, quality_metric = test_color_metric(pcdA, pcdB, args.e)
    print("Color metric: ", color_metric)
    print("Quality metric: ", quality_metric)

if __name__ == "__main__":
    main()