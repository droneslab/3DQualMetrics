import numpy as np
import open3d as o3d
import sys,os
import argparse
import json
import subprocess

from QualityMetric_v4 import  calculate_quality_metric, calculate_color_metric, calculate_color_metric_cos, calculate_color_metric_hsv
import copy

def switch_channels(pcd, channel_order):
    # channel_order: list of 3 integers, each in range(3)
    # channel_order = [0, 1, 2] means no change
    # channel_order = [2, 1, 0] means swapping red and blue
    # channel_order = [1, 0, 2] means swapping green and blue
    pcd_colors = np.asarray(pcd.colors)
    pcd_colors = pcd_colors[:, channel_order]
    # pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    return pcd_colors

def swap_r_b(pcd):
    # Swap the red and blue channels
    pcd_colors = np.asarray(pcd.colors)
    new_colors = pcd_colors.copy()
    new_colors[:,[0,2]] = new_colors[:,[2,0]]
    return new_colors

def swap_g_b(pcd):
    # Swap the green and blue channels
    pcd_colors = np.asarray(pcd.colors)
    new_colors = pcd_colors.copy()
    new_colors[:,[1,2]] = new_colors[:,[2,1]]
    return new_colors

def swap_r_g(pcd):
    # Swap the red and green channels
    pcd_colors = np.asarray(pcd.colors)
    new_colors = pcd_colors.copy()
    new_colors[:,[0,1]] = new_colors[:,[1,0]]
    return new_colors

def shift_r(pcd, shift):
    # Shift the red channel by a given amount
    pcd_colors = np.asarray(pcd.colors)
    new_colors = pcd_colors.copy()
    new_colors[:,0] = (new_colors[:,0] + shift) % 256
    return new_colors

def shift_g(pcd, shift):
    # Shift the green channel by a given amount
    pcd_colors = np.asarray(pcd.colors)
    new_colors = pcd_colors.copy()
    new_colors[:,1] = (new_colors[:,1] + shift) % 256
    return new_colors

def shift_b(pcd, shift):
    # Shift the blue channel by a given amount
    pcd_colors = np.asarray(pcd.colors)
    new_colors = pcd_colors.copy()
    new_colors[:,2] = (new_colors[:,2] + shift) % 256
    return new_colors




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
        # color_metric = calculate_color_metric(pcdA_np, pcdA_colors, pcdB_np, pcdB_colors, e)
        color_metric_cos_hsv = calculate_color_metric_hsv(pcdA_np, pcdA_colors, pcdB_np, pcdB_colors, e)
        color_metric_cos_rgb = calculate_color_metric_cos(pcdA_np, pcdA_colors, pcdB_np, pcdB_colors, e)
        # pqm = calculate_quality_metric(pcdA, pcdB, e, 1, 1)
        return color_metric_cos_hsv, color_metric_cos_rgb

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

def random_degrade(pcd):
    # Use shift and swtich channel functions to randomly degrade the colors of the pointcloud

    rand1 = np.random.randint(0, 255)
    rand2 = np.random.randint(0, 255)
    rand3 = np.random.randint(0, 255)

    randa = np.random.randint(0, 2)
    randb = np.random.randint(0, 2)
    randc = np.random.randint(0, 2)

    pcd_colors = switch_channels(pcd, [randa, randb, randc])
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

    pcd_colors = shift_r(pcd, rand1)
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    pcd_colors = shift_g(pcd, rand2)
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    pcd_colors = shift_b(pcd, rand3)
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

    shift_values = [rand1, rand2, rand3]
    switch_values = [randa, randb, randc]

    return pcd, shift_values, switch_values

def unit_shift(pcd):
    pcd_colors = shift_r(pcd, 1)
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    pcd_colors = shift_g(pcd, 1)
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    pcd_colors = shift_b(pcd, 1)
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

    return pcd




def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test color metric')
    parser.add_argument('--pcdA', type=str, help='Path to reference pointcloud')
    parser.add_argument('--pcdB', type=str, help='Path to candidate pointcloud')
    parser.add_argument('-e', type=float, help='Threshold for color metric')
    parser.add_argument('--output', type=str, help='Path to output file')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    args = parser.parse_args()

    # Load pointclouds
    pcdA = o3d.io.read_point_cloud(args.pcdA)
    # pcdB = o3d.io.read_point_cloud(args.pcdB)

    pcdB = copy.deepcopy(pcdA)

    # output = {}
    # for i in range (10):
    #     pcdB, shift_values, switch_values = random_degrade(pcdB)
    #     color_metric, quality_metric = test_color_metric(pcdA, pcdB, args.e)
    #     print("Color metric: ", color_metric)
    #     # print("Quality metric: ", quality_metric)
    #     print("Shift values: ", shift_values)
    #     print("Switch values: ", switch_values)

    #     # Add to dictionary
    #     output[i] = {
    #         "color_metric": color_metric,
    #         "quality_metric": quality_metric,
    #         "shift_values": shift_values,
    #         "switch_values": switch_values
    #     }

    # # Write to file
    # with open(args.output, 'w') as f:
    #     json.dump(output, f)

    output = {}
    shift_counter = 0
    for i in range (int(args.epochs)):
        color_metric_cos_hsv, color_metric_cos_rgb = test_color_metric(pcdA, pcdB, args.e)
        pcdB = unit_shift(pcdB)
        shift_counter += 1
        # print("Color metric: ", color_metric)
        print ("Color metric (cosine similarity, HSV): ", color_metric_cos_hsv)
        print ("Color metric (cosine similarity, RGB): ", color_metric_cos_rgb)
        # print("Quality metric: ", quality_metric)
        print("Shift values: ", shift_counter)
        # print("Switch values: ", switch_values)

        # Add to dictionary
        output[i] = {
            # "color_metric": color_metric,
            "color_metric_cos_hsv": color_metric_cos_hsv,
            "color_metric_cos_rgb": color_metric_cos_rgb,
            # "quality_metric": quality_metric,
            "shift_values": shift_counter
        }

    # Write to file
    with open(args.output, 'w') as f:
        json.dump(output, f)


    # Plot
    from matplotlib import pyplot as plt

    # Plot the color metric
    # Shift values is x axis, color metric is y axis, two lines for two color metrics
    x = [shift for shift in range(int(args.epochs))]
    y1 = [output[i]["color_metric_cos_hsv"] for i in range(int(args.epochs))]
    y2 = [output[i]["color_metric_cos_rgb"] for i in range(int(args.epochs))]

    plt.plot(x, y1, label='Color metric (cosine similarity, HSV)')
    plt.plot(x, y2, label='Color metric (cosine similarity, RGB)')
    plt.xlabel('Shift values')
    plt.ylabel('Color metric')
    plt.title('Color metric vs shift values')
    plt.legend()
    plt.show()

    # def visualize_side_by_side(pcd1, pcd2):
    #     # Create a new point cloud that is a deep copy of the first one
    #     pcd1_transformed = copy.deepcopy(pcd1)
    #     # Shift the new point cloud to the right of the original one
    #     pcd1_transformed.translate([1, 0, 0], relative=False)
    #     # Draw both point clouds
    #     o3d.visualization.draw_geometries([pcd1, pcd1_transformed])

    # # Visualize both pointclouds
    # visualize_side_by_side(pcdA, pcdB)
    
    # Visualize pointcloud B
    # o3d.visualization.draw_geometries([pcdB])

    # Save pcdB
    # o3d.io.write_point_cloud("/home/yashturkar/Workspace/PCQM/build/samples/ford-color-test-pcdB_rand.ply", pcdB, write_ascii=True)

    # Calculate color metric
    # color_metric, quality_metric = test_color_metric(pcdA, pcdB, args.e)
    # print("Color metric: ", color_metric)
    # print("Quality metric: ", quality_metric)


if __name__ == "__main__":
    main()


'''

Source -> 100,90,80

Diff -> 20,0,20

Cand -> 80,90,100

'''