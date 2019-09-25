#
# Child Growth Monitor - Free Software for Zero Hunger
# Copyright (c) 2019 Dr. Christian Pfitzner <christian.pfitzner@th-nuernberg.de> for Welthungerhilfe
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#


import numpy as np
import cv2
import pandas as pd

from pyntcloud import PyntCloud
from pyntcloud.io import write_ply

import sys
import os 


def fuse_point_cloud(points, rgb_vals, confidence, seg_vals): 
    df = pd.DataFrame(columns=['x', 'y', 'z','red', 'green', 'blue', 'c', 'seg'])

    df['x']     = points[:, 0]                              # saving carthesian coordinates
    df['y']     = points[:, 1]
    df['z']     = points[:, 2]

    df['red']   = rgb_vals[:, 2].astype(np.uint8)           # saving the color
    df['green'] = rgb_vals[:, 1].astype(np.uint8)
    df['blue']  = rgb_vals[:, 0].astype(np.uint8)

    df['c']     = confidence[:].astype(np.float)            # saving the confidence

    df['seg'] = seg_vals[:].astype(np.float)                # saving the segmentation


    new_pc      = PyntCloud(df)
    return new_pc


def write_color_ply(fname, points, color_vals, confidence):
    new_pc = fuse_point_cloud(points, color_vals, confidence)
    write_ply(fname, new_pc.points, as_text=True)





from cgm_fusion.calibration import get_intrinsic_matrix, get_extrinsic_matrix, get_k


'''
Function to get the depth from a point cloud as an image for visualization
'''
def get_viz_depth(ply_path):

    calibration_file =  '/whhdata/calibration.xml'
    if not os.path.exists(calibration_file):                # check if the califile exists
        logging.error ('Calibration does not exist')
        return 

    # get a default black image
    height         = 1200
    width          = 1920 
    nr_of_channels = 1
    viz_image = np.zeros((height,width,nr_of_channels), np.uint8)

    # get the points from the pointcloud
    try:
        cloud  = PyntCloud.from_file(ply_path)         # load the data from the files
    except ValueError as e: 
        logging.error(" Error reading point cloud ")
        logging.error(str(e))
        
        
    points = cloud.points.values[:, :3]                     # get x y z
    z      = cloud.points.values[:, 2]                      # get only z coordinate

    # calculate the min and the max value of z 
    # z_min  = np.amin(z)
    # z_max  = np.amax(z)

    # print('Minimum z value: ' + str(z_min))
    # print('Maximum z value: ' + str(z_max))

    # iterat of the points and calculat the x y coordinates in the image
    # get the data for calibration 
    # TODO make this an on function
    intrinsic  = get_intrinsic_matrix()
    ext_d      = get_extrinsic_matrix(4)

    r_vec      = ext_d[:3, :3]
    t_vec      = -ext_d[:3, 3]

    k1, k2, k3 = get_k()
    im_coords, _ = cv2.projectPoints(points, r_vec, t_vec, intrinsic[:3, :3], np.array([k1, k2, 0, 0]))


    # manipulate the pixels color value depending on the z coordinate
    # TODO make this a function
    for i, t in enumerate(im_coords):
        x, y = t.squeeze()
        x = int(np.round(x))
        y = int(np.round(y))
        if x >= 0 and x < width and y >= 0 and y < height:
            #z = 255*(z[i] - z_min) / z_max
            viz_image[x,y] = z[i] * 1000

    # return the image after pricessing
    cv2.imwrite('/tmp/depth_visualization.png', viz_image) 
    return viz_image

# '''
# Function to get the rgb from a point cloud as an image for visualization
# '''
# def get_viz_rgb(ply_path):


# '''
# Function to get the confidence from a point cloud as an image for visualization
# '''
# def get_viz_confidence(ply_path):


# '''
# Function to get the segmentation from a point cloud as an image for visualization
# '''
# def get_viz_segmentation(ply_path):

