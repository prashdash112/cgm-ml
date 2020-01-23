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


'''
import the necessary functions
'''
import sys
import os  
import cv2
import numpy as np
import pandas as pd
import open3d as o3d


from cgm_fusion.calibration import get_intrinsic_matrix, get_extrinsic_matrix, get_k
from cgm_fusion.utility import write_color_ply, fuse_point_cloud

from pyntcloud import PyntCloud


def apply_fusion(calibration_file, pcd_file, jpg_file, seg_path):
    ''' 
    check the path if everything is correct
    '''
    if not os.path.exists(pcd_file):                        # check all files exist
        logging.error ('Point cloud does not exist')
        return
        
    if not os.path.exists(jpg_file):                        # check if the jpg file exists
        logging.error ('Image does not exist')
        return 

    if not os.path.exists(seg_path):                        # check if segmentation exists
        logging.error('Segmentation not found')
        return

    if not os.path.exists(calibration_file):                # check if the califile exists
        logging.error ('Calibration does not exist')
        return 


    try:
        cloud      = PyntCloud.from_file(pcd_file)         # load the data from the files
    except ValueError:
        logging.error(" Error reading point cloud ")
        raise
        return

    jpg        = cv2.imread(jpg_file, -1)       
    jpg        = cv2.flip( jpg, 0 )

    seg        = cv2.imread(seg_path, -1)
    seg        = cv2.flip( seg, 0)

    hh, ww, _  = jpg.shape

    points     = cloud.points.values[:, :3]
    confidence = cloud.points.values[:, 3]
    
    # get the data for calibration
    intrinsic  = get_intrinsic_matrix()
    ext_d      = get_extrinsic_matrix(4)

    r_vec      = ext_d[:3, :3]
    t_vec      = -ext_d[:3, 3]

    k1, k2, k3 = get_k()
    im_coords, _ = cv2.projectPoints(points, r_vec, t_vec, intrinsic[:3, :3], np.array([k1, k2, 0, 0]))


    color_vals   = np.zeros_like(points)
    segment_vals = np.zeros_like(points)

    for i, t in enumerate(im_coords):
        x, y = t.squeeze()
        x = int(np.round(x))
        y = int(np.round(y))
        if x >= 0 and x < ww and y >= 0 and y < hh:
            color_vals[i, :]   = jpg[y, x]
            segment_vals[i, :] = seg[y, x] 




    # convert from pyntcloud to open3d
    cloud_open3d = o3d.io.read_point_cloud(pcd_file)

    # calculate the normals from the existing cloud
    cloud_open3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # print("Print a normal vector of the 0th point")
    # print(cloud_open3d.normals[0])
    # print("Print the normal vectors of the first 10 points")
    # print(np.asarray(cloud_open3d.normals)[:10, :])
    # print("x: " )
    # print(np.asarray(cloud_open3d.normals)[0,0])
    # print("y: ")
    # print(np.asarray(cloud_open3d.normals)[0,1])
    # print("z: ")
    # print(np.asarray(cloud_open3d.normals)[0,2])
    # print("")


    fused_point_cloud = fuse_point_cloud(points, color_vals, confidence, segment_vals, np.asarray(cloud_open3d.normals))

    return  fused_point_cloud





