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