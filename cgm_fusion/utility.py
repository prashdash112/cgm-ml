import numpy as np
import cv2
import pandas as pd

from pyntcloud import PyntCloud
from pyntcloud.io import write_ply


def fuse_point_cloud(points, rgb_vals, confidence): 
    df = pd.DataFrame(columns=['x', 'y', 'z','red', 'green', 'blue', 'c'])
    df['x'] = points[:, 0]
    df['y'] = points[:, 1]
    df['z'] = points[:, 2]
    df['red'] = rgb_vals[:, 2].astype(np.uint8)
    df['green'] = rgb_vals[:, 1].astype(np.uint8)
    df['blue'] = rgb_vals[:, 0].astype(np.uint8)

    df['c'] = confidence[:].astype(np.float)

    new_pc = PyntCloud(df)
    return new_pc

def write_color_ply(fname, points, color_vals, confidence):
    new_pc = fuse_point_cloud(points, color_vals, confidence)
    write_ply(fname, new_pc.points, as_text=True)