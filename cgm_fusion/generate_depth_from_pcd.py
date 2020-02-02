import sys
sys.path.insert(0, "..")

#from cgm_fusion import calibration
from cgm_fusion import fusion
from cgm_fusion import utility

import numpy


# fusion.get_depth_image_from_point_cloud(calibration_file="dummy", pcd_file="/tmp/cloud_debug.ply", output_file="dummy")
utility.get_depth_channel(ply_path="/tmp/cloud_debug.ply")

