import sys
sys.path.insert(0, "..")

from cgm_fusion import calibration
from cgm_fusion import fusion
from cgm_fusion import utility

import numpy
utility.get_viz_depth('/tmp/cloud_debug.ply')
