import os
import shutil
import sys
from os import walk
from shutil import copyfile

import pcd2depth

if len(sys.argv) != 2:
    print('You did not enter input folder')
    print('Usage: python convertpcd2depth.py <pcd file or directory>')
    sys.exit(1)

pcd_file_or_dir = sys.argv[1]
pcds = []
for (dirpath, dirnames, filenames) in walk(pcd_file_or_dir):
    pcds = filenames
pcds.sort()
try:
    shutil.rmtree('output')
except:
    print('no previous data to delete')
os.mkdir('output')
os.mkdir('output/depth')
copyfile(pcd_file_or_dir + '/../camera_calibration.txt', 'output/camera_calibration.txt')
for i in range(len(pcds)):
    calibration = pcd_file_or_dir + '/../camera_calibration.txt'
    pcd = pcd_file_or_dir + '/' + pcds[i]
    depthfile = 'output/depth/' + pcds[i] + '.depth'
    pcd2depth.process(calibration, pcd, depthfile)
print('Data exported into folder output')
