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
pcd = []
for (dirpath, dirnames, filenames) in walk(pcd_file_or_dir):
    pcd = filenames
pcd.sort()
try:
    shutil.rmtree('output')
except:
    print('no previous data to delete')
os.mkdir('output')
os.mkdir('output/depth')
copyfile(pcd_file_or_dir + '/../camera_calibration.txt', 'output/camera_calibration.txt')
for i in range(len(pcd)):
    pcd2depth.process(pcd_file_or_dir + '/../camera_calibration.txt', pcd_file_or_dir + '/' + pcd[i], 'output/depth/' + pcd[i] + '.depth')
print('Data exported into folder output')
