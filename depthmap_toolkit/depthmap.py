import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import utils
import zipfile

#export data
def export(type, filename):
  if type == 'obj':
    utils.exportOBJ('export/' + filename)
  if type == 'pcd':
    utils.exportPCD('export/' + filename)

#click on data
last = [0, 0, 0]
def onclick(event):
  width = utils.getWidth()
  height = utils.getHeight()
  if event.xdata is not None and event.ydata is not None:
    x = width - int(event.ydata) - 1
    y = height - int(event.xdata) - 1
    if x > 1 and y > 1 and x < width - 2 and y < height - 2:
      depth = utils.parseDepth(x, y)
      if depth:
        res = utils.convert2Dto3D(calibration[1], x, y, depth)
        if res:
          diff = [last[0] - res[0], last[1] - res[1], last[2] - res[2]]
          dst = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
          res.append(dst)
          print('x=' + str(res[0]) + ', y=' + str(res[1]) + ', depth=' + str(res[2]) + ', diff=' + str(res[3]))
          last[0] = res[0]
          last[1] = res[1]
          last[2] = res[2]
          return
      print('no valid data')

#get input
def process(plt, dir, depth, rgb):

  #extract depthmap
  with zipfile.ZipFile(dir + '/depth/' + depth, 'r') as zip_ref:
    zip_ref.extractall('.')
  utils.parseData('data')

  #read rgb data
  global im_array
  if rgb:
    width = utils.getWidth()
    height = utils.getHeight()
    pil_im = Image.open(dir + '/rgb/' + rgb)
    pil_im = pil_im.resize((width, height), Image.ANTIALIAS)
    im_array = np.asarray(pil_im)
  else:
    im_array = 0

  #parse calibration
  global calibration
  calibration = utils.parseCalibration(dir + '/camera_calibration.txt')

#show result
def showResult():
  fig = plt.figure()
  fig.canvas.mpl_connect('button_press_event', onclick)
  width = utils.getWidth()
  height = utils.getHeight()
  output = np.zeros((width, height, 3))
  for x in range(width):
    for y in range(height):
      depth = utils.parseDepth(x, y)
      if (depth):
        #convert ToF coordinates into RGB coordinates
        vec = utils.convert2Dto3D(calibration[1], x, y, depth)
        vec[0] += calibration[2][0]
        vec[1] += calibration[2][1]
        vec[2] += calibration[2][2]
        vec = utils.convert3Dto2D(calibration[0], vec[0], vec[1], vec[2])

        #set output pixel
        output[x][height - y - 1][0] = utils.parseConfidence(x, y)
        if im_array and vec[0] > 0 and vec[1] > 1 and vec[0] < width and vec[1] < height:
          output[x][height - y - 1][1] = im_array[vec[1]][vec[0]][1] / 255.0 #test matching on RGB data
        output[x][height - y - 1][2] = 1.0 - min(depth / 2.0, 1.0) #depth data scaled to be visible
  plt.imshow(output, extent=[0, height, 0, width])
