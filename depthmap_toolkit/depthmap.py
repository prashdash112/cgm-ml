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

#mask object in the center of the depthmap
def focusObjectDetection(output):
  
  width = utils.getWidth()
  height = utils.getHeight()
  
  #parameters
  backgroundThreshold = 0.5 #threshold to differ object and background (hard edge)
  floorThreshold = 0.25 #threshold to differ object and floor (soft edge)
  depthThreshold = 0.3 #threshold to differ object and background (in meters)
  offset = 8 #amount of pixels from top where starts the selection (to skip unwanted edges)
  y1 = 8 #amount of pixels from left where starts the selection (to skip unwanted edges)
  y2 = height - 8 #amount of pixels from left where ends the selection (to skip unwanted edges)
  
  #top-down lines left part
  for y in range((y1 + y2) / 2, y1, -1):
    #find top of the object
    l = offset
    for x in range(offset, width):
      if (output[x][height - y - 1][1] > backgroundThreshold):
        l = x
        break
    #find bottom of the object
    r = offset
    for x in range(offset, width):
      if (output[width - x - 1][height - y - 1][1] > floorThreshold):
        r = width - x - 1
        break
    #connect top and down with a line
    if (l >= r):
      break;
    for x in range(l, r):
      output[x][height - y - 1][2] = 1
  
  #top-down lines right part
  for y in range((y1 + y2) / 2, y2):
    #find top of the object
    l = offset
    for x in range(offset, width):
      if (output[x][height - y - 1][1] > backgroundThreshold):
        l = x
        break
    #find bottom of the object
    r = offset
    for x in range(offset, width):
      if (output[width - x - 1][height - y - 1][1] > floorThreshold):
        r = width - x - 1
        break
    #connect top and down with a line
    if (l >= r):
      break;
    for x in range(l, r):
      output[x][height - y - 1][2] = 1
      
  #top-down cleaning
  for x in range(0, width):
    #clean left part
    current = 0
    valid = 1
    for y in range((y1 + y2) / 2, y1, -1):
      depth = utils.parseDepth(x, y)
      if (current == 0):
        current = depth
      if (depth):
        if (abs(depth - current) > depthThreshold):
          valid = 0
        else:
          current = depth
          valid = 1
      if (depth == 0 or valid == 0):
        output[x][height - y - 1][2] = 0
        
    #clean right part
    current = 0
    valid = 1
    for y in range((y1 + y2) / 2, y2):
      depth = utils.parseDepth(x, y)
      if (current == 0):
        current = depth
      if (depth):
        if (abs(depth - current) > depthThreshold):
          valid = 0
        else:
          current = depth
          valid = 1
      if (depth == 0 or valid == 0):
        output[x][height - y - 1][2] = 0
      

  return output
    

#show edges
def showEdges():
  fig = plt.figure()
  fig.canvas.mpl_connect('button_press_event', onclick)
  width = utils.getWidth()
  height = utils.getHeight()
  output = np.zeros((width, height, 3))
  for x in range(1, width - 1):
    for y in range(1, height - 1):
      depth = utils.parseDepth(x, y)
      mx = utils.parseDepth(x - 1, y)
      px = utils.parseDepth(x + 1, y)
      my = utils.parseDepth(x, y - 1)
      py = utils.parseDepth(x, y + 1)
      edge = (abs(depth - mx) + abs(depth - px) + abs(depth - my) + abs(depth - py))
      if edge > 0.015: #noise filter
        output[x][height - y - 1][1] = edge * 10

  output = focusObjectDetection(output)
  plt.imshow(output, extent=[0, height, 0, width])

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
