import matplotlib.pyplot as plt
import numpy as np
import utils
import zipfile

utils.setWidth(int(240 * 0.75))  # 180
utils.setHeight(int(180 * 0.75))  # 135
width = utils.getWidth()
height = utils.getHeight()


def convert_to_depthmap(calibration, points):
    output = np.zeros((width, height, 3))
    for point in points:
        v = utils.convert3Dto2D(calibration[1], point[0], point[1], point[2])
        x = int(width - v[0] - 1)
        y = int(height - v[1] - 1)
        if x >= 0 and y >= 0 and x < width and y < height:
            output[x][y][0] = point[3]
            output[x][y][2] = point[2]
    return output

    # first channel of output is still zero

def process(calibration_path, pcd, depthfile):

    # read PCD and calibration
    calibration = utils.parseCalibration(calibration_path)
    points = utils.parsePCD(pcd)

    # convert to depthmap
    output = convert_to_depthmap(calibration, points)

    # write depthmap
    with open('data', 'wb') as f:
        f.write(str(width) + 'x' + str(height) + '_0.001_255\n')
        for y in range(height):
            for x in range(width):
                depth = int(output[x][y][2] * 1000)
                confidence = int(output[x][y][0] * 255)
                f.write(chr(depth / 256))
                f.write(chr(depth % 256))
                f.write(chr(confidence))

    # zip data
    with zipfile.ZipFile(depthfile, "w", zipfile.ZIP_DEFLATED) as zip_:
        zip_.write('data', 'data')
        zip_.close()

    # visualsiation for debug
    #print str(width) + "x" + str(height)
    #plt.imshow(output)
    #plt.show()
