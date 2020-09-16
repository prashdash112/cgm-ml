from glob2 import glob
from pathlib import Path
import sys
from typing import List

import numpy as np
from PIL import Image

REPO_DIR = Path(__file__).parents[1]
sys.path.append(str(REPO_DIR / 'depthmap_toolkit'))

from pcd2depth import convert_to_depthmap
import utils


def calculate_light_score(image: np.array) -> float:
    """Same as calculated on phone (keeping original Zhang's logic)

    Args:
        image: array of RGB pixels (float value in range 0-1) of shape (3, height, width)

    Returns:
        light score: floating point describing how good the light is in the imge
    """

    averageColor = sum(image) / float(image.size())  # RGB, averageColor has shape (3,)

    lightIntensity = (averageColor[0] + averageColor[1] + averageColor[2]) / 2  # 0-1.5
    lightOverbright = min(max(lightIntensity - 1.0, 0.0), 0.99)  # 0-0.99
    artifact_light_estimation = min(lightIntensity, 0.99) - lightOverbright  # 0-1
    return artifact_light_estimation


def get_dataset() -> List[str]:
    dataset_path = REPO_DIR.parent / 'cgm-ml-service' / 'data' / 'anon-bmz-version5.0_incomplete' / 'qrcode'
    list_of_qrcode_paths = [p for p in dataset_path.iterdir()]

    pcd_path = dataset_path / "1583438029-ekdtz7qyed/measure/1596993849899/pc/pc_1583438029-ekdtz7qyed_1596993849899_200_006.pcd"
    rgb_path = dataset_path / "1583438029-ekdtz7qyed/measure/1596993849899/rgb/rgb_1583438029-ekdtz7qyed_1596993849899_200_64350.850504759.jpg"
    return pcd_path, rgb_path


if __name__ == "__main__":
    pcd_path, rgb_path = get_dataset()

    calibration_path = str(REPO_DIR / 'depthmap_toolkit' / 'camera_calibration.txt')

    calibration: List[List[float]] = utils.parseCalibration(calibration_path)
    points: List[List[float]] = utils.parsePCD(pcd_path)  # list of 1000s of points where each point is a list of 4 floats

    depthmap_3channel = convert_to_depthmap(calibration, points)
    print(depthmap_3channel.shape)

    depthmap = depthmap_3channel[:, :,2]

    im = Image.fromarray(depthmap * 100)
    assert im.mode == 'F'
    im_rgb = im.convert('RGB')
    im_rgb.save('image.png')
