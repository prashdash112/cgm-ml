#
# Child Growth Monitor - Free Software for Zero Hunger
# Copyright (c) 2019 Tristan Behrens <tristan@ai-guru.de> for Welthungerhilfe
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

import os
import sys
sys.path.insert(0, "..")
import numpy as np
from cgmcore import utils
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import shutil
import cv2
import face_recognition
import glob


input_path = "/whhdata/qrcode"
output_path_root = "/localssd/anondata/qrcode"
output_path_root = "/data/home/tbehrens/notebooks/cgm-ml/cgm_scripts/anondata/qrcode"
galleries_path_root = "/localssd/anondata/galleries"
galleries_path_root = "/data/home/tbehrens/notebooks/cgm-ml/cgm_scripts/anondata/galleries"


def main():
    '''
    Main runner code.
    - Copy data while maintaing the original directory structure
    - Remove personal info.
    '''
    
    # Make sure that destination folder exists.
    if os.path.exists(output_path_root) == False:
        os.makedirs(output_path_root)
        
    # Make sure that galleries folder exists.
    if os.path.exists(galleries_path_root) == False:
        os.makedirs(galleries_path_root)
    
    print("Getting data for anonymization. This might take a while...")
    # TODO remove this
    #qrcode_paths = glob.glob(os.path.join(input_path, "*"))
    #qrcode_paths = [path for path in paths if os.path.isdir(path) == True]
    qrcode_paths = ["/whhdata/qrcode/RJ_WHH_1134", "/whhdata/qrcode/MP_WHH_2438", "/whhdata/qrcode/RJ_WHH_2446", "/whhdata/qrcode/RJ_WHH_1350", "/whhdata/qrcode/MH_WHH_0323",
"/whhdata/qrcode/RJ_WHH_0339", "/whhdata/qrcode/RJ_WHH_0005", "/whhdata/qrcode/MP_WHH_2080"]
    #qrcode_paths = qrcode_paths[0:1]
    
    # This method is called in multiple processes.
    def process_qrcode_path(qrcode_path):
        processed_images_count = 0
        skipped_images_count = 0
        
        # Anonymize individual files.
        for dirpath, dirnames, filenames in os.walk(qrcode_path):
            qrcode_target_path = os.path.join(output_path_root, dirpath[len(input_path) + 1:])
            
            # Go through all the files in the folder.
            for filename in filenames:
                # Ignore non-measurement files and non images.
                if "measurements" in dirpath and (filename.endswith(".jpg") or filename.endswith(".png")):
                    source_path = os.path.join(dirpath, filename)
                    target_folder = os.path.join(output_path_root, dirpath[len(input_path) + 1:])
                    if os.path.exists(target_folder) == False:
                        os.makedirs(target_folder)
                    target_path = os.path.join(qrcode_target_path, filename)
                    has_been_processed = blur_faces_in_file(source_path, target_path)
                    
                    if has_been_processed == True:
                        processed_images_count += 1
                    else:
                        skipped_images_count += 1
            
        # Create thumbnail galleries.
        for dirpath, dirnames, filenames in os.walk(output_path_root):
            if dirpath.endswith("rgb") == False:
                continue
            qrcode_target_path = os.path.join(output_path_root, dirpath[len(input_path) + 1:])
            create_galleries(dirpath, filenames) 
                        
        return (processed_images_count, skipped_images_count)
    
    # Run this in multiprocess mode.
    results = utils.multiprocess(
        qrcode_paths, 
        process_method=process_qrcode_path, 
        process_individial_entries=True, 
        progressbar=True,
        number_of_workers=1,
        disable_gpu=True
    )
    
    # Count how many files have been processed.
    processed_images_count_total = 0
    skipped_images_count_total = 0
    for processed_images_count, skipped_images_count in results:
        processed_images_count_total += processed_images_count
        skipped_images_count_total += skipped_images_count_total
    print("Anonymized {}, skipped {}".format(processed_images_count_total, skipped_images_count_total)) 
    # TODO write statistics to log.
    
    print(results)
 

def blur_faces_in_file(source_path, target_path):

    # Read the image.
    image = cv2.imread(source_path)[:,:,::-1]
    
    # Rotate image 90degress to the right.
    image = np.swapaxes(image, 0, 1)

    # Scale image down for faster prediction.
    small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    
    # Find face locations.
    face_locations = face_recognition.face_locations(small_image, model="cnn")
    
    # Check if image should be used.
    should_be_used = should_image_be_used(source_path, len(face_locations))
    
    # Skip the image?
    if should_be_used == False:
        return False
    
    #if len(face_locations) == 0:
    #    return False
    
    # Blur the image.
    for top, right, bottom, left in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Extract the region of the image that contains the face.
        face_image = image[top:bottom, left:right]

        # Blur the face image.
        face_image = cv2.GaussianBlur(face_image, (99, 99), 30)

        # Put the blurred face region back into the frame image.
        image[top:bottom, left:right] = face_image

    # Rotate image back.
    image = np.swapaxes(image, 0, 1)
    
    # Write image to hard drive.
    cv2.imwrite(target_path, image[:,:,::-1])
    
    return True
    

def should_image_be_used(source_path, number_of_faces):
    """
    Determines if an image should be skipped or not.
    """
    
    # Cases for front scan.
    if "_100_" in source_path or "_200_" in source_path or "_104_" in source_path:
        
        # Artifact unusable.
        if number_of_faces == 0:
            return False
        
        # Need to evaluate output of face recognition model.
        elif number_of_faces == 1:
            return True
        
        # Artifact can be ignored for now.
        else:
            return False
        
    # Cases for 360 scan.
    elif "_101_" in source_path or "_201_" in source_path or "_107_" in source_path:
        
        # Might be useful.
        if number_of_faces == 0:
            return True
        
        # Need to evaluate output of face recognition model.
        elif number_of_faces == 1:
            return True
        
        # Might be useful.
        else:
            return True

    # Cases for back scan.
    elif "_102_" in source_path or "_202_" in source_path or "_110_" in source_path:

        # Might be useful based on child pose and number of people detected.
        if number_of_faces == 0:
            return True
        
        # Might be useful based on face blur outcome, child pose, number of people detected.
        elif number_of_faces == 1:
            return True
        
        # Might be useful
        else:
            return True
    
    assert False, "Should not happen! " + source_path

    
def create_galleries(dirpath, filenames):
    dirpath_split = dirpath.split("/")
    timestamp = dirpath_split[-2]
    qrcode = dirpath_split[-4]
    
    # Filter files by poses. Create one gallery per qrcode, per timestamp, and per pose.
    infixes = ["_100_", "_200_", "_104_", "_101_", "_201_", "_107_", "_102_", "_202_", "_110_"]
    for infix in infixes:
        filtered_filenames = [os.path.join(dirpath, filename) for filename in filenames if infix in filename]
        if len(filtered_filenames) == 0:
            continue

        # Create the gallery.
        gallery_file_name = "{}_{}_{}.jpg".format(qrcode, timestamp, infix.replace("_", "")) 
        gallery_file_path = os.path.join(galleries_path_root, gallery_file_name)
        utils.render_artifacts_as_gallery(
            filtered_filenames,
            targets=None,
            qr_code=qrcode,
            timestamp=timestamp, 
            num_columns=10, 
            target_size=(1920 // 8, 1080 // 8), 
            image_path=gallery_file_path,
            use_plt=False
        )
    
    
if __name__ == '__main__':
    main()

