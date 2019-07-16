#
# Child Growth Monitor - Free Software for Zero Hunger
# Copyright (c) 2019 Dr. Christian Pfitzner <christian.pfitzner@th-nuernberg.de> for Welthungerhilfe
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
from numpy import size
import dbutils
import progressbar #@todo remove this
import logging

#import the neccessary packages for the sensor fusion
import cgm_fusion.utility
import cgm_fusion.calibration 
from cgm_fusion.fusion import apply_fusion 

# import core packages from cgm
from cgmcore.utils import load_pcd_as_ndarray
from cgmcore import  utils

# import packages for visualizationi 
from pyntcloud import PyntCloud
from timeit import default_timer as timer


# Connect to database.
main_connector = dbutils.connect_to_main_database()



''' 
    return the normalized timestamps with values between 0 and 1
'''
def get_timestamps_from_rgb(qr_code):
    connector1 = dbutils.connect_to_main_database()

    # get all artifacts of a certain unique qr code
    sql_statement  = "SELECT  path, type, tango_timestamp FROM artifact "
    sql_statement += " WHERE qr_code = '{}'".format(qr_code)
    sql_statement += " AND type = 'rgb'"

    all_rgb = connector1.execute(sql_statement, fetch_all=True)
    
    timestamps = [x[2] for x in all_rgb]
    path       = [x[0] for x in all_rgb]

    
    if( len(timestamps) == 0): 
        error = np.array([])
        return [error, path]
    
    timestamps      = np.asarray(timestamps)
    return [timestamps, path]



''' 
    import the timestamp from the pcd file from the header
'''
def get_timestamp_from_pcd(pcd_path): 
    filename  = str(pcd_path[0])
    infile    = open(filename, 'r')
    firstLine = infile.readline()

    # get the time from the header of the pcd file
    import re
    timestamp = re.findall("\d+\.\d+", firstLine)

    return float(timestamp[0])  # index error? IndexError




''' 
    get the timestamps out of the pcd files
'''
def get_timestamps_from_pcd(qr_code): 
    connector2 = dbutils.connect_to_main_database()

    sql_statement  = "SELECT  path FROM artifact "
    sql_statement += " WHERE qr_code = '{}'".format(qr_code)
    sql_statement += " AND type = 'pcd'" 
    path = connector2.execute(sql_statement, fetch_all=True)
    timestamps = np.array([])

    #iterate over all paths pointing to pcds
    for p in path: 
        
        try: 
            stamp = get_timestamp_from_pcd(p)
            timestamps = np.append(timestamps, stamp)
        except IndexError: 
            error = np.array([])
            print ("Error with timestamp")
            return [error, path]



    if( len(timestamps) == 0): 
        error = np.array([])
        return [error, path]
    
    return timestamps, path





''' 
    return the index of the element in a nparray to the target value
'''
def find_closest(A, target):
    #A must be sorted
    idx   = A.searchsorted(target)
    idx   = np.clip(idx, 1, len(A)-1)
    left  = A[idx-1]
    right = A[idx]
    idx  -= target - left < right - target
    return idx




def update_qrs(unique_qr_codes):
    # initialize the rrogress bar with the maxium number of unique qr codes
    bar = progressbar.ProgressBar(max_value=len(unique_qr_codes))
    qr_counter = 0

    for qr in unique_qr_codes:

        # update the qr code counter in every loop
        bar.update(qr_counter)
        qr_counter = qr_counter + 1

        logging.error(qr_counter)

        if(qr_counter < 688):
            continue

        logging.error(qr)

        if qr == "{qrcode}":
            continue
        if qr == "data":
            continue
        # if qr == "MH_WHH_0158":
        #     continue
        # if qr == "MH_WHH_0740":
        #     continue

        [norm_rgb_time, rgb_path] = get_timestamps_from_rgb(qr)
        [norm_pcd_time, pcd_path] = get_timestamps_from_pcd(qr)

        # check if a qr code has rgb and pcd, otherwise the previous function returned -1

        if ( size(norm_rgb_time) == 0 ):
            logging.error("wrong size of jpg")
            logging.error("size rgb: " + str(size(norm_rgb_time)))
            continue

        if ( size(norm_pcd_time) == 0 ): 
            logging.error("wrong size of pcd")    
            logging.error("size pcd: " + str(size(norm_pcd_time)))
            continue

        i = 0
        for pcd in norm_pcd_time:
            nn = find_closest(norm_rgb_time, pcd)

            logging.error("timestamp of rgb: " + "{0:.2f}".format(round(pcd,2))               + " with index " + str(i)) # + " path: " + str(pcd_path[i]))
            logging.error("timestamp of jpg: " + "{0:.2f}".format(round(norm_rgb_time[nn],2)) + " with index " + str(nn))# + " path: " + str(rgb_path[nn]))

            # get the original file path 
            path, filename = os.path.split(str(pcd_path[i]))

            pcd_file = pcd_path[i]
            pcd_file = pcd_file[0]
            jpg_file = rgb_path[nn]

            i = i+1

            cali_file = '/whhdata/calibration.xml'
            # the poin
            t cloud is fused and additionally the cloud is saved as ply in the same folder
            try: 
                fused_cloud = apply_fusion(cali_file, pcd_file, jpg_file)
            except: 
                print ("Something went wrong. ")
                continue
            # now save the new dat
            a to the folder
            fused_folder, pc_filename = os.path.split(str(pcd_file))

            pcd_path_old = pcd_file

            # replace the pcd and the pc_ in the path for fused data
            pc_filename = pcd_path_old.replace(".pcd", ".ply")
            pc_filename = pc_filename.replace("pc_",   "pcrgb_");

            logging.info("writing new fused data to: " + pc_filename)
            # print ("timing: " + str(start - timer()))
            try: 
                fused_cloud.to_file(pc_filename)
            except AttributeError :
                print (" skipping this file to save ") 
                continue
    bar.finish()




def main():
    # get a list of all unique qr_codes
    sql_statement = "SELECT DISTINCT artifact.qr_code FROM artifact  ORDER BY qr_code ASC;"
    unique_qr_codes = main_connector.execute(sql_statement, fetch_all=True)


    # todo: remove the (1) or (2) backup ?
    unique_qr_codes = [x[0] for x in unique_qr_codes]


    # initialze log file
    logging.basicConfig(filename='/tmp/command_update_fusion.log',level=logging.DEBUG, format='%(asctime)s %(message)s')


    #update_qrs(unique_qr_codes)
    
    # Run this in multiprocess mode.
    utils.multiprocess(unique_qr_codes, process_method=update_qrs, process_individial_entries=False, progressbar=False)
    print("Done.")





if __name__ == "__main__":
    main()
