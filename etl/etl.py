"""
Issue-16 | Export Training Data from Firebase to Cloud Data Storage


"""
import logging
import os
import glob2
import json
from etl.data_reader import DataReader
from etl.data_writer import DataWriter
from etl.data_loader import DataLoaderFactory
import configparser

log = logging.getLogger(__name__)

# 1. get all qr codes which can be used
# 2. setup processing of individual qr code


class ETL:
    def __init__(self):
        """
        read parameters from config file for pointcloud or voxelgrid
        """
        self.config = configparser.ConfigParser()
        self.data_reader = None
        self.data_writer = None
        self.data_loader = None

    def initialize(self, config_path):
        self.config.read(config_path)
        dataset_path = self.config['DataReader']['dataset_path']
        output_targets = self.config['DataReader']['output_targets'].split(',')
        self.data_reader = DataReader(dataset_path, output_targets)
        input_type = self.config['MAIN']['input_type']
        self.data_loader = DataLoaderFactory.factory(input_type)
        self.data_writer = DataWriter()

    def run(self):
        log.info("ETL: RUN")
        log.info("Create qr code dictionary")
        qrcode_dict = self.data_reader.create_qrcodes_dictionary()
        log.info("Created qr code dictionary. Number of qr codes = %d" %
                 len(qrcode_dict))
        # push each qr code to a queue
        # process each qr code, sending the output to the writer
        # writer creates the necessary files (h5)
        return
        # TODO Work in progress to load data and send it to writer
        for qrcode in qrcode_dict:
            log.info("Processing QR code %s" % qrcode)
            targets, jpg_paths, pcd_paths = qrcode_dict[qrcode]
            x_input, file_path = self.data_loader.load_data(jpg_paths,
                                                            pcd_paths)
            y_output = targets
            # 3 parts of output are : x_input, y_output, file_path
            self.data_writer.write(qrcode, x_input, y_output, file_path)
            log.info("Completed processing QR code %s" % qrcode)