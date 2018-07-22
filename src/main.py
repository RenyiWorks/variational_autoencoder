import sys
import os
import configparser
import logging
import datetime
import time
import re

import constants
from data_generator import DataGenerator
from network_controller import NetworkController

DEFAULT_CONFIG_PATH = os.path.join(constants.config_path,'default.ini')

def main(ini_file_path):

    # CONFIG
    config = set_up_config(ini_file_path)

    # SETUP
    init_directories(config)

    # LOGGER
    logger = initialise_logger(config)
    logger.info('Program has started')

    # DATA
    data_generator = DataGenerator(config)
    #print(data_generator.data['dev'][0][:,:,0])
    #print(data_generator.data['dev'][0][:,:,1])

    # NETWORK & TRAIN
    network_controller = NetworkController(config)
    network_controller.train(data_generator)

    # TEST & STATISTICS
    # network_controller.test_and_log(data_generator)
    network_controller.sess.close()
  

# ----------------------------------------------------------------
# CONFIG


def set_up_config(ini_file):
    '''
        The default config file can be overriden by many
        new ones, thus this method helps in loading in the
        default ini file and place the current, fresh settings
        on the top of it.
    '''

    # Reading in the default settings and the new one
    default_config = configparser.ConfigParser()
    default_config.read(DEFAULT_CONFIG_PATH)
    new_config = configparser.ConfigParser()
    new_config.read(ini_file)

    # Overriding old variables with the new ones
    for header_name, header_dict in new_config.items():
        for variable_name, value in header_dict.items():
            try:
                default_config.set(header_name, variable_name, value)
            except ValueError:
                value = value.replace('%','%%') #to handle dateformat literals
                default_config.set(header_name, variable_name, value)

    return default_config  

# ----------------------------------------------------------------
# SETUP

def init_directories(config):

    # if user initialised root, skip
    if not config['PATH']['workspace_root'] == 'undefined':
        path = config['PATH']['workspace_root']
        if not os.path.exists(path):
            os.makedirs(path)
        return

    # retrieving data from config
    date_format = config['SETUP']['date_format']
    time_format = config['SETUP']['time_format']
    root_folder = config['PATH']['bin_path']

    # current date & time
    now = datetime.datetime.now()
    date = now.strftime(date_format)
    time = now.strftime(time_format)

    # create date folder
    path = root_folder + date
    if not os.path.exists(path):
        os.makedirs(path)

    # create time folder
    path = path + '/' + time
    if not os.path.exists(path):
        os.makedirs(path)

    # Save path
    config.set('PATH', 'workspace_root', path + '/')


# ----------------------------------------------------------------
# LOG

class ElapsedFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = record.created - self.start_time
        #using timedelta here for convenient default formatting
        elapsed = datetime.timedelta(seconds = elapsed_seconds)
        return "{} {:20s} {:8s} : {}".format(elapsed,
                                          record.filename,
                                          record.levelname,
                                          record.getMessage())


def initialise_logger(config):
    '''
        This function must be the first logging call, it initialises
        the main logger object
    '''

    # outgoing file path
    root_dir = config['PATH']['workspace_root']
    log_file = root_dir + 'info.log'

    # controller
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    # file logger
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(ElapsedFormatter())

    # stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(ElapsedFormatter())

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ----------------------------------------------------------------
# RUN


if __name__ == "__main__":
    ini_file_path = sys.argv[1]
    main(ini_file_path)

