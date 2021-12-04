
import os

import configparser
config = configparser.ConfigParser()
config.read("config.ini")

os.remove(config['SGD_ARTEFACTS']['study_path'])

os.remove(config['NB_ARTEFACTS']['study_path'])
