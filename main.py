from main_data_collector import main_data_collector
from main_train import main_train
from main_identify import main_identifier
from .utils.id_assigner import main_id_assigner
import argparse

# parser = argparse.ArgumentParser(
#     description='Comand line tool for student face identification')

# # TODO

RAW_PATH = 'resources/face_data/raw'
CONFIG_PATH_PREFIX = 'resources/config/'

input('--> [STEP 1] - Data collection...')
main_data_collector()
config_file_name = input('--> [STEP 2] - Data transform...')
main_id_assigner(raw_folder_path=RAW_PATH,
                 config_file_path=CONFIG_PATH_PREFIX + config_file_name)
