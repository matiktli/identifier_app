from main_data_collector import main_data_collector
from main_train import main_train
from main_identify import main_identifier
from .utils.file_renamer import main_renamer
from .utils.id_assigner import main_id_assigner
from .utils.raw_data_splitter import main_spliter
# import argparse
import sys

# parser = argparse.ArgumentParser(
#     description='Comand line tool for student face identification')

# # TODO

RAW_PATH = 'resources/face_data/raw'
CONFIG_PATH_PREFIX = 'resources/config/'
TRAIN_PATH, TEST_PATH = 'resources/face_data/train', 'resources/face_data/test'
MODELS_PATH_PREFIX = 'resources/models/'

print('\n--> [STEP 1] - Data collection...\n')
input('PROCEED...[key]')

print('----> Collecting data')
main_data_collector()

print('\n--> [STEP 2] - Data transform...\n')
if input('PROCEED...[key] [SPACE to exit]') == ' ':
    input('TERMINATED...')
    sys.exit()
config_file_name = input('Give setup name: ***.json')

print('----> Prepering splitter')
main_id_assigner(raw_folder_path=RAW_PATH,
                 config_file_path=CONFIG_PATH_PREFIX + config_file_name + '.JSON')

print('----> Assigning ids to data')
main_renamer(raw_folder_path=RAW_PATH)

print('----> Splitting data')
main_spliter(config_file_path=CONFIG_PATH_PREFIX + config_file_name + '.JSON',
             raw_folder_path=RAW_PATH, train_path=TRAIN_PATH, test_path=TEST_PATH)

print('\n--> [STEP 3] - Training model...\n')
input('PROCEED...[key]')
model_name = input('Give model name: ***.h5')
epochs = int(input('Give number of epochs: (10?)'))
steps = int(input('Give number of steps: (8?)'))
students_no = int(input('Give students number: (5?)'))
main_train(TRAIN_PATH, TEST_PATH,
           model_path=MODELS_PATH_PREFIX + model_name + '.h5', possible_choices=students_no,
           epochs=epochs, steps=steps)
