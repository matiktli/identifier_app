from main_data_collector import main_data_collector
from main_train import main_train
from main_identify import main_identifier
from utils.file_renamer import main_renamer
from utils.id_assigner import main_id_assigner
from utils.raw_data_splitter import main_spliter
from utils.file_transformer import main_transform_raw_data
# import argparse
import sys

# parser = argparse.ArgumentParser(
#     description='Comand line tool for student face identification')

# # TODO

RAW_PATH = 'resources/face_data/raw'
CONFIG_PATH_PREFIX = 'resources/config/'
TRAIN_PATH, TEST_PATH = 'resources/face_data/train', 'resources/face_data/test'
MODELS_PATH_PREFIX = 'resources/models/'
CLASSIFIER_PATH = 'resources' + '/haarcascade_frontalface_default.xml'

print('\n--> [STEP 1] - Data collection...\n')
input('PROCEED...[key]')
students_no = int(input('Give students number: (3?)\n'))

for i in range(0, students_no):
    print(f'----> Collecting data{i}')
    main_data_collector()

print('\n--> [STEP 2] - Data transform...\n')
config_file_name = input('Give setup name: ***.json\n')

print('----> Prepering splitter')
main_id_assigner(raw_folder_path=RAW_PATH,
                 config_file_path=CONFIG_PATH_PREFIX + config_file_name + '.JSON')

print('----> Assigning ids to data')
main_renamer(raw_folder_path=RAW_PATH)

print('----> Transforming images for better model prediction...')
main_transform_raw_data(RAW_PATH)

print('----> Splitting data')
main_spliter(config_file_path=CONFIG_PATH_PREFIX + config_file_name + '.JSON',
             raw_folder_path=RAW_PATH, train_path=TRAIN_PATH, test_path=TEST_PATH)

print('\n--> [STEP 3] - Training model...\n')
input('PROCEED...[key]')
model_name = input('Give model name: ***.h5\n')
epochs = int(input('Give number of epochs: (10?)\n'))
steps = int(input('Give number of steps: (8?)\n'))
main_train(TRAIN_PATH, TEST_PATH,
           model_path=MODELS_PATH_PREFIX + model_name + '.h5', possible_choices=students_no,
           epochs=epochs, steps=steps)


print('\n TRAINED\n')
main_identifier(CONFIG_PATH_PREFIX + config_file_name + '.JSON',
                CLASSIFIER_PATH, MODELS_PATH_PREFIX + model_name + '.h5')
