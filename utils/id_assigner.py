import os
import json
from config_utils import *

CONFIG_PATH = '../resources/config/config_ids_4.JSON'
FACE_DATA_PATH = '../resources/face_data/raw'


def __assign_new_people(data, path=FACE_DATA_PATH):
    existing_ids = list(map(lambda student:
                            student['name'], data['students']))

    for dir_name in os.listdir(path):

        if dir_name not in existing_ids:
            last_id = len(data['students']) - 1
            new_person = {'id': last_id + 1, 'name': dir_name}
            print(
                f'Found new person: [{dir_name}] assigning id: [{last_id + 1}].')
            data['students'].append(new_person)

    return data


# Main func of file
def main_id_assigner(config_file_path=CONFIG_PATH, raw_folder_path=FACE_DATA_PATH):
    config_data = get_config_json(config_file_path)
    config_data = __assign_new_people(config_data, raw_folder_path)
    save_config_json(config_data, config_file_path)


main_id_assigner()
