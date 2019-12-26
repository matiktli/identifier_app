import os
import json

CONFIG_PATH = '../resources/config/config_ids.JSON'
FACE_DATA_PATH = '../resources/face_data/raw'


def get_config_json(path=CONFIG_PATH):
    with open(path, 'r') as config_file:
        data = json.load(config_file)
        return data


def save_config_json(data, path=CONFIG_PATH):
    with open(path, 'w') as config_file:
        json.dump(data, config_file)


def assign_new_people(data, path=FACE_DATA_PATH):
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


config_data = get_config_json()
config_data = assign_new_people(config_data)
save_config_json(config_data)
