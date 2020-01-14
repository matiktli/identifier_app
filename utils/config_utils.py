import json
import os.path


def get_config_json(path):
    if not os.path.isfile(path):
        return {'students': []}
    with open(path, 'r') as config_file:
        data = json.load(config_file)
        return data


def save_config_json(data, path):
    with open(path, 'w') as config_file:
        json.dump(data, config_file)
