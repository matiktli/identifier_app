import json


def get_config_json(path):
    with open(path, 'r') as config_file:
        data = json.load(config_file)
        return data


def save_config_json(data, path):
    with open(path, 'w') as config_file:
        json.dump(data, config_file)
