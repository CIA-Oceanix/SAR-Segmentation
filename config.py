import json

from Rignak_Misc.path import get_local_file

CONFIG_FILENAME = get_local_file(__file__, 'config.json')


def get_config(key, config_filename=CONFIG_FILENAME):
    with open(config_filename, 'r') as file:
        config = json.load(file)
    return config[key]
