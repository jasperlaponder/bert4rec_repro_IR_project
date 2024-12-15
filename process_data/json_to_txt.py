import os
import json
import numpy as np

def import_json_files(file_path):
    with open(file_path, 'r') as fp:
        data = json.load(fp)
    return data

def flatten_dict(d):
    flattened = {}
    for key, values in d.items():
        if isinstance(values, list):
            flattened[key] = [item for sublist in values for item in sublist]
        else:
            flattened[key] = values
    return flattened

def print_dict_to_txt(file_name, data_dict):
    with open(file_name, 'w') as f:
        for key, value in data_dict.items():
            for item in value:
                f.write(f"{key} {item}\n")

if __name__ == "__main__":
    tafeng_path = '/home/jasper/Uni/aprec_repro/process_data/tafeng_merged.json'
    instacart_path = '/home/jasper/Uni/aprec_repro/process_data/instacart_merged.json'
    dunnhumby_path = '/home/jasper/Uni/aprec_repro/process_data/dunnhumby_merged.json'
    paths = [tafeng_path, instacart_path, dunnhumby_path]
    for path in paths:
        data = import_json_files(path)
        flattened_data = flatten_dict(data)
        print_dict_to_txt(f'output{os.path.basename(path)}.txt', flattened_data)