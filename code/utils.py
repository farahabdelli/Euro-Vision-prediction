import json
import pickle as pkl

import pandas as pd


def load_json_data(target_file_path):
    with open(target_file_path) as json_file:
        data = json.load(json_file)
    return data


def load_csv_data(target_file_path):
    return pd.read_csv(target_file_path)


def load_model_data(target_file_path):
    with open(target_file_path, "rb") as pkl_file:
        model = pkl.load(pkl_file)
    return model
