import numpy as np
import json
from pathlib import Path
import os

def logsumexp_trick(x):
    """
    For exp(x_i) / sum_i exp(x_i) avoid over flow
    """

    c = x.max()
    y = c + np.log(np.sum(np.exp(x - c)))

    return np.exp(x - y)

def save_dict_with_arrays_as_json(dictionary, file_path):
    """
    Save a dictionary to a JSON file, converting NumPy arrays to lists for compatibility.

    Parameters:
        dictionary (dict): The dictionary to save.
        file_path (str): The file path to save the dictionary.
    """

    def convert_arrays(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_arrays(v) for k, v in obj.items()}
        return obj

    dict_to_save = convert_arrays(dictionary)

    # Save the dictionary as JSON
    with open(file_path, 'w') as f:
        json.dump(dict_to_save, f, indent=4)

def load_json_with_arrays(file_path):
    """
    Load a dictionary from a JSON file, converting lists back into NumPy arrays where necessary.

    Parameters:
        file_path (str): The file path to load the dictionary from.

    Returns:
        dict: The loaded dictionary with NumPy arrays where applicable.
    """

    def convert_back_to_arrays(obj):
        if isinstance(obj, list):
            try:
                return np.array(obj)
            except ValueError:
                return obj
        if isinstance(obj, dict):
            return {k: convert_back_to_arrays(v) for k, v in obj.items()}
        return obj

    with open(file_path, 'r') as f:
        data = json.load(f)

    return convert_back_to_arrays(data)

def ensure_dir_exists(directory):
    """
    Check if the directory exists, and if not, create it.

    Parameters:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist. Creating it now.")
        os.makedirs(directory)
    else:
        print(f"Directory '{directory}' already exists.")

def check_and_load_dict(file_path):
    """
    Check if the file exists, if yes do nothing, else call script to generate data.

    Parameters:
        file_path (str): file path

    """
    if os.path.exists(file_path):
        print(f"File found: {file_path}")
    else:
        print(f"File not Found. Generating and Saving...")
        # TOBEDONE

