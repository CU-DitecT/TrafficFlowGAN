import os, datetime
import csv
import json
import logging
import math
import numpy as np


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting metrics_factory...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        if isinstance(list(d.values())[0], list):
            d = {k: [float(vi) for vi in v] for k, v in d.items()}
        else:
            d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

def check_and_make_dir(path):
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError:
            print("Creation of the directory " + str(path) + " failed")


def write_lines_to_file(fp, write_type, lines):
    with open(fp, write_type) as f:
        f.writelines([l + '\n' for l in lines])


def write_line_to_file(fp, write_type, line):
    with open(fp, write_type) as f:
        f.write(line + '\n')


def get_time_now():
    now = datetime.datetime.now()
    now = str(now).replace(" ", "-")
    now = now.replace(":", "-")
    return now


def write_to_log(s):
    fp = 'tmp/'
    check_and_make_dir(fp)
    fp += 'log.txt'
    t = get_time_now()
    write_line_to_file(fp, 'a+', t + ':: ' + s)


def write_line_to_csv(fp, line):
    with open(fp, "a+") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(line)
