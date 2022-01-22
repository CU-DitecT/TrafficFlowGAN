import numpy as np
import os
import itertools
from subprocess import check_output, check_call
from src.utils import check_exist_and_create, load_json
import json
import argparse
import shutil

parser = argparse.ArgumentParser()

parser.add_argument("--experiment_root_dir", default=None)
parser.add_argument("--mode", default=None)
parser.add_argument('--delete_existing_test_results', default=False, action='store_true',
                    help="delete the existing test results in each experiment folder first")


def search_and_repleace(hyper_params, key, value):
    for k, v in hyper_params.items():
        if k == key:
            hyper_params[k] = value
        if isinstance(v, dict):
            search_and_repleace(hyper_params[k], key, value)


def initialize_experiment(experiment_dir, hyper_params):
    check_exist_and_create(experiment_dir)
    file_path = os.path.join(experiment_dir, "experiment_setting.json")
    with open(file_path, "w") as f:
        json.dump(hyper_params, f, indent=4)


def replace_experiment_setting(experiment_dir, hyper_params):
    if not os.path.exists(experiment_dir):
        return False
    file_path = os.path.join(experiment_dir, "experiment_setting.json")
    with open(file_path, "w") as f:
        json.dump(hyper_params, f, indent=4)
    return True


def check_exist_and_add_suffix(experiment_dir, suffix, N=0):
    while os.path.exists(experiment_dir):
        N += 1
        if "round" in experiment_dir:
            list_char = experiment_dir.split("=")
            list_char[-1] = str(N)
            experiment_dir = "=".join(list_char)
        else:
            experiment_dir = experiment_dir + f"-{suffix:s}={N:d}"

    return experiment_dir, N

def check_exist_and_add_suffix_test(experiment_dir, suffix, N=0):
    while os.path.exists(os.path.join(experiment_dir, "test_result")):
        N += 1
        if "round" in experiment_dir:
            list_char = experiment_dir.split("=")
            list_char[-1] = str(N)
            experiment_dir = "=".join(list_char)
        else:
            experiment_dir = experiment_dir + f"-{suffix:s}={N:d}"

    return experiment_dir, N

def remove_all_test_results(experiment_root_dir):
    for folder in os.listdir(experiment_root_dir):
        if not os.path.isdir(os.path.join(experiment_root_dir, folder)):
            continue
        for file_or_folder in os.listdir(os.path.join(experiment_root_dir, folder)):
            if file_or_folder == "test_result":
                shutil.rmtree(os.path.join(experiment_root_dir, folder, file_or_folder))

            # launch job
args = parser.parse_args()
# data_dir = "data/ngsim/ngsim_v_dt=1"
Max_round = 1
experiment_root_dir = args.experiment_root_dir
# experiment_root_dir = "experiments_hyper_tune/ngsim_v_dt=1/pinn_drop"
json_path = os.path.join(experiment_root_dir, 'experiment_setting.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = load_json(json_path)

json_path = os.path.join(experiment_root_dir, 'tuning_params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
hyper_params = load_json(json_path)

# hyper_params would be like
# {
#   "alpha":[0.1, 0.4, 0.7, 0.9],
#   "n_train": [50,100,150,200,250,300,350,400,450,500]
# }

# delete current test results if applicable
if args.delete_existing_test_results is True:
    print("delete existing test results")
    remove_all_test_results(experiment_root_dir)

# get the loop index
flag = 0
while flag == 0:
    keys = list(hyper_params.keys())
    values = list(hyper_params.values())
    value_combines = list(itertools.product(*values))
    for combine in value_combines:
        experiment_name = ""
        for key, value in zip(keys, combine):
            search_and_repleace(params, key, value)
            experiment_name += f"{key}={value}-"
        # delete the last "_"
        experiment_name = experiment_name[:-1]

        experiment_dir = os.path.join(experiment_root_dir, experiment_name)
        # if folder exist, at suffix "round=x"
        N = 0
        suffix = "round"
        if args.mode == "train":
            experiment_dir, N = check_exist_and_add_suffix(experiment_dir, suffix, N=N)
            if N > Max_round-1:
                flag = 1
                continue
            initialize_experiment(experiment_dir, params)
            cmd = f"python main.py --experiment_dir {experiment_dir} --mode {args.mode}"
            print(cmd)
            check_output(cmd, shell=True)

        elif args.mode == "test":
            # update the experiment_setting in case of any difference between the training and test version
            experiment_dir, N = check_exist_and_add_suffix_test(experiment_dir, suffix, N=N)
            is_dir_exist = replace_experiment_setting(experiment_dir, params)
            if is_dir_exist is not True:
                flag=1
                continue
            cmd = f"python main.py --experiment_dir {experiment_dir} --mode {args.mode}"
            print(cmd)
            check_output(cmd, shell=True)
        else:
            raise ValueError("wrong mode")





    # cmd = f"python viz.py --experiment_dir {experiment_dir} --data_dir {data_dir} --debug --sudoku --interval --metrics_statistic"