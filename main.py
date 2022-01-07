import argparse
import logging
import os
import sys

import numpy as np
import torch

from src.utils import Params, save_checkpoint, load_checkpoint
from src.models.flow import RealNVP
# from src.metrics import instantiate_losses, instantiate_metrics, functionalize_metrics
from src.utils import set_logger, delete_file_or_folder
from src.training import training

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', default='experiments/arz',
                    help="Directory containing experiment_setting.json")
parser.add_argument('--data_dir', default='data/arz', help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, file location containing weights to reload")
parser.add_argument('--mode', default='train',
                    help="train, test, or train_and_test")
parser.add_argument('--n_hidden', default=3)

parser.add_argument('--force_overwrite', default=False, action='store_true',
                    help="For debug. Force to overwrite")

# Set the random seed for the whole graph for reproductible experiments
if __name__ == "__main__":
    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    json_path = os.path.join(args.experiment_dir, 'experiment_setting.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Safe Overwrite. Avoid to overwrite the previous experiment by mistake.
    force_overwrite = args.force_overwrite
    if force_overwrite is True:
        safe_files = ["experiment_setting.json"]
        if args.restore_from is not None:
            safe_files.append(os.path.split(args.restore_from)[-2])

        if args.mode == "test":
            # every file under the root of the "experiment_dir"
            for file_folder in os.listdir(args.experiment_dir):
                if file_folder != "test":
                    safe_files.append(file_folder)

        # delete everything that is not in "safe_files"
        for file_folder in os.listdir(args.experiment_dir):
            if file_folder not in safe_files:
                delete_file_or_folder(os.path.join(args.experiment_dir, file_folder))

    # Set the logger
    set_logger(os.path.join(args.experiment_dir, 'train.log'))
    logging.info(" ".join(sys.argv))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    train_feature = np.loadtxt(os.path.join(args.data_dir, 'train_feature.csv'), delimiter=",", dtype=np.float32)
    train_label = np.loadtxt(os.path.join(args.data_dir, 'train_label.csv'), delimiter=",", dtype=np.float32)

    logging.info("train feature shape: " + f"{train_feature.shape}")
    logging.info("train label shape: " + f"{train_label.shape}")
    logging.info("- done.")

    logging.info("Creating the model...")

    # create model
    input_dim = params.affine_coupling_layers["z_dim"] + params.affine_coupling_layers["c_dim"]
    
    output_dim = params.affine_coupling_layers["z_dim"]
    s_args = (input_dim, output_dim, 
              params.affine_coupling_layers["s_net"]["n_hidden"],
              params.affine_coupling_layers["s_net"]["hidden_dim"])

    t_args = (input_dim, output_dim,
              params.affine_coupling_layers["t_net"]["n_hidden"],
              params.affine_coupling_layers["t_net"]["hidden_dim"])

    s_kwargs = {"activation_type": params.affine_coupling_layers["s_net"]["activation_type"],
                "last_activation_type": params.affine_coupling_layers["s_net"]["last_activation_type"]}

    t_kwargs = {"activation_type": params.affine_coupling_layers["t_net"]["activation_type"],
                "last_activation_type": params.affine_coupling_layers["t_net"]["last_activation_type"]}

    model = RealNVP(params.affine_coupling_layers["z_dim"],
                    params.affine_coupling_layers["n_transformation"],
                    s_args,
                    t_args,
                    s_kwargs,
                    t_kwargs)
    if torch.cuda.is_available():
        model = model.cuda()
        logging.info("Enable cuda")
    else:
        logging.info("cuda is not available")
    logging.info("- done.")

    # create optimizer
    if params.affine_coupling_layers["optimizer"]["type"] == "Adam":
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad == True]
                                     , **params.affine_coupling_layers["optimizer"]["kwargs"])
    else:
        raise ValueError("optimizer not in searching domain.")

    ##########################################
    # Train the model
    ##########################################
    # For the training, if restore_from is not None, it will first load the model to be restored, and remove everything in the experiment_dir
    # except for the experiment_setting.json. You can interpret it as "one experiment_dir means one training", i.e if you
    # want to try different rounds of training, you have to create different subfolders.
    #
    # For the testing (testing is not implemented now), if restore_from is None, it by default do two test tasks, one for "best_weights" and the other for
    # "last weights". In each task there are params.rounts_test rounds of test, and the results are saved as a dictionary,
    # where the key is the name of the metrics, and the values are lists of results for different rounds. The results name
    # contains names like "best_weights" and "last_weights"
    #                  if restore_from is not None, it will load model from restore_from, and save the results in alias
    #                  that is the file name, like "after-100-epochs."
    #
    # Test results are saved at the "test" folder
    #
    # Note that train and test can be separate. If restore_from is not None only in the "test" mode, it will not remove
    # previous results, as there is no training. Instead it will add a result file, contains restore alias, in the folder
    # "test"
    #
    # restore_from: the directory for the model file.
    if args.mode == "train" or args.mode == "train_and_test":
        logging.info("Starting training for {} epoch(s)".format(params.epochs))
        training(model, optimizer, train_feature, train_label,
                 restore_from=args.restore_from, batch_size=params.batch_size, epochs=params.epochs,
                 experiment_dir=args.experiment_dir,
                 save_frequency=params.save_frequency,
                 verbose_frequency=params.verbose_frequency,
                 save_each_epoch=params.save_each_epoch
                 )
