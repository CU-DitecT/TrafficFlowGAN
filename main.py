import argparse
import logging
import os
import sys

import numpy as np
import torch

from src.utils import Params, save_checkpoint, load_checkpoint
from src.models.flow import RealNVP
from src.models.flow_learning_z import RealNVP_lz
# from src.metrics import instantiate_losses, instantiate_metrics, functionalize_metrics
from src.utils import set_logger, delete_file_or_folder
from src.training import training, test, test_multiple_rounds
from src.dataset.arz_data import arz_data_loader
from src.dataset.lwr_data import lwr_data_loader
from src.dataset.burgers_data import burgers_data_loader
from src.dataset.ngsim_data import ngsim_data_loader
from src.layers.discriminator import Discriminator

from src.layers.physics import GaussianLWR
from src.layers.physics import GaussianARZ
from src.layers.physics import GaussianBurgers
from src.metrics import instantiate_losses, instantiate_metrics, functionalize_metrics

torch.autograd.set_detect_anomaly(True)

#CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    logging.info("Enable cuda")
else:
    device = torch.device('cpu')
    logging.info("cuda is not available")

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', default='experiments/ngsim_learning_z', #burgers_learning_z
                    help="Directory containing experiment_setting.json")
parser.add_argument('--restore_from', default= None, #"experiments/lwr_learning_z/weights/last.pth.tar",
                    help="Optional, file location containing weights to reload")
parser.add_argument('--mode', default='train',
                    help="train, test, or train_and_test")
parser.add_argument('--n_hidden', default=3)
parser.add_argument('--noise', default=0.2)

parser.add_argument('--test_sample', default=3)  # 100

parser.add_argument('--test_rounds', default=1)  # 3
parser.add_argument('--nlpd_use_mean', default='True')
parser.add_argument('--nlpd_n_bands', default=1000)
parser.add_argument('--force_overwrite', default=False, action='store_true',
                    help="For debug. Force to overwrite")

# Set the random seed for the whole graph for reproductible experiments
if __name__ == "__main__":
    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    json_path = os.path.join(args.experiment_dir, 'experiment_setting.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # if test, us
    if args.mode == "test":
        device = torch.device('cpu')
        logging.info("In the test mode, use cpu")

    # Safe Overwrite. Avoid to overwrite the previous experiment by mistake.
    force_overwrite = args.force_overwrite
    if force_overwrite is True:
        safe_files = ["experiment_setting.json", "safe_folder"]
        if args.restore_from is not None:
            safe_files.append(os.path.split(args.restore_from)[-2])

        if args.mode == "test":
            # every file under the root of the "experiment_dir"
            for file_folder in os.listdir(args.experiment_dir):
                if file_folder != "test_result":
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

    # arz_data = arz_data_loader(params.data_arz['loop_number'],params.data_arz['noise_scale'],params.data_arz['noise_number'])
    # train_feature, train_label = arz_data.load_data()
    if params.data['type'] == 'lwr':
        data_loaded = lwr_data_loader(params.data['loop_number'], params.data['noise_scale'],
                                      params.data['noise_number'], params.data['noise_miu'], params.data['noise_sigma'])
        train_feature, train_label, train_feature_phy, X, T = data_loaded.load_data()
        test_feature, Exact_rho = data_loaded.load_test()
        test_label = Exact_rho.flatten()[:, None]
        gaussion_noise = np.random.normal(params.data['noise_miu'], params.data['noise_sigma'],
                                          test_label.shape[0]).reshape(-1, 1)
        test_label = np.concatenate([test_label, gaussion_noise], 1)

    elif params.data['type'] == 'arz':
        data_loaded = arz_data_loader(params.data['loop_number'], params.data['noise_scale'],
                                      params.data['noise_number'])
        train_feature, train_label, train_feature_phy, X, T = data_loaded.load_data()
        test_feature, Exact_rho, Exact_u = data_loaded.load_test()
        mean, std = data_loaded.load_bound()
        test_label_rho = Exact_rho.flatten()[:, None]
        test_label_u = Exact_u.flatten()[:, None]
        test_label = np.concatenate([test_label_rho, test_label_u], 1)


    elif params.data['type'] == 'ngsim':
        data_loaded = ngsim_data_loader(params.data['loop_number'], params.data['noise_scale'],
                                      params.data['noise_number'])
        train_feature, train_label, train_feature_phy, x, t,idx = data_loaded.load_data()
        test_feature, Exact_rho, Exact_u = data_loaded.load_test()
        test_label_rho = Exact_rho.flatten()[:, None]
        test_label_u = Exact_u.flatten()[:, None]
        mean, std = data_loaded.load_bound()
        test_label = np.concatenate([test_label_rho, test_label_u], 1)


    elif params.data['type'] == 'burgers':
        data_loaded = burgers_data_loader(params.data['noise_scale'],params.data['noise_number'], 
                                            params.data['noise_miu'], params.data['noise_sigma'])
        train_feature, train_label, train_feature_phy, X, T = data_loaded.load_data()
        test_feature, Exact_rho = data_loaded.load_test()
        test_label = Exact_rho.flatten()[:, None]
        gaussion_noise = np.random.normal(params.data['noise_miu'], params.data['noise_sigma'],
                                          test_label.shape[0]).reshape(-1, 1)
        test_label = np.concatenate([test_label, gaussion_noise], 1)
    logging.info("load data: " + f"{params.data['type']}")
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
                "last_activation_type": params.affine_coupling_layers["s_net"]["last_activation_type"],
                "device":device}

    t_kwargs = {"activation_type": params.affine_coupling_layers["t_net"]["activation_type"],
                "last_activation_type": params.affine_coupling_layers["t_net"]["last_activation_type"],
                "device":device}

    # get physics
    if (params.physics["type"] == "none") | (params.physics["hypers"]["alpha"] == 1):
        physics = None
    elif params.physics["type"] == "lwr":
        physics = GaussianLWR(params.physics["meta_params_value"],
                              params.physics["meta_params_trainable"],
                              params.physics["lower_bounds"],
                              params.physics["upper_bounds"],
                              params.physics["hypers"],
                              train=(params.physics["train"] == "True"))
        physics.to(device)
    elif params.physics["type"] == "arz":
        physics = GaussianARZ(params.physics["meta_params_value"],
                              params.physics["meta_params_trainable"],
                              params.physics["lower_bounds"],
                              params.physics["upper_bounds"],
                              params.physics["hypers"],
                              train=(params.physics["train"] == "True"),
                              device=device).to(device)
        physics.to(device)
    elif params.physics["type"] == "burgers":
        physics = GaussianBurgers(params.physics["meta_params_value"],
                              params.physics["meta_params_trainable"],
                              params.physics["lower_bounds"],
                              params.physics["upper_bounds"],
                              params.physics["hypers"],
                              train=(params.physics["train"] == "True"))
        physics.to(device)
    else:
        raise ValueError("physics type not in searching domain.")

    # metric_fns = [instantiate_metrics(i) for i in params.metrics]
    metric_fns = [instantiate_metrics(i) for i in params.metrics]
    metric_fns = dict(zip(params.metrics, metric_fns))
    if params.learning_z == "False":
        model = RealNVP(params.affine_coupling_layers["z_dim"],
                        params.affine_coupling_layers["n_transformation"],
                        params.affine_coupling_layers["train"],
                        device,
                        s_args,
                        t_args,
                        s_kwargs,
                        t_kwargs)
        model.to(device)
    if params.learning_z == "True":
        input_dim_z = params.affine_coupling_layers["c_dim"]
        output_dim_z = params.affine_coupling_layers["z_dim"]
        z_miu_args = (input_dim_z, output_dim_z,
                      params.affine_coupling_layers["z_miu_net"]["n_hidden"],
                      params.affine_coupling_layers["z_miu_net"]["hidden_dim"])

        z_sigma_args = (input_dim_z, output_dim_z,
                        params.affine_coupling_layers["z_sigma_net"]["n_hidden"],
                        params.affine_coupling_layers["z_sigma_net"]["hidden_dim"])
        z_miu_kwargs = {"activation_type": params.affine_coupling_layers["z_miu_net"]["activation_type"],
                        "last_activation_type": params.affine_coupling_layers["z_miu_net"]["last_activation_type"],
                        "device":device}

        z_sigma_kwargs = {"activation_type": params.affine_coupling_layers["z_sigma_net"]["activation_type"],
                          "last_activation_type": params.affine_coupling_layers["z_sigma_net"]["last_activation_type"],
                          "device": device}
        model = RealNVP_lz(params.affine_coupling_layers["z_dim"],
                           params.affine_coupling_layers["n_transformation"],
                           params.affine_coupling_layers["train"], mean,std,
                           device,
                           s_args,
                           t_args,
                           s_kwargs,
                           t_kwargs,
                           z_miu_args, z_sigma_args,
                           z_miu_kwargs, z_sigma_kwargs)
        model.to(device)
    #### discriminator
    if args.mode == "test":
        discriminator=None
    else:
        discriminator = Discriminator((96,25,2)).to(device)
    # create optimizer
    if params.affine_coupling_layers["optimizer"]["type"] == "Adam":
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad == True]
                                     , **params.affine_coupling_layers["optimizer"]["kwargs"])
    else:
        raise ValueError("optimizer not in searching domain.")

    if physics is not None:
        if params.physics["optimizer"]["type"] == "Adam":
            optimizer_physics = torch.optim.Adam(
                [p for p in physics.torch_meta_params.values() if p.requires_grad == True]
                , **params.physics["optimizer"]["kwargs"])
        elif params.physics["optimizer"]["type"] == "SGD":
            optimizer_physics = torch.optim.SGD(
                [p for p in physics.torch_meta_params.values() if p.requires_grad == True]
                , **params.physics["optimizer"]["kwargs"])
        elif params.physics["optimizer"]["type"] == "none":
            optimizer_physics = None
    else:
        optimizer_physics = None

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

    if (args.mode == "train") or (args.mode == "train_and_test"):
        logging.info("Starting training for {} epoch(s)".format(params.epochs))
        training(model, optimizer,discriminator, train_feature, train_label, train_feature_phy,device,
                 restore_from=args.restore_from, batch_size=params.batch_size, epochs=params.epochs,
                 physics=physics,
                 physics_optimizer=optimizer_physics,
                 experiment_dir=args.experiment_dir,
                 save_frequency=params.save_frequency,
                 verbose_frequency=params.verbose_frequency,
                 save_each_epoch=params.save_each_epoch,
                 verbose_computation_time = params.verbose_computation_time
                 )

    if args.mode == "train_and_test":
        logging.info("Starting training for {} epoch(s)".format(params.epochs))
        training(model, optimizer, discriminator, train_feature, train_label,device,
                 restore_from=args.restore_from, batch_size=params.batch_size, epochs=params.epochs,
                 physics=physics,
                 physics_optimizer=optimizer_physics,
                 experiment_dir=args.experiment_dir,
                 save_frequency=params.save_frequency,
                 verbose_frequency=params.verbose_frequency,
                 save_each_epoch=params.save_each_epoch,
                 verbose_computation_time=params.verbose_computation_time
                 )

        # run test
        # !While, the used GPU memory may not be released. So it is recommended to run mode=train and then mode=test!#
        device = torch.device('cpu')
        logging.info("Before testing, switch to cpu")
        physics.to(device)
        model.to(device)

        restore_from = os.path.join(args.experiment_dir, "weights/last.path.tar")
        save_dir = os.path.join(args.experiment_dir, "test_result/")
        model_alias = args.experiment_dir.split('/')[-1]
        test_multiple_rounds(model, test_feature, test_label, test_rounds=args.test_rounds, save_dir=save_dir,
                             model_alias=model_alias,
                             restore_from=restore_from, metric_functions=metric_fns, n_samples=args.test_sample,
                             noise=args.noise, args=args)
        print('train_and_test done')

    if args.mode == "test":
        restore_from = os.path.join(args.experiment_dir, "weights/last.path.tar")
        save_dir = os.path.join(args.experiment_dir, "test_result/")
        model_alias = args.experiment_dir.split('/')[-1]
        test_multiple_rounds(model, test_feature, test_label,
                             test_rounds=args.test_rounds,
                             save_dir=save_dir,
                             model_alias=model_alias,
                             restore_from=restore_from,
                             metric_functions=metric_fns,
                             n_samples=args.test_sample,
                             noise=args.noise,
                             args=args)
        print('test done')
