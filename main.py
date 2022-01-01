import argparse
import logging
import os
import sys

import numpy as np


from src.utils import Params
from src.models.flow import RealNVP
from src.metrics import instantiate_losses, instantiate_metrics, functionalize_metrics
from src.utils import set_logger, delete_file_or_folder
from src.training import training, test_multiple_rounds



parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', default='experiments/arz',
                    help="Directory containing experiment_setting.json")
parser.add_argument('--data_dir', default='data/arz', help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, file location containing weights to reload")
parser.add_argument('--mode', default='train',
                    help="train, test, or train_and_test")

parser.add_argument('--force_overwrite', default=False, action='store_true',
                    help="For debug. Force to overwrite")

# Set the random seed for the whole graph for reproductible experiments
if __name__ == "__main__":
    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    json_path = os.path.join(args.experiment_dir, 'experiment_setting.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Load the parameters from the dataset, that gives the size etc. into params
    json_path = os.path.join(args.data_dir, 'data_para.json')
    assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
    params.update(json_path)

    # Check if force to overwrite
    force_overwrite = args.force_overwrite
    if force_overwrite is True:
        for file_folder in os.listdir(args.experiment_dir):
            if file_folder != 'experiment_setting.json':
                delete_file_or_folder(os.path.join(args.experiment_dir, file_folder))

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    model_dir_has_best_weights = os.path.isdir(os.path.join(args.experiment_dir, "best_weights"))
    overwritting = model_dir_has_best_weights and args.restore_from is None

    if args.mode != "test":
        assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"


    # Set the logger
    set_logger(os.path.join(args.experiment_dir, 'train.log'))
    logging.info(" ".join(sys.argv))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    train_data = np.loadtxt(os.path.join(args.data_dir, 'train_data.csv'), delimiter=",", dtype=np.float32)
    validation_data = np.loadtxt(os.path.join(args.data_dir, 'validation_data.csv'), delimiter=",", dtype=np.float32)
    collocation_data = np.loadtxt(os.path.join(args.data_dir, 'collocation_data.csv'), delimiter=",", dtype=np.float32)
    test_data = np.loadtxt(os.path.join(args.data_dir, 'test_data.csv'), delimiter=",", dtype=np.float32)

    logging.info("train data shape: " + f"{train_data.shape}")
    logging.info("validation data shape: " + f"{validation_data.shape}")
    logging.info("collocation data shape: " + f"{collocation_data.shape}")
    logging.info("test data shape: " + f"{test_data.shape}")
    logging.info("- done.")

    train_data = tf.convert_to_tensor(train_data)
    validation_data = tf.convert_to_tensor(validation_data)
    collocation_data = tf.convert_to_tensor(collocation_data)
    test_data = tf.convert_to_tensor(test_data)

    # train_data = (train_data[:,:3], train_data[:,3:])
    # validation_data = (validation_data[:, :3], validation_data[:, 3:])
    # collocation_data = (collocation_data[:, :3], collocation_data[:, 3:])
    # test_data = (test_data[:, :3], test_data[:, 3:])

    train_feature_dict = {"train": train_data[:params.n_train, :3],
                          "collocation": collocation_data[:params.n_collocation, :3]}
    train_target = train_data[:params.n_train, 3:]
    validation_data = (validation_data[:params.n_valid, :3], validation_data[:params.n_valid, 3:])
    test_feature = test_data[:params.n_test, :3]
    test_target = test_data[:params.n_test, 3:]

    X_mean, X_std = np.mean(train_data[:params.n_train, :3], axis=0, keepdims=True), np.std(
        train_data[:params.n_train, :3], axis=0, keepdims=True)
    Y_mean, Y_std = np.mean(train_target, axis=0, keepdims=True), np.std(train_target, axis=0, keepdims=True)

    logging.info("extracting data that is actually used...")
    logging.info(
        "train feature shape: " + f"{train_feature_dict['train'].numpy().shape}, " + "train target shape: " + f"{train_target.numpy().shape}")
    logging.info(
        "validation feature shape: " + f"{validation_data[0].numpy().shape}, " + "validation target shape: " + f"{validation_data[1].numpy().shape}")
    logging.info("collocation feature shape: " + f"{train_feature_dict['collocation'].numpy().shape}")
    logging.info(
        "test feature shape: " + f"{test_feature.numpy().shape}, " + "test target shape: " + f"{test_target.numpy().shape}")

    logging.info("- done.")

    logging.info("Creating the model...")
    layer_specs = dict()
    optimizers = dict()
    hyper_params = dict()

    # global hyper parameters, like lam, beta, gamma, which are the physics weights
    hyper_params["global_hyper_params"] = params.global_hyper_params

    loss_fns = {k: instantiate_losses(v) for k, v in params.loss_fns.items()}
    metric_fns = [instantiate_metrics(i) for i in params.metrics]
    metrics_fns_test = dict()
    for k in params.final_metrics:
        metrics_fns_test[k] = functionalize_metrics(k)
    # construct layer_specs
    for net_name, net_attri in params.layer_specs.items():
        # create model
        layer_specs[net_name] = dict()
        hyper_params[net_name] = dict()

        # create num_step, layer_type
        hyper_params[net_name]["num_step"] = net_attri["num_step"]

        # create layer_spec
        if net_name.split('_')[-1] == "generator":
            layers = [params.x_dim + params.z_dim] + \
                     [params.n_hidden_nodes_generator] * params.n_hidden_layer_generator + \
                     [params.y_dim]
            layer_specs[net_name]["net"] = Generator
            layer_specs[net_name]["net_config"] = (layers, X_mean, X_std, Y_mean, Y_std, params.z_dim)

        elif net_name.split('_')[-1] == "discriminator":
            layers = [params.x_dim + params.y_dim] + \
                     [params.n_hidden_nodes_discriminator] * params.n_hidden_layer_discriminator + \
                     [1]
            layer_specs[net_name]["net"] = Discriminator
            layer_specs[net_name]["net_config"] = (layers, X_mean, X_std, Y_mean, Y_std)

        elif net_name.split('_')[-1] == "estimator":
            layers = [params.x_dim + params.y_dim] + \
                     [params.n_hidden_nodes_estimator] * params.n_hidden_layer_estimator + \
                     [params.z_dim]
            layer_specs[net_name]["net"] = PosteriorEstimator
            layer_specs[net_name]["net_config"] = (layers, X_mean, X_std, Y_mean, Y_std)
        elif net_name.split('_')[-1] == "physics":
            physics_type = net_attri["physics_type"]
            if physics_type == "S-IDM":
                layer_specs[net_name]["net"] = StochasticIDM
                layer_specs[net_name]["net_config"] = (net_attri["meta_params_value"],
                                                             net_attri["meta_params_trainable"],
                                                             net_attri["lower_bounds"],
                                                             net_attri["upper_bonuds"])
            else:
                raise ValueError("invalid physics type")
        else:
            raise ValueError("network name not in searching domain:[Generator, Discriminator, PosteriorEstimator]")

        layer_specs[net_name]["kwargs"] = {"name": net_name,
                                           "layer_type": net_attri["layer_type"],
                                           "constraint": net_attri["constraint"]}

        # create optimizer
        if net_attri["optimizer"] == "Adam":
            optimizers[net_name] = keras.optimizers.Adam(learning_rate=net_attri["learning_rate"])
        elif net_attri["optimizer"] == "RMSprop":
            optimizers[net_name] = keras.optimizers.RMSprop(learning_rate=net_attri["learning_rate"])
        else:
            raise ValueError("optimizer not in searching domain.")

    if params.gan_type == "GAN":
        model = GAN(layer_specs, hyper_params)
        model.compile(optimizer=optimizers, loss=loss_fns, metrics=metric_fns)
    else:
        raise ValueError("model not in searching domain")
    logging.info("- done.")

    ##########################################
    # Train the model
    ##########################################
    # For the training, if restore_from is not None, it will first load the model to be restored, and remove everything in the experiment_dir
    # except for the experiment_setting.json. You can interpret it as "one experiment_dir means one training", i.e if you
    # want to try different rounds of training, you have to create different subfolders.
    #
    # For the testing, if restore_from is None, it by default do two test tasks, one for "best_weights" and the other for
    # "last weights". In each task there are params.rounts_test rounds of test, and the results are saved as a dictionary,
    # where the key is the name of the metrics, and the values are lists of results for different rounds. The results name
    # contains folder names like "best_weights"
    #                  if restore_from is not None, it will load model from restore_from, and save the results in alias
    #                  that is the file name, like "after-100-epochs."
    #
    # Test results are saved at the "test" folder
    #
    # Note that train and test can be separate. If restore_from is not None only in the "test" mode, it will not remove
    # previous results, as there is no training. Instead it will add a result file, contains restore alias, in the folder
    # "test"
    #
    # Here let's reclaim the different between restore_name, restore_from, restore alias:
    # restore_from: the directory for the model file.
    # restore_name: the model file name, omitting the extension.
    # restore_alias: this one is special. Every model has its file name, in the form of "after-epoch-xxx.h5". For best_weigths,
    #                and last_weights, we just use its folder name like "best_weights", to indicate the model we used.
    #                So if the restore_from is None, we use "best_weights" and "last_weights" as the aliases.
    #                If the restore_from is not None, as we don't have folder anymore, or the model can be any model that
    #                is hard to give an alias, we use its restore_name as the restore_alias. That is, its restore_name is
    #                the same as the restore_alias.
    if args.mode == "train" or args.mode == "train_and_test":
        logging.info("Starting training for {} epoch(s)".format(params.epoch))
        training(model, train_feature_dict, train_target, validation_data=validation_data,
                 restore_from=args.restore_from, batch_size=params.batch_size, epochs=params.epoch,
                 experiment_dir=args.experiment_dir,
                 keep_latest_model_max=params.keep_latest_model_max,
                 train_PUNN = (params.train_PUNN=="True"),
                 train_PINN = (params.train_PINN=="True"))
    if args.mode == "test" or args.mode == "train_and_test":
        logging.info("Starting test for {} round(s)".format(params.rounds_test))

        if args.restore_from is not None:
            restore_name = os.path.splitext(os.path.basename(args.restore_from))[0]
            test_multiple_rounds(model, (test_feature, test_target),
                                                 params.rounds_test,
                                                 save_dir=os.path.join(args.experiment_dir, "test"),
                                                 model_alias = restore_name,
                                                 metric_functions=metrics_fns_test,
                                                 restore_from=args.restore_from,
                                                 n_samples=params.n_samples,
                                                 params=params)
        else:
            #by default, predict using both last weights and best weights
            model_aliases = ["best_weights", "last_weights"]
            for restore_alias in model_aliases:
                # Restore alias use the folder name, like "best_weights",
                # while restore name use the file name of the .h5 file, like "after-epoch-688".
                # Restore_from is the location of the .5 file

                restore_from = os.path.join(args.experiment_dir, restore_alias) # folder name

                # get the file name, such as "after-epoch-100"
                file_candicates = os.listdir(restore_from)
                file_name = file_candicates[0] if "after-epoch" in file_candicates[0] else file_candicates[1] # as there is only one file that does not start as "after-epoch", which is "checkpoint"
                file_name = os.path.splitext(file_name)[0]

                restore_from = os.path.join(restore_from, file_name)
                # use 0 because only contains 1 file, by default; Or you can intepret as this function can only choose one file, by default.

                test_multiple_rounds(model, (test_feature, test_target),
                                     params.rounds_test,
                                     save_dir=os.path.join(args.experiment_dir, "test"),
                                     model_alias=restore_alias,
                                     metric_functions=metrics_fns_test,
                                     restore_from=restore_from,
                                     n_samples=params.n_samples,
                                     params=params
                                     )

