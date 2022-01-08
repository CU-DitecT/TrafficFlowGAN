import torch
import numpy as np
import os
import src.utils as utils

import logging
import os
from src.utils import save_dict_to_json, check_exist_and_create
from torch.utils.tensorboard import SummaryWriter


def training(model, optimizer, train_feature, train_target, restore_from=None,
             epochs=1000,
             batch_size=None,
             experiment_dir=None,
             save_frequency=1,
             verbose_frequency=1,
             save_each_epoch="False"):
    # Initialize tf.Saver instances to save weights during metrics_factory
    X_train = train_feature
    y_train = train_target
    begin_at_epoch = 0
    writer = SummaryWriter(os.path.join(experiment_dir, "summary"))
    weights_path = os.path.join(experiment_dir, "weights")
    check_exist_and_create(weights_path)

    if restore_from is not None:
        assert os.path.isfile(restore_from), "restore_from is not a file"
        # restore model
        begin_at_epoch = utils.load_checkpoint(restore_from, model, optimizer, begin_at_epoch)
        logging.info(f"Restoring parameters from {restore_from}, restored epoch is {begin_at_epoch:d}")

    best_loss = 10000
    best_last_train_loss = {"best":
                                {"loss": 100,
                                 "epoch": 0},
                            "last":
                                {"loss": 100,
                                 "epoch": 0},
                            }
    for epoch in range(begin_at_epoch, epochs):
        # shuffle the data
        np.random.seed(1)
        idx = np.random.choice(X_train.shape[0], X_train.shape[0], replace=False)
        X_train = X_train[idx, :]
        y_train = y_train[idx, :]


        #### hard code
        # y_train[:,0] = y_train[:,1]
        ####

        # train step
        num_steps = X_train.shape[0] // batch_size
        for step in range(num_steps):
            x_batch = X_train[step * batch_size:(step + 1) * batch_size, :]
            y_batch = y_train[step * batch_size:(step + 1) * batch_size, :]

            loss, activation = model.log_prob(y_batch, x_batch)
            loss = -loss.mean()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # evaluation
            activation_eval = model.eval(x_batch)
            a = activation_eval["x1_eval"].detach().cpu().numpy()
            idx = np.argsort(a)[-1]
            # 4947
            a = 1

        # logging
        if verbose_frequency > 0:
            if epoch % verbose_frequency == 0:
                logging.info(f"Epoch {epoch + 1}/{epochs}    loss={loss:.3f}")

        # saving at every "save_frequency" or at the last epoch
        if (epoch % save_frequency == 0) | (epoch == begin_at_epoch + epochs - 1):
            is_best = loss < best_loss
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': model.state_dict(),
                                   'optim_dict': optimizer.state_dict()},
                                  is_best=is_best,
                                  checkpoint=weights_path,
                                  save_each_epoch=save_each_epoch)

            # if best loss, update the "best_last_train_loss"
            if is_best:
                best_loss = loss
                best_last_train_loss["best"]["loss"] = best_loss
                best_last_train_loss["best"]["epoch"] = epoch+1

            # update and save the latest "best_last_train_loss"
            best_last_train_loss["last"]["loss"] = loss
            best_last_train_loss["last"]["epoch"] = epoch+1

            save_path = os.path.join(experiment_dir, "best_last_train_loss.json")
            save_dict_to_json(best_last_train_loss, save_path)

            # save loss to tensorboard
            writer.add_scalar("loss/train", loss, epoch+1)

            # save activation to tensorboard
            for k, v in activation.items():
                writer.add_histogram(f"activation_train/{k:s}", v, epoch+1)
            for k, v in activation_eval.items():
                writer.add_histogram(f"activation_eval/{k:s}", v, epoch+1)


def test(model, data_test,
                restore_from=None,
                metric_functions = None,
                n_samples = None,
                params=None,
         train_PUNN = True):
    # experiment_dir: where the model json file locates
    # Initialize tf.Saver instances to save weights during metrics_factory

    if restore_from is not None:
        if ("phys" in model.name) and (train_PUNN is False):
            meta_params = load_json(restore_from)
            layer_specs, hyper_params = model.layer_specs, model.hyper_params
            layer_specs["primal_physics"]["net_config"] = list(layer_specs["primal_physics"]["net_config"])
            layer_specs["primal_physics"]["net_config"][0] = meta_params
            layer_specs["primal_physics"]["net_config"] = tuple(layer_specs["primal_physics"]["net_config"])
            parent_class = model.__class__
            model =parent_class(layer_specs, hyper_params, name=model.name)
        else:
            model.load_weights(restore_from).expect_partial()
    else:
        raise FileExistsError("model not exist in "+ restore_from)

    # make prediction
    pre_train = not train_PUNN
    test_feature, test_target = data_test
    test_prediction = model.predict(test_feature, n_repeat=n_samples, pre_train=pre_train)

    # convert to numpy
    test_target = test_target.numpy()
    test_prediction = test_prediction.numpy()

    metrics_dict = dict()
    kl = None
    for k, func in metric_functions.items():
        if k == 'nlpd':
            use_mean = True if params.nlpd_use_mean == "True" else False
            metrics_dict[k] = [func(test_target, test_prediction,
                                   use_mean = use_mean,
                                   n_bands = params.nlpd_n_bands)]
        elif k == "kl":
            kl = func(test_target, test_prediction)
            metrics_dict[k] = [np.mean(kl)]
        else:
            metrics_dict[k] = [func(test_target, test_prediction)]


    return metrics_dict, test_prediction, kl





def test_multiple_rounds(model, data_test, test_rounds,
                         save_dir = None,
                         model_alias = None,
                        train_PUNN = True,
                **kwargs):
    metrics_dict, test_prediction, kl = test(model, data_test, train_PUNN=train_PUNN,
                                         **kwargs)
    logging.info("Restoring parameters from {}".format(kwargs["restore_from"]))
    if test_rounds > 1:
        for i in range(test_rounds-1):
            metrics_dict_new,_,_ = test(model, data_test, train_PUNN = train_PUNN,
                                    **kwargs)
            for k in metrics_dict.keys():
                metrics_dict[k] += metrics_dict_new[k]
    check_and_make_dir(os.path.join(save_dir, model_alias))
    save_path_metric = os.path.join(save_dir, model_alias,
                                    f"metrics_test.json")
    save_path_prediction = os.path.join(save_dir, model_alias,
                                        f"predictions_test.csv")
    save_path_feature = os.path.join(save_dir, model_alias,
                                        f"features_test.csv")
    save_path_target = os.path.join(save_dir, model_alias,
                                        f"targets_test.csv")
    save_path_kl = os.path.join(save_dir, model_alias,
                                    f"kl_test.csv")


    save_dict_to_json(metrics_dict, save_path_metric)
    np.savetxt(save_path_prediction, test_prediction, delimiter=",")
    np.savetxt(save_path_feature, data_test[0], delimiter=",")
    np.savetxt(save_path_target, data_test[1], delimiter=",")
    np.savetxt(save_path_kl, kl, delimiter=",")




