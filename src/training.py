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
        idx = np.random.choice(X_train.shape[0], X_train.shape[0], replace=False)
        X_train = X_train[idx, :]
        y_train = y_train[idx, :]

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
                writer.add_histogram(f"activation/{k:s}", v, epoch+1)










