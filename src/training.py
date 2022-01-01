import torch
import numpy as np
import os
import utils

from .helper_funcs import save_dict_to_json, check_and_make_dir

import logging
import os
from src.utils import check_exist_and_delete, check_not_empty_and_delete, delete_file_or_folder, load_json


def training(model, optimizer,  train_feature_dict, train_target,  validation_data=None, restore_from=None,
             epochs = 1000,
                experiment_dir = None,
             batch_size = None,
             keep_latest_model_max = None,
             n_repeat=None,
             train_PUNN = True,
             train_PINN = True):
    # Initialize tf.Saver instances to save weights during metrics_factory
    X_train = train_feature_dict["train"]
    X_collocation = train_feature_dict["collocation"]
    y_train = train_target
    begin_at_epoch = 0

    if restore_from is not None:
        logging.info("Restoring parameters from {}".format(restore_from))
        assert os.path.isfile(restore_from), "restore_from is not a file"
        # restore model
        begin_at_epoch  = utils.load_checkpoint(restore_from, model, optimizer, begin_at_epoch)

    best_loss = 10000
    for epoch in range(begin_at_epoch, begin_at_epoch+epochs):
        # Run one epoch
        if epoch % 100 == 0:
            logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
            num_steps = X_train.shape[0] // batch_size
            for step in range(num_steps):
                x_batch = X_train[step * batch_size:(step + 1) * batch_size, :]
                y_batch = y_train[step * batch_size:(step + 1) * batch_size, :]

                # random sample collocation batch
                random_idx = np.random.choice(X_collocation.shape[0], batch_size, replace=False)
                x_batch_collocation = X_collocation.numpy()[random_idx, :]
                x_dict = {"train": x_batch,
                          "collocation": x_batch_collocation}






