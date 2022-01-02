import torch
import numpy as np
import os
import src.utils as utils

from .helper_funcs import save_dict_to_json, check_and_make_dir

import logging
import os
from src.utils import check_exist_and_delete, check_not_empty_and_delete, delete_file_or_folder, load_json


def training(model, optimizer, train_feature, train_target, restore_from=None,
             epochs=1000,
             batch_size=None,
             experiment_dir=None):
    # Initialize tf.Saver instances to save weights during metrics_factory
    X_train = train_feature
    y_train = train_target
    begin_at_epoch = 0

    if restore_from is not None:
        logging.info("Restoring parameters from {}".format(restore_from))
        assert os.path.isfile(restore_from), "restore_from is not a file"
        # restore model
        begin_at_epoch = utils.load_checkpoint(restore_from, model, optimizer, begin_at_epoch)

    best_loss = 10000
    for epoch in range(begin_at_epoch, begin_at_epoch + epochs):
        # Run one epoch
        if epoch % 100 == 0:
            logging.info("Epoch {}/{}".format(epoch + 1, epochs))
            num_steps = X_train.shape[0] // batch_size
            for step in range(num_steps):
                x_batch = X_train[step * batch_size:(step + 1) * batch_size, :]
                y_batch = y_train[step * batch_size:(step + 1) * batch_size, :]

                loss = -model.log_prob(y_batch, x_batch).mean()
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            logging.info("loss=%.3f" % loss)
