import numpy as np
from scipy.stats import multivariate_normal
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.metrics_factory.get_KL import get_KL
from src.metrics_factory.get_NLPD import get_NLPD, get_NLPD_old


def instantiate_losses(loss_name):
    loss_dict = {
        "mse":tf.keras.losses.MeanSquaredError(name="mean_squared_error"),
        "kl":tf.keras.losses.BinaryCrossentropy(from_logits=True, name="binary_crossentropy")
    }


    return loss_dict[loss_name]

def instantiate_metrics(metric_name):
    metric_dict = {
        "mse": tf.keras.metrics.MeanSquaredError(),
        "rmse": tf.keras.metrics.RootMeanSquaredError(),
        "mae": tf.keras.metrics.MeanAbsoluteError(),
    }

    return metric_dict[metric_name]

def functionalize_metrics(metric_name):
    metric_dict = {
        "mse": lambda y, y_pred: mean_squared_error(y,
                np.mean(y_pred, axis=1, keepdims=True).repeat(y.shape[1], axis=1)
                                                    ),
        "mae": lambda y, y_pred: mean_absolute_error(y,
                np.mean(y_pred, axis=1, keepdims=True).repeat(y.shape[1], axis=1)
                                                    ),
        "rmse": lambda y, y_pred: mean_squared_error(y,
                np.mean(y_pred, axis=1, keepdims=True).repeat(y.shape[1], axis=1),
                                                    squared=False),
        "kl": lambda y, y_pred: get_KL(y, y_pred),
        "nlpd": get_NLPD
    }
    return metric_dict[metric_name]


# test the code: run metrics.py in the cmd
if __name__ == "__main__":
    import time
    r = np.random.RandomState(1)
    y_true = np.arange(20.0)
    y_true = np.tile(y_true.reshape(-1,1),  (1,10))
    y_true += r.randn(*y_true.shape)

    y_pred = np.arange(20.0)
    y_pred = np.tile(y_pred.reshape(-1,1), (1,10))
    y_pred += r.randn(*y_pred.shape)

    print("y_true shape: ", y_true.shape)
    print("y_pred shape: ", y_pred.shape)

    print("calculating direct distance metrics")
    mse = mean_squared_error(y_true,
                np.mean(y_pred, axis=1, keepdims=True).repeat(y_pred.shape[1], axis=1)
                                                    )
    mae = mean_absolute_error(y_true,
                np.mean(y_pred, axis=1, keepdims=True).repeat(y_pred.shape[1], axis=1)
                                                    )
    print("mse: ", mse)
    print("mae: ", mae)
    print("--done")

    print("calculating distributional metrics")
    start_time = time.time()
    kl = np.mean(get_KL(y_true, y_pred))
    duration = time.time() - start_time
    print("kl: ", kl, "duration: ", duration)

    start_time = time.time()
    nlpd_old = get_NLPD_old(y_true, y_pred, use_mean=True)
    duration = time.time() - start_time
    print("nlpd using for loop: ", nlpd_old, "duration: ", duration)

    start_time = time.time()
    nlpd = get_NLPD(y_true, y_pred, use_mean=True)
    duration = time.time() - start_time
    print("nlpd using no loop: ", nlpd, "duration: ", duration)


    # use the functionalize_metrics and try again
    metrics = ['mse', 'mae', 'rmse', 'kl', 'nlpd']

    metrics_dict = dict()
    for func_name in metrics:
        if func_name == "kl":
            metrics_dict[func_name] = np.mean(functionalize_metrics(func_name)(y_true, y_pred))
        else:
            metrics_dict[func_name] = functionalize_metrics(func_name)(y_true, y_pred)

    print(metrics_dict)

