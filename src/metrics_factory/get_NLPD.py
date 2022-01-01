
import numpy as np

def Parzen_NLPD(yTrue, yPred, bw):
    # new: use no for loop
    yTrue =yTrue.reshape(-1,1) # for vector-level calculation
    E = -0.5 * np.power( (yTrue - yPred) / bw, 2)
    max_exp = np.max(E, axis=-1, keepdims=True)
    max_exp_rep = np.tile(max_exp, (1, yPred.shape[1]))
    exp_ = np.exp(E-max_exp_rep)
    constant = 0.5 * np.log(2*np.pi) + np.log(yPred.shape[1] * bw)
    nlpd = -np.log(np.sum(exp_, axis=1, keepdims=True)) - max_exp + constant
    return np.mean(nlpd)



def min_Parzen_NLPD(yTrue, yPred, n_bands):
    windows = np.linspace(0.01, 5, n_bands)
    nlpd = []
    for bw in windows:
        nlpd.append(Parzen_NLPD(yTrue, yPred, bw))
    inx = np.argmin(np.asarray(nlpd))
    return nlpd[inx], windows[inx], nlpd



def get_NLPD(y, y_pred, n_bands=1000, use_mean=True):
    if use_mean is True:
        y = np.mean(y, axis=1)
    elif use_mean is False:
        y = y[:,0]
    else:
        raise ValueError("use mean invalid value")

    # calculate the window
    _, w, _ = min_Parzen_NLPD(y, y_pred, n_bands)
    assert w < 10 , "best NLPD window approaching limit, consider increase window"
    NLPD = Parzen_NLPD(y, y_pred, w)
    return NLPD




#######################################################################
################# below are old methods using for loop #################
#######################################################################

def Parzen_NLPD_old(yTrue, yPred, bw):
    n_instances = yTrue.shape[0]
    nlpd = np.zeros((n_instances))

    assert yPred[0].shape[0] == yPred.shape[1]
    for i in range(n_instances):
        n_samples = yPred[i].shape[0]
        yt = np.tile(yTrue[i], n_samples)

        E = -0.5 * np.power((yt - yPred[i].flatten()) / bw, 2)

        max_exp = np.max(E, axis=-1, keepdims=True)

        max_exp_rep = np.tile(max_exp, n_samples)
        exp_ = np.exp(E - max_exp_rep)

        constant = 0.5 * np.log(2 * np.pi) + np.log(n_samples * bw)
        nlpd[i] = -np.log(np.sum(exp_)) - max_exp + constant
    return np.mean(nlpd)

def min_Parzen_NLPD_old(yTrue, yPred, n_bands):
    windows = np.linspace(0.01, 5, n_bands)
    nlpd = []
    for bw in windows:
        nlpd.append(Parzen_NLPD_old(yTrue, yPred, bw))
    inx = np.argmin(np.asarray(nlpd))
    return nlpd[inx], windows[inx], nlpd

def get_NLPD_old(y, y_pred, n_bands=1000, use_mean = True):
    if use_mean is True:
        y = np.mean(y, axis=1)
    elif use_mean is False:
        y = y[:,0]
    else:
        raise ValueError("use mean invalid value")
    # calculate the window
    _, w, _ = min_Parzen_NLPD_old(y, y_pred, n_bands)
    assert w < 4.9 , "best NLPD window approaching limit, consider increase window"
    NLPD = Parzen_NLPD_old(y, y_pred, w)
    return NLPD

