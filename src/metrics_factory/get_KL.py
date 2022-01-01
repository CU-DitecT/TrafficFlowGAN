from scipy import stats
import numpy as np


def get_kde_curve(data):
    data = data[:,None]
    xmin, xmax = data.min(), data.max()
    X_marginal = np.linspace(xmin, xmax, 100)[:,None]
    positions_marginal = X_marginal.flatten()
    values_marginal = data.flatten()
    gkde = stats.gaussian_kde(values_marginal)
    KDE_marginal = gkde.evaluate(positions_marginal)
    return X_marginal, KDE_marginal

def get_kde(data):
    data = data[:,None]
    xmin, xmax = data.min(), data.max()
    X_marginal = np.linspace(xmin, xmax, 100)[:,None]
    positions_marginal = X_marginal.flatten()
    values_marginal = data.flatten()
    gkde = stats.gaussian_kde(values_marginal)
    return X_marginal, gkde
    
    
    
 
def get_KL(real_data, sim_data):
    # real_data: N*M 2-d array. Every row is the distribution of rho for ONE (x,t)
    # sim_data: N*M 2-d array. Every row is the predicted distribution of rho
    
    KL = []
    for i in range(real_data.shape[0]):
        X_real, KDE_real = get_kde(real_data[i,:])
        X_sim, KDE_sim = get_kde(sim_data[i,:])
        X_general = np.linspace( min(np.vstack([X_real, X_sim]).flatten()),
                       max(np.vstack([X_real, X_sim]).flatten()),
                       1000)
        p_real = KDE_real.pdf(X_general) / sum(KDE_real.pdf(X_general))
        p_sim = KDE_sim.pdf(X_general) / sum(KDE_sim.pdf(X_general))
        r_star = np.log(p_real + 1e-6) - np.log(p_sim + 1e-6)
        kl = sum(p_real * r_star)
        KL.append(kl)
    return KL


