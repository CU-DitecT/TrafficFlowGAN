# generatre arz data:
import scipy.io
import numpy as np
from pyDOE import lhs

class burgers_data_loader():

    def __init__(self, Noise_scale, Noise_number, noise_miu, noise_sigma):

        self.noise = Noise_scale
        self.N_noise = Noise_number 
        self.N_u = 100
        self.N_i = 50
        self.N_f = 10000
        self.miu = noise_miu
        self.sigma = noise_sigma


    def load_test(self):
        return self.X_star, self.Exact

    def load_data(self):
        data = scipy.io.loadmat('data/burgers/burgers_shock.mat')

        t = data['t'].flatten()[:,None]
        x = data['x'].flatten()[:,None]
        Exact = np.real(data['usol']).T
        self.Exact = Exact
        X, T = np.meshgrid(x,t)


        X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
        u_star = Exact.flatten()[:,None] 
        self.X_star=X_star.astype(np.float32)
        # Doman bounds
        lb = X_star.min(0)
        ub = X_star.max(0)

        # initial conditions t = 0
        xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
        uu1 = Exact[0:1,:].T

        # boundary conditions x = lb
        xx2 = np.hstack((X[:,0:1], T[:,0:1]))
        uu2 = Exact[:,0:1]

        # boundary conditions, x = ub
        xx3 = np.hstack((X[:,-1:], T[:,-1:]))
        uu3 = Exact[:,-1:]

        X_u_train = np.vstack([xx2, xx3]) 
        u_train = np.vstack([uu2, uu3])

        X_f_train = lb + (ub-lb)*lhs(2, self.N_f)
        X_f_train = np.vstack([X_f_train, X_u_train, xx1])

        # selecting N_u boundary points for training
        idx = np.random.choice(X_u_train.shape[0], self.N_u, replace=False)
        X_u_train = X_u_train[idx, :]
        u_train = u_train[idx,:]

        # selecting N_i initial points for training
        idx = np.random.choice(xx1.shape[0], self.N_i, replace=False)
        X_i_train = xx1[idx, :]
        u_i_train = uu1[idx, :]

        # adding boundary and initial points
        X_u_train = np.vstack([X_u_train, X_i_train])
        u_train = np.vstack([u_train, u_i_train])

        X_u_repeat = np.repeat(X_u_train,self.N_noise ,axis = 0)

        u_train_repeat = u_train + self.noise *np.random.randn(u_train.shape[0], u_train.shape[1])
        for i in range(self.N_noise -1):
            u_train_repeat = np.hstack((u_train_repeat, u_train + self.noise *np.random.randn(u_train.shape[0], u_train.shape[1])))
        u_train_repeat = u_train_repeat.reshape(-1,1)
        gaussion_noise = np.random.normal(self.miu,self.sigma,u_train_repeat.shape[0]).reshape(-1,1)
        rho_noisie_repeat  = np.concatenate((u_train_repeat, gaussion_noise),axis=1)

        return X_u_repeat.astype(np.float32), rho_noisie_repeat.astype(np.float32), X_f_train.astype(np.float32), x,t,idx
