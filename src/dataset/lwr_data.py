import scipy.io
import numpy as np


class lwr_data_loader():

    def __init__(self, Loop_number, Noise_scale, Noise_number,noise_miu, noise_sigma):

        self.noise = Noise_scale
        self.N_noise = Noise_number 
        self.N_u = 100000
        self.N_loop = Loop_number
        self.miu = noise_miu
        self.sigma = noise_sigma # for generating gaussion noise
    def load_test(self):
        return self.X_star, self.Exact_rho
    def load_data(self):
        data = scipy.io.loadmat('data/lwr/rho_bellshape_10grid_DS10_gn_eps005_solver2_ring.mat')

        t = data['t'].flatten()[:,None]# 960 by 1
        x = data['x'].flatten()[:,None]# 240 by 1
        Exact_rho = np.real(data['rho']).T


        X, T = np.meshgrid(x,t) #

        X_repeat = X.flatten()[:,None].repeat(self.N_noise, axis=1)
        T_repeat = T.flatten()[:,None].repeat(self.N_noise, axis=1)
        # all points inside
        X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) # hstack is column wise stack, 960*240 by 2
        self.X_star=X_star.astype(np.float32)
        # index_x = np.where((X.flatten()[:,None] >= 0.4) & (X.flatten()[:,None] <=0.6))
        # index_t = np.where((T.flatten()[:,None] >= 0) & (T.flatten()[:,None] <=0.5))
        # print(index_x[0].shape,index_t[0].shape)
        # index_shock_wave = np.intersect1d(index_x[0], index_t[0])
        # print(index_shock_wave)
        # data2 = np.array([0.0]*650)
        idx2 = np.random.choice(range(960), 650, replace=False)
        data2 = np.array([0.0]*960)
        idx2 = np.array(list(range(960))) #np.random.choice(range(960), 650, replace=False)
        # t2 = t[idx].T#np.array(idx2).T

        # low boundary
        X_star2 = np.hstack((data2.flatten()[:,None], t.flatten()[:,None]))
        X_star2_repeat = np.repeat(X_star2, self.N_noise,axis = 0)

        idx3 = np.random.choice(range(960*241),  self.N_u, replace=False)
        X_star3 = X_star[idx3,:]
        X_star3_repeat = np.repeat(X_star3,self.N_noise,axis = 0)


        rho_star = Exact_rho.flatten()[:,None] # 960*240 by 1
        self.Exact_rho = Exact_rho


        # Doman bounds
        lb = X_star.min(0) # [0, 0]
        ub = X_star.max(0) # [1, 3] 

        gap = int(240.0/(self.N_loop-1))
        loopidx = [i*gap for i in range(self.N_loop-1)]
        if self.N_loop == 4: loopidx[-1] = 120
        loopidx.append(239)
        span = np.array(loopidx)

        idx = []
        for i in range(960):
            seg = list(span+i*241)
            idx+= seg

        X_rho_train = X_star[idx,:]
        # add shcock wave
        # X_rho_shockwave = X_star[index_shock_wave,:]
        # X_rho_train = np.vstack((X_rho_train,X_rho_shockwave))
        ##
        X_rho_repeat = np.repeat(X_rho_train,self.N_noise ,axis = 0)
        rho_train = rho_star[idx,:]
        # add shcok wave
        # rho_train_shockwave = rho_star[index_shock_wave,:]
        # rho_train = np.vstack((rho_train,rho_train_shockwave))

        # rho_train = rho_train + noise*np.random.randn(rho_train.shape[0], rho_train.shape[1])
        rho_train_repeat = rho_train + self.noise *np.random.randn(rho_train.shape[0], rho_train.shape[1])
        for i in range(self.N_noise -1):
            rho_train_repeat = np.hstack((rho_train_repeat, rho_train + self.noise *np.random.randn(rho_train.shape[0], rho_train.shape[1])))
        rho_train_repeat = rho_train_repeat.reshape(-1,1)
        gaussion_noise = np.random.normal(self.miu,self.sigma,rho_train_repeat.shape[0]).reshape(-1,1)
        rho_noisie_repeat  = np.concatenate((rho_train_repeat, gaussion_noise),axis=1)

        return X_rho_repeat.astype(np.float32), rho_noisie_repeat.astype(np.float32),X,T



