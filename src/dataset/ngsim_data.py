# generatre arz data:
import scipy.io
import numpy as np
import pickle

class ngsim_data_loader():

    def __init__(self, Loop_number, Noise_scale, Noise_number):

        self.noise = Noise_scale
        self.N_noise = Noise_number 
        self.N_u = 20000
        self.N_loop = Loop_number

    def load_test(self):
        return self.X_star, self.Exact_rho, self.Exact_u
    def load_bound(self):
        return self.mean, self.std
    def load_data(self):
        with open('data/Ngsim/US101_Lane1to5_t1.5s30.pickle', 'rb') as f:
            data_pickle = pickle.load(f)

        Loop_dict = {3: [0, 7, 20],
                     4: [0, 5, 11, 20],
                     5: [0, 4, 8, 13, 20],
                     6: [0, 3, 7, 11, 14, 20],
                     7: [0, 3, 6, 9, 12, 15, 20],
                     8: [0, 2, 5, 8, 11, 13, 16, 20],
                     9: [0, 2, 4, 7, 9, 12, 14, 17, 20],
                     10: [0, 2, 4, 6, 8, 11, 13, 15, 17, 20],
                     11: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                     12: [0, 1, 3, 5, 7, 9, 11, 12, 14, 16, 18, 20],
                     13: [0, 1, 3, 5, 6, 8, 10, 11, 13, 15, 16, 18, 20],
                     14: [0, 1, 3, 4, 6, 7, 9, 11, 12, 14, 15, 17, 18, 20],
                     15: [0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20],
                     16: [0, 1, 2, 4, 5, 6, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20],
                     18: [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18, 19, 20]}

        xx = np.array(data_pickle['s']).flatten()[:, None]
        tt = np.array(data_pickle['t']).flatten()[:, None]
        rhoMat = np.array([np.array(ele) for ele in data_pickle['rhoMat']])*50
        uMat = np.array([np.array(ele) for ele in data_pickle['vMat']])
        # rhoMat_smooth = scipy.ndimage.uniform_filter(rhoMat, size=5, mode='nearest')
        # uMat_smooth = scipy.ndimage.uniform_filter(uMat, size=5, mode='nearest')

        rhoMat =rhoMat *50
        # update 0203 zm: cut the ngsim
        #rhoMat = rhoMat[:,:170]
        #uMat = uMat[:,:170]
        #tt = tt[:170,:]



        X, T = np.meshgrid(xx, tt)
        print(len(X), len(X[0]))  # 21 by 1770

        N_u = int(len(X) * len(X[0]) * 0.8)
        print('N_u:', N_u)

        x = X.flatten()[:, None]  # 21*1770 by 1
        t = T.flatten()[:, None]  # 21*1770 by 1
        # Exact_rho = rhoMat_smooth.T  # 1770 by 21
        # Exact_u = uMat_smooth.T
        Exact_rho = rhoMat.T  # 1770 by 21
        Exact_u = uMat.T
        self.Exact_rho = Exact_rho
        self.Exact_u = Exact_u
        N_loop = Loop_dict[self.N_loop]

        X_star = np.hstack((x, t)) # hstack is column wise stack, 241*960 (after flatten) by 2
        self.X_star = X_star.astype(np.float32)

        idx = np.random.choice(X_star.shape[0], N_u, replace=False)  # N_u = 22000 out of 37170 for Auxiliary points
        idx2 = []

        for i in range(Exact_rho.shape[0]):  # for observations on the loops
            base = i * Exact_rho.shape[1]
            index = [base + ele for ele in N_loop]
            idx2 += index


        rho_star = Exact_rho.flatten()[:,None] # 241*960 by 1
        u_star = Exact_u.flatten()[:,None] # 241*960 by 1



        # Doman bounds
        lb = X_star.min(0) # [0, 0]
        ub = X_star.max(0) # [1, 3] 


        ######################################################################
        ######################## Noiseles Data ###############################
        ######################################################################
        X_rho_train = X_star[idx2, :]  # [x, 0.0] and 100 points from left bound selected
        X_rho_repeat = np.repeat(X_rho_train, self.N_noise, axis=0)

        X_star3= X_star[idx, :]

        rho_train = rho_star[idx2, :]

        u_train = u_star[idx2, :]


        rho_train_repeat = rho_train + self.noise*np.random.randn(rho_train.shape[0], rho_train.shape[1])
        for i in range(self.N_noise-1):
            rho_train_repeat = np.hstack((rho_train_repeat, rho_train + self.noise*np.random.randn(rho_train.shape[0], rho_train.shape[1])))
        rho_train_repeat = rho_train_repeat.reshape(-1,1)


        u_train_repeat = u_train + self.noise*np.random.randn(u_train.shape[0], u_train.shape[1])
        for i in range(self.N_noise-1):
            u_train_repeat = np.hstack((u_train_repeat, u_train + self.noise*np.random.randn(u_train.shape[0], u_train.shape[1])))
        u_train_repeat = u_train_repeat.reshape(-1,1)
        rho_u_repeat = np.concatenate((rho_train_repeat,u_train_repeat),axis=1)

        X_rho_repeat = X_rho_repeat.astype(np.float32)
        rho_u_repeat = rho_u_repeat.astype(np.float32)
        X_star3 = X_star3.astype(np.float32)
        X_rho_u  = np.concatenate((rho_u_repeat, X_rho_repeat),axis=1)
        self.mean = np.mean(X_rho_u, axis=0)
        self.std = np.std(X_rho_u, axis=0)

        return X_rho_repeat, rho_u_repeat, X_star3, xx, tt, idx2
