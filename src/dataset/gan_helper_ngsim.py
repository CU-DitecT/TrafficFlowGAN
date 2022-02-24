import scipy.io
import numpy as np
import torch
import pickle

class gan_helper():

    def __init__(self, Noise_scale, N_loop):

        self.N_loop = N_loop
        self.noise = Noise_scale
        self.load_ngsim()
        self.Loop_dict = {3: [0, 7, 20],
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
        self.span = self.Loop_dict[self.N_loop]

    def load_ngsim(self):

        with open('data/Ngsim/US101_Lane1to5_t1.5s30.pickle', 'rb') as f:
            data_pickle = pickle.load(f)

        xx = np.array(data_pickle['s']).flatten()[:, None]
        tt = np.array(data_pickle['t']).flatten()[:, None]
        X, T = np.meshgrid(xx, tt)
        rhoMat = np.array([np.array(ele) for ele in data_pickle['rhoMat']])
        uMat = np.array([np.array(ele) for ele in data_pickle['vMat']])
        Exact_rho = rhoMat.T  # 1770 by 21
        Exact_u = uMat.T
        self.X_T_low_d = np.hstack((X[::20,:].flatten()[:, None], T[::20,:].flatten()[:, None])).astype(np.float32)

        self.rho_u_low_d = np.stack((Exact_rho[::20,:], Exact_u[::20,:]), axis=0)

    def load_ground_truth(self):
        real_figure = np.expand_dims((self.rho_u_low_d + self.noise * np.random.randn(self.rho_u_low_d.shape[0], self.rho_u_low_d.shape[1],self.rho_u_low_d.shape[2])),axis=0).astype(np.float32)
        return real_figure[:,:,:,self.span], real_figure
    def reshape_to_figure(self, rho, u):
        rho = rho.reshape(-1,21)
        u = u.reshape(-1,21)
        rho_u = torch.stack((rho, u), axis=0)
        return rho_u.unsqueeze(0)[:,:,:,self.span], rho_u.unsqueeze(0)

