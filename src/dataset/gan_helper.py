import scipy.io
import numpy as np
import torch

class gan_helper():

    def __init__(self, Noise_scale, N_loop):

        self.N_loop = N_loop
        self.noise = Noise_scale
        self.shape = (96,24)
        self.load_arz()
        if self.N_loop>1:
            gap = int(240.0/(self.N_loop-1))
            loopidx = [i*gap for i in range(self.N_loop-1)]
            if self.N_loop ==4: loopidx[-1]=120
            loopidx.append(239)
        else:
            loopidx = [0]

        self.span = np.array(loopidx)//10

    def load_arz(self):

        data = scipy.io.loadmat('data/arz/ARZ_greenshieldSim_epsbell_infer_ring_May14_0.02tau.mat')

        t = data['t'].flatten()[:, None]  # 960 by 1
        x = data['x'].flatten()[:, None]  # 241 by 1
        Exact_rho = np.real(data['rho']).T  # 241 by 960, and after transpose, becomes 960*241
        Exact_u = np.real(data['u']).T
        X, T = np.meshgrid(x, t)  # each is 960 by 241

        shift = 1
        rho_low_d = np.zeros((self.shape[0], self.shape[1] + shift))
        u_low_d = np.zeros((self.shape[0], self.shape[1] + shift))
        T_low_d = np.zeros((self.shape[0], self.shape[1] + shift))
        X_low_d = np.zeros((self.shape[0], self.shape[1] + shift))
        ddx = Exact_rho.shape[0] // self.shape[0]
        ddy = Exact_rho.shape[1] // self.shape[1]

        for i in range(Exact_rho.shape[0]):
            for j in range(Exact_rho.shape[1]):
                if i % ddx == 0 and j % ddy == 0:
                    rho_low_d[i // ddx][j // ddy] = Exact_rho[i][j]
                    u_low_d[i // ddx][j // ddy] = Exact_u[i][j]
                    T_low_d[i // ddx][j // ddy] = T[i][j]
                    X_low_d[i // ddx][j // ddy] = X[i][j]

        self.rho_low_d = rho_low_d
        self.u_low_d = u_low_d
        self.rho_u_low_d = np.stack((rho_low_d,u_low_d),axis=0)
        self.X_T_low_d = np.hstack((X_low_d.flatten()[:,None], T_low_d.flatten()[:,None])).astype(np.float32)

    def load_ground_truth(self):
        real_figure = np.expand_dims((self.rho_u_low_d + self.noise * np.random.randn(self.rho_u_low_d.shape[0], self.rho_u_low_d.shape[1],self.rho_u_low_d.shape[2])),axis=0).astype(np.float32)
        return real_figure[:,:,:,self.span], real_figure
    def reshape_to_figure(self, rho, u):
        rho = rho.reshape(-1,self.shape[1]+1)
        u = u.reshape(-1,self.shape[1]+1)
        rho_u = torch.stack((rho, u), axis=0)
        return rho_u.unsqueeze(0)[:,:,:,self.span], rho_u.unsqueeze(0)

