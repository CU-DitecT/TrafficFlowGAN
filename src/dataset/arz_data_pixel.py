# generatre arz data:
import scipy.io
import numpy as np

class arz_pixel_data_loader():

    def __init__(self, Loop_number, Noise_scale, Noise_number):

        self.noise = Noise_scale
        self.N_noise = Noise_number 
        self.N_u = 20000
        self.N_loop = Loop_number
    def load_bound(self):
        return self.mean.astype(np.float32), self.std.astype(np.float32)
    def load_test(self):
        return self.X_star, self.Exact_rho, self.Exact_u

    def load_data(self):

        data = scipy.io.loadmat('data/arz/ARZ_greenshieldSim_epsbell_infer_ring_May14_0.02tau.mat')


        t = data['t'].flatten()[:,None]# 960 by 1
        x = data['x'].flatten()[:,None]# 241 by 1
        Exact_rho = np.real(data['rho']).T #241 by 960, and after transpose, becomes 960*241
        Exact_u = np.real(data['u']).T


        X, T = np.meshgrid(x,t) # each is 960 by 241

        X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) # hstack is column wise stack, 241*960 (after flatten) by 2
        self.X_star = X_star.astype(np.float32)

        self.Exact_rho = Exact_rho
        self.Exact_u = Exact_u
        rho_star = Exact_rho.flatten()[:, None]  # 241*960 by 1
        u_star = Exact_u.flatten()[:, None]  # 241*960 by 1
        idx3 = np.random.choice(range(960 * 241), 150000, replace=False)  # collocation points
        X_star3 = X_star[idx3, :]

        # Doman bounds
        lb = X.min(0) # [0, 0]
        ub = T.max(0) # [1, 3]

        # print(lb)
        # print(ub)

        ######################################################################
        ######################## Noiseles Data ###############################
        ######################################################################

        if self.N_loop>1:
            gap = int(240.0/(self.N_loop-1))
            loopidx = [i*gap for i in range(self.N_loop-1)]
            if self.N_loop ==4: loopidx[-1]=120
            loopidx.append(239)
        else:
            loopidx = [0]
        span = np.array(loopidx)
        idx = []
        for i in range(960):
            seg = list(span + i * 241)
            idx += seg

        print(len(idx))
        X_rho_train = X_star[idx, :]
        rho_train = rho_star[idx, :]
        u_train = u_star[idx, :]


        X_rho_repeat = np.repeat(X_rho_train, self.N_noise, axis=0)

        rho_train_repeat = rho_train + self.noise * np.random.randn(rho_train.shape[0], rho_train.shape[1])
        for i in range(self.N_noise - 1):
            rho_train_repeat = np.hstack(
                (rho_train_repeat, rho_train + self.noise * np.random.randn(rho_train.shape[0], rho_train.shape[1])))
        rho_train_repeat = rho_train_repeat.reshape(-1, 1)

        u_train_repeat = u_train + self.noise * np.random.randn(u_train.shape[0], u_train.shape[1])
        for i in range(self.N_noise - 1):
            u_train_repeat = np.hstack(
                (u_train_repeat, u_train + self.noise * np.random.randn(u_train.shape[0], u_train.shape[1])))
        u_train_repeat = u_train_repeat.reshape(-1, 1)
        rho_u_repeat = np.concatenate((rho_train_repeat, u_train_repeat), axis=1)

        X_rho_repeat = X_rho_repeat.astype(np.float32)
        rho_u_repeat = rho_u_repeat.astype(np.float32)
        X_star3 = X_star3.astype(np.float32)
        X_rho_u = np.concatenate((rho_u_repeat, X_rho_repeat), axis=1)
        self.mean = np.mean(X_rho_u, axis=0)
        self.std = np.std(X_rho_u, axis=0)


        # overwrite x_rho_repeat, rho_u_repeat
        X_repeat = X[::10, np.array(span)]
        T_repeat = T[::10, np.array(span)]
        X_T_repeat = np.stack([X_repeat, T_repeat], axis=0)
        X_T_repeat = np.stack([X_T_repeat for _ in range(self.N_noise)], axis=0)

        rho_repeat = Exact_rho[::10, np.array(span)]
        u_repeat = Exact_u[::10, np.array(span)]
        rho_u_repeat = np.stack([rho_repeat, u_repeat], axis=0)
        rho_u_repeat = np.stack([rho_u_repeat for _ in range(self.N_noise)], axis=0)
        rho_u_repeat += np.random.randn(*rho_u_repeat.shape) * self.noise

        return X_T_repeat.astype(np.float32), rho_u_repeat.astype(np.float32), X_star3.astype(np.float32), x,t,idx
