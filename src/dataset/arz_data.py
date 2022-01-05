# generatre arz data:
import scipy.io
import numpy as np
# Noise level (in this noise free case is zero)

class arz_data_loader():

    def __init__(self, Loop_number, Noise_scale, Noise_number):

        self.noise = Noise_scale
        self.N_noise = Noise_number 
        self.N_u = 20000
        self.N_loop = Loop_number

    def load_data(self):
        data = scipy.io.loadmat('../../data/arz/ARZ_greenshieldSim_epsbell_infer_ring_May14_0.02tau.mat')


        t = data['t'].flatten()[:,None]# 960 by 1
        x = data['x'].flatten()[:,None]# 241 by 1
        Exact_rho = np.real(data['rho']).T #241 by 960, and after transpose, becomes 960*241
        Exact_u = np.real(data['u']).T


        X, T = np.meshgrid(x,t) # each is 960 by 241

        X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) # hstack is column wise stack, 241*960 (after flatten) by 2

        data2 = np.array([0.0]*960)
        idx2 = np.random.choice(range(960), 750, replace=False) # points on the boundary
        #idx2 = np.array(list(range(960)))# all boundary points are uses for regularization
        X_star2 = np.hstack((data2.flatten()[:,None], t.flatten()[:,None]))


        idx3 = np.random.choice(range(960*241), 150000, replace=False) # collocation points
        X_star3 = X_star[idx3,:]


        rho_star = Exact_rho.flatten()[:,None] # 241*960 by 1
        u_star = Exact_u.flatten()[:,None] # 241*960 by 1


        # Doman bounds
        lb = X_star.min(0) # [0, 0]
        ub = X_star.max(0) # [1, 3] 

        print(lb)
        print(ub)

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
            seg = list(span+i*241)
            idx+= seg


        #idx = list(range(241)) # only use the initial
        print(len(idx))
        X_rho_train = X_star[idx,:]
        rho_train = rho_star[idx,:]
        u_train = u_star[idx,:]

        X_rho_repeat = np.repeat(X_rho_train,self.N_noise,axis = 0)


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
        X_rho_u  = np.concatenate((rho_u_repeat, X_rho_repeat),axis=1)

        return X_rho_repeat, rho_u_repeat
