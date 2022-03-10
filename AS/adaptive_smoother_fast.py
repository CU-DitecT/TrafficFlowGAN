import pickle
import pandas as pd
import os
import numpy as np
import seaborn as sns
import scipy.io
import sys
sns.set(style = 'white', font_scale = 1.5)

import matplotlib.pyplot as plt
path = os.path.join("..", "data", "arz")
data = scipy.io.loadmat(os.path.join(path, "ARZ_greenshieldSim_epsbell_infer_ring_May14_0.02tau.mat"))


# data
u = data['u'][:-1,:]
rho = data['rho'][:-1,:]

# normalize the data
rhoM = rho.max()
uM = u.max()
#rho = rho / rhoM
#u = u /uM
#q = rho*u


t = data['t']
s = data['x']
dt = data['t'][0][1] - data['t'][0][0]
dx = data['x'][0][1] - data['x'][0][0]


# LF solver

PARA = {
    "rho_jam": 1.13,  # whatever, just a default value
    "u_free": 1.02,
    "tau": 0.02,
    "is_h_original_with_beta": 0,
    "is_h_only_with_Ueq": 1,

    # grid
    'dt': dt,
    'dx': dx,
}

sum_h_swithces = PARA["is_h_original_with_beta"] + \
                 PARA["is_h_only_with_Ueq"]

# if sum_h_switthes == 0, by default use the h_only_with_Ueq
assert sum_h_swithces <= 1



# AS parameters
# reviewer suggested to use parameters in paper "Low-Rank Hankel Tensor Completion for Traffic
# Speed Estimation"
Use_Reviewer_Suggested_Para = False

if Use_Reviewer_Suggested_Para is True:
    param = {"sigma": 200 * 0.3048,  # ft to m
        "tau": 10, # s
        "c_free": 54.6*0.3048, # ft/s to m/s
        "c_cong": -10.0*0.3048, # ft/s to m/s
        "V_thr": 40.0*0.3048, # ft/s to m/s
        "DV": 9.1*0.3048, # ft/s to m/s
         "dx":dx,
         "dt":dt
        }
else:
    param = {"sigma": dx/2,
    "tau": dt/2,
    "c_free": 1, # km/h to m/s
    "c_cong": -1,
    "V_thr": 1,
    "DV": 0.2,
     "dx":dx,
     "dt":dt
    }
    

N = rho.shape[0]
T = rho.shape[1]
print('dt=', dt)
print('dx=', dx)
print('N=', N)
print('T=', T)
print('dx/dt', dx/dt)

# AS parameters
# reviewer suggested to use parameters in paper "Low-Rank Hankel Tensor Completion for Traffic
# Speed Estimation"


# window in the t direction: we don't need use all 1770 t.
half_window = 20
twice_window = 41

LOOPS = {
    2: [0, 239], \
    4: [0, 80, 120, 239], \
    6: [0, 48, 96, 144, 192, 239], \
    10: [0, 26, 52, 78, 104, 130, 156, 182, 208, 239], \
    14: [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180, 198, 216, 239]
}

# the time index of the observation, which is the whole index set
T = np.array([i for i in range(rho.shape[1])])



class AdaSM():
    def __init__(self, rho, u, X_loop, T_loop):
        self.rho = rho                                 # ground-truth rho, 21*1770
        self.u = u                                     # ground-truth u, 21*1770
        self.X = np.arange(rho.shape[0])               # np.arange(21)
        self.T = np.arange(rho.shape[1])               # np.arange(1770)
        self.X_loop = np.array(X_loop)                 # position of loops, like [0,20]
        self.T_loop = T                                # identical to T
        self.get_phi_and_mid_idx()
        
        
    def get_phi_and_mid_idx(self):
        XX, TT = np.meshgrid(self.X, self.T, indexing="ij") # dim: 21*1770
        XX_loop, TT_loop = np.meshgrid(self.X_loop, self.T, indexing="ij") # dim: N_loop*1770
        dist_T = np.expand_dims(TT,(2,3)) - np.expand_dims(TT_loop, (0,1)) # dim: 21*1770*N_loop*1770
        
        # get dist T
        dist_T_first = dist_T[:,:half_window,:,:twice_window] # dim: 21*50*N_loop*101
        dist_T_mid = dist_T[:,half_window:-half_window,:,:]   # dim: 21*1670*N_loop*1770
        idx_mid = np.where(abs(dist_T_mid)<=half_window)      
        dist_T_mid = dist_T_mid[idx_mid].reshape(240,960-2*half_window,N_loop,-1) # dim: 21*1670*N_loop*101

        dist_T_last = dist_T[:,-half_window:,:,-twice_window:] # dim: 21*50*N_loop*101

        dist_T = np.concatenate([dist_T_first, dist_T_mid, dist_T_last], axis = 1) # dim: 21*1770*N_loop*101
        del(dist_T_first, dist_T_mid, dist_T_last)
        
        
        # get dist_X
        dist_X = np.expand_dims(XX, (2,3)) - np.expand_dims(XX_loop, (0,1))
        dist_X = self._filter_by_t(dist_X, idx_mid)
        
        
        # get phi
        
        dist_X = dist_X*param['dx']
        dist_T_free = dist_T*param['dt'] - dist_X/param['c_free']
        dist_T_cong = dist_T*param['dt'] - dist_X/param['c_cong']

        phi_free = self._get_phi(dist_X, dist_T_free)
        phi_cong = self._get_phi(dist_X, dist_T_cong)
        del(dist_X, dist_T_free, dist_T_cong)
        
        self.idx_mid = idx_mid
        self.phi_free = phi_free
        self.phi_cong = phi_cong
        
    def predict(self):
        rho_loop = self.rho[self.X_loop,:][:,self.T_loop]
        rho_loop = np.repeat(rho_loop[np.newaxis,:,:], 960 , axis = 0)
        rho_loop = np.repeat(rho_loop[np.newaxis,:,:,:], 240, axis = 0)
        rho_loop = self._filter_by_t(rho_loop, self.idx_mid)

        u_loop = self.u[self.X_loop,:][:,self.T]
        u_loop = np.repeat(u_loop[np.newaxis,:,:], 960, axis = 0)
        u_loop = np.repeat(u_loop[np.newaxis,:,:,:], 240, axis = 0)
        u_loop = self._filter_by_t(u_loop, self.idx_mid)
        
        V_free = self._aggregate(u_loop, self.phi_free)
        V_cong = self._aggregate(u_loop, self.phi_cong)
        W = 0.5*( 1 + np.tanh( (param['V_thr'] - np.minimum(V_free, V_cong))/param['DV'] ) )
        
        rho_star = self._get_filtered_z(rho_loop, W, self.phi_free, self.phi_cong)
        u_star = self._get_filtered_z(u_loop, W, self.phi_free, self.phi_cong)
        
        return rho_star, u_star
        
    def _filter_by_t(self, arr, idx_mid):
        arr_first = arr[:,:half_window,:,:twice_window]
        arr_mid = arr[:,half_window:-half_window,:,:]
        arr_mid = arr_mid[idx_mid].reshape(240,960-2*half_window,N_loop,-1)
        arr_last = arr[:,-half_window:,:,-twice_window:]

        arr = np.concatenate([arr_first, arr_mid, arr_last], axis = 1)
        return arr
    
    def _get_phi(self,dist_x,dist_t):
        return np.exp( -1*(abs(dist_x)/param["sigma"]+abs(dist_t)/param["tau"])  )
    
    def _aggregate(self,z, phi):
        # z could be rho or u
        # phi could be phi_cong or phi_free
        numerator = np.sum(phi*z, axis=(2,3))
        denominator = np.sum(phi, axis=(2,3))
        return numerator/denominator
    
    def _get_filtered_z(self, z, W, phi_free, phi_cong):
        z_free = self._aggregate(z, phi_free)
        z_cong = self._aggregate(z, phi_cong)
        return W*z_cong + (1-W)*z_free
    
    
N_loop = 4
adam = AdaSM(rho, u, LOOPS[N_loop], T)
rho_star, u_star = adam.predict()
#error_rho = np.linalg.norm(rho-rho_star,2)/np.linalg.norm(rho,2)
#error_u = np.linalg.norm(u-u_star,2)/np.linalg.norm(u,2)
error_rho = np.sqrt( np.square((rho - rho_star)**2).mean() )
error_u = np.sqrt( np.square((u - u_star)**2).mean() )
print("loop = ", N_loop, ",    error_rho = %.3e"%error_rho, ",    error_u = %.3e"%error_u)