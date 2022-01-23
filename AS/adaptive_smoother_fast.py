import pickle
import pandas as pd
import os
import numpy as np
import seaborn as sns
import sys
sns.set(style = 'white', font_scale = 1.5)

import matplotlib.pyplot as plt
path = "."
with open(os.path.join(path, "US101_Lane1to5_t1.5s30.pickle"), "rb") as f:
    data = pickle.load(f)


# data
rho = data['rhoMat']
q = data['qMat']
u = data['vMat']
dx = data['s'][1] - data['s'][0]
dt = data['t'][1] - data['t'][0]



# AS parameters
# reviewer suggested to use parameters in paper "Low-Rank Hankel Tensor Completion for Traffic
# Speed Estimation"
Use_Reviewer_Suggested_Para = True

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
    "c_free": 0.277778*70, # km/h to m/s
    "c_cong": 0.277778*(-15),
    "V_thr": 0.277778*60,
    "DV": 0.277778*20,
     "dx":dx,
     "dt":dt
    }
    
    
# window in the t direction: we don't need use all 1770 t.
half_window = 50
twice_window = 101


LOOPS = {2: [0,20],\
                #3: [0,7,20],\
    4: [0,5,11,20],\
                #5: [0,4,8,13,20],\
    6: [0,3,7,11,14,20],\
                #7: [0,3,6,9,12,15,20],\
    8: [0,2,5,8,11,13,16,20],\
                #9: [0,2,4,7,9,12,14,17,20],\
    10: [0,2,4,6,8,11,13,15,17,20],\
                #11: [0,2,4,6,8,10,12,14,16,18,20],\
    12: [0,1,3,5,7,9,11,12,14,16,18,20],\
                #13: [0,1,3,5,6,8,10,11,13,15,16,18,20],\
    14: [0,1,3,4,6,7,9,11,12,14,15,17,18,20],\
                #15: [0,1,2,4,5,7,8,10,11,13,14,16,17,19,20],\
                #16: [0,1,2,4,5,6,8,9,11,12,13,15,16,17,19,20],\
                #18: [0,1,2,3,4,6,7,8,9,11,12,13,14,15,17,18,19,20]
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
        dist_T_mid = dist_T_mid[idx_mid].reshape(21,1770-2*half_window,N_loop,-1) # dim: 21*1670*N_loop*101

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
        rho_loop = np.repeat(rho_loop[np.newaxis,:,:], 1770, axis = 0)
        rho_loop = np.repeat(rho_loop[np.newaxis,:,:,:], 21, axis = 0)
        rho_loop = self._filter_by_t(rho_loop, self.idx_mid)

        u_loop = self.u[self.X_loop,:][:,self.T]
        u_loop = np.repeat(u_loop[np.newaxis,:,:], 1770, axis = 0)
        u_loop = np.repeat(u_loop[np.newaxis,:,:,:], 21, axis = 0)
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
        arr_mid = arr_mid[idx_mid].reshape(21,1770-2*half_window,N_loop,-1)
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
    
    
Error_rho = []
Error_u = []
loops_to_use = [2,4,6,8,10,12,14]
for N_loop in [2,4,6,8,10,12,14]:
    adam = AdaSM(rho, u, LOOPS[N_loop], T)
    rho_star, u_star = adam.predict()
    error_rho = np.linalg.norm(rho-rho_star,2)/np.linalg.norm(rho,2)
    error_u = np.linalg.norm(u-u_star,2)/np.linalg.norm(u,2)
    Error_rho.append(error_rho)
    Error_u.append(error_u)
    print("loop = ", N_loop, ",    error_rho = %.3e"%error_rho, ",    error_u = %.3e"%error_u)
    
    
df = pd.DataFrame({"num_loop": loops_to_use,
                  "Error_rho": Error_rho,
                  "Error_u": Error_u
                  })
df.to_csv("AS_result.csv", index = None)