import numpy as np
from numpy import eye, dot
import math
import matplotlib.pyplot as plt
import os
import sys
from filterpy.kalman import ExtendedKalmanFilter as EKF
from filterpy.stats import logpdf
from filterpy.common import pretty_str, reshape_z
import os, sys
from filterpy.kalman import UnscentedKalmanFilter

import seaborn as sns
import math
sns.set(style = 'white', font_scale = 1.5)
import pickle

from arz_model import QHU_3Para, QHU_Green_Shield, LF
class QHU_3Para_augDerivative(QHU_3Para):
    def dQ(self,rho):
        rho_max = self.para['rho_jam']
        alpha = self.para['alpha']

        a = (self.g(1) - self.g(0)) / rho_max
        b = self.dg(rho / rho_max) / rho_max

        return alpha * (a - b)
    def dg(self,rho):
        lamda = self.para['lambda']
        p = self.para['p']
        numer = lamda ** 2 * (rho - p)
        denum = self.g(rho)
        return numer / denum
    def dU(self,rho):
        return (self.dQ(rho)*rho - self.Q(rho)) / rho**2
    def dh(self,rho):
        para = self.para
        if para["is_h_original_with_beta"] == 1:
            beta = para["beta"]; gamma = para["gamma"]; rho_jam = para["rho_jam"]
            x = rho / rho_jam
            t_1 = 1/2/np.sqrt(x/(1-x)); t_2=1/(1-x)**2; t_3 = 1/rho_jam
            return beta*t_1*t_2*t_3
        else:
            # by default use the simplest h function
            return  -1 * self.dU(rho)
        
class QHU_Green_Shield_augDerivative(QHU_Green_Shield):
    def dU(self, rho):
        para = self.para
        u_free = para["u_free"]; rho_jam = para["rho_jam"]
        return -1* u_free/rho_jam
    def dQ(self, rho):
        return 1*self.U(rho) + rho*self.dU(rho)

    def dh(self,rho):
        para = self.para
        if para["is_h_original_with_beta"] == 1:
            beta = para["beta"]; gamma = para["gamma"]; rho_jam = para["rho_jam"]
            x = rho / rho_jam
            t_1 = 1/2/np.sqrt(x/(1-x)); t_2=1/(1-x)**2; t_3 = 1/rho_jam
            return beta*t_1*t_2*t_3
        else:
            # by default use the simplest h function
            return  -1 * self.dU(rho)
        

class Partial_ARZ():
    def __init__(self,  QHU, PARA):
        '''
        for the df_rho, because it is widely used and structurally simple, we save it as a value
        for others, like the du, we write the function to obtain it
        among these 'others', if it is intermeidate value, like dy_temp, we write a function and then save it as a value
        
        'df_rho_drho_L' the first rho means the rho in the next state, to seperate with the drho, which mean current state
        '''
        
        self.PARA = PARA
        self.QHU = QHU
        self.C = PARA['dt']/PARA['dx']
        self.Lambda = PARA['dt']/PARA['tau']
        #self.rho_L = rho_L; self.rho_R = rho_R
        #self.u_L = u_L; self.u_R = u_R
    def calculate_partial(self,rho_L, rho_R, y_L, y_R):
        # f_rho
        C = self.C
        Lambda = self.Lambda
        self.rho = self.get_f_rho(rho_L, rho_R, y_L, y_R)
        self.y = self.get_f_y(rho_L, rho_R, y_L, y_R)
        
        # df_rho
        self.df_rho_drho_L = 1/2 + C/2 * ( self.QHU.U(rho_L) + rho_L*self.QHU.dU(rho_L) )
        self.df_rho_drho_R = 1/2 - C/2 * ( self.QHU.U(rho_R) + rho_R*self.QHU.dU(rho_R) )
        self.df_rho_dy_L =  1 * C/2 * np.ones(y_L.shape)
        self.df_rho_dy_R = -1 * C/2 * np.ones(y_R.shape)
        
        # y
        self.df_y_drho_L =  1 * C/2 * (-1*y_L**2/rho_L**2 + y_L*self.QHU.dU(rho_L))
        self.df_y_drho_R = -1 * C/2 * (-1*y_R**2/rho_R**2 + y_R*self.QHU.dU(rho_R))
        self.df_y_dy_L = (1/2 - Lambda/2)*y_L + C/2 * ( 2*y_L/rho_L + self.QHU.U(rho_L) )
        self.df_y_dy_R = (1/2 - Lambda/2)*y_R - C/2 * ( 2*y_R/rho_R + self.QHU.U(rho_R) )
        
    def get_f_rho(self, rho_L, rho_R, y_L, y_R):
        C = self.C
        term_1 = (rho_L + rho_R)/2
        term_2 = C/2 * ( 
            (y_R + rho_R*self.QHU.U(rho_R)) -
            (y_L + rho_L*self.QHU.U(rho_L))
        )
        return term_1 - term_2
    
    def get_f_y(self, rho_L, rho_R, y_L, y_R):
        C = self.C
        Lambda = self.Lambda
        term_1 = (y_L + y_R)/2
        term_2 = C/2* (
            (y_R**2/rho_R + y_R*self.QHU.U(rho_R)) - 
            (y_L**2/rho_L + y_L*self.QHU.U(rho_L))
        )
        term_3 = Lambda/2*(y_L+y_R)
        return term_1 - term_2 - term_3
    
    
    
# the matrix and its auxiliary


# ##################
# f_at isthe p_arz.calculate_partial().rho
def f_at(x, **kwargs):
    p_arz = kwargs['p_arz']
    N = int(round(len(x)/2))
    
    #rho_current = x[:N].reshape(-1,1)
    #u_current = x[N:].reshape(-1,1)
    
    #rho_L = np.vstack([rho_current[-1], rho_current[:-1]])
    #rho_R = np.vstack([rho_current[1:], rho_current[0]])
    #u_L = np.vstack([u_current[-1], u_current[:-1]])
    #u_R = np.vstack([u_current[1:], u_current[0]])
    
    rho_L = np.hstack([x[0], x[:N-1]])
    rho_R = np.hstack([x[1:N], x[N-1]])
    y_L = np.hstack([x[N], x[N:2*N-1]])
    y_R = np.hstack([x[N+1:], x[-1]])
    p_arz.calculate_partial(rho_L, rho_R, y_L, y_R)
    
    rho = p_arz.rho.flatten()
    y = p_arz.y.flatten()
    return np.hstack([rho,y])
    
    
# #################

# h_at
def h_at(x, loops):
    N = int(round(len(x)/2))
    result_rho = [x[i] for i in loops]
    result_y = [x[N+i] for i in loops]
    return np.array(result_rho + result_y)

def Hjacobian(x, N, loops):
    # N is actually the length of x
    H_jacobian = np.zeros( (2*len(loops),N) )
    for i in range(len(loops)):
        H_jacobian[i, loops[i]] = 1
        H_jacobian[i+len(loops), loops[i]+int(round(N/2))] = 1
    return H_jacobian


def Fjacobian(x, **f_kwargs):
    # rho: 0, 1, 2,..., N-1
    # u  : N, N+1, ..., 2N-1
    p_arz = f_kwargs['p_arz']
    F = np.zeros((len(x), len(x)))
    N = int(round(len(x)/2))
    rho_L = np.hstack([x[0], x[:N-1]])
    rho_R = np.hstack([x[1:N], x[N-1]])
    y_L = np.hstack([x[N], x[N:2*N-1]])
    y_R = np.hstack([x[N+1:], x[-1]])
    p_arz.calculate_partial(rho_L, rho_R, y_L, y_R)
    
    F[0, 0] = p_arz.df_rho_drho_L[0]# rho[0] partial rho L, 0 itself
    F[0, 1] = p_arz.df_rho_drho_R[0]# rho[0] partial rho R
    F[0, 0+N] = p_arz.df_rho_dy_L[0]# rho[0] partial u L
    F[0, 1+N] = p_arz.df_rho_dy_R[0]# rho[0] partial u  
    
    F[N-1, -1+N] = p_arz.df_rho_drho_R[-1]# rho[-1] partial rho R, -1 itself
    F[N-1, -2+N] = p_arz.df_rho_drho_L[-1]# rho[-1] partial rho L
    F[N-1, -1+N+N] = p_arz.df_rho_dy_R[-1]# rho[-1] partial u R, -1 itself
    F[N-1, -2+N+N] = p_arz.df_rho_dy_L[-1]# rho[-1] partial u L
    
    F[N, 0] = p_arz.df_y_drho_L[0]# u[0] partial rho L, 
    F[N, 1] = p_arz.df_y_drho_R[0]# u[0] partial rho R
    F[N, 0+N] = p_arz.df_y_dy_L[0]# u[0] partial u L, 
    F[N, 1+N] = p_arz.df_y_dy_R[0]# u[0] partial u R, 
    
    F[-1+N+N, -1+N] = p_arz.df_y_drho_R[-1]# u[-1] partial rho R 
    F[-1+N+N, -2+N] = p_arz.df_y_drho_L[-1]# u[-1] partial rho L
    F[-1+N+N, -1+N+N] = p_arz.df_y_dy_R[-1]# u[-1] partial u R
    F[-1+N+N, -2+N+N] = p_arz.df_y_dy_L[-1]# u[-1] partial u L

    for i in range(1, int(round(len(x)/2)) - 1):
        F[i, i-1] = p_arz.df_rho_drho_L[i]#rho partial rho L
        F[i, i+1] = p_arz.df_rho_drho_R[i]#rho partial rho R
        F[i, i-1+N] = p_arz.df_rho_dy_L[i]#rho partial u L
        F[i, i+1+N] = p_arz.df_rho_dy_R[i]#rho partial u R
        
        F[i+N, i-1] = p_arz.df_y_drho_L[i]#u partial rho L
        F[i+N, i+1] = p_arz.df_y_drho_R[i]#u partial rho R
        F[i+N, i-1+N] = p_arz.df_y_dy_L[i]#u partial u L
        F[i+N, i+1+N] = p_arz.df_y_dy_R[i]#u partial u R
        
        #F[i, i - 1] = partial_l(x[i - 1], **f_kwargs)
        #F[i, i + 1] = partial_r(x[i + 1], **f_kwargs)
    # arbitrary limit
    #for i in range(F.shape[0]):
    #    for j in range(F.shape[1]):
    #        F[i,j] = max(-10, F[i,j])
    #        F[i,j] = min(10, F[i,j])
    
    return F

class ARZEKF(EKF):
    def __init__(self, dim_x, dim_z_1, dim_z_2, std_pred, std_update):
        self.std_pred = std_pred
        self.std_update = std_update
        EKF.__init__(self, dim_x, dim_z_1+dim_z_2)
        
        # Q matrix
        half_N = int(round(dim_x/2))
        A = np.eye(half_N) * std_pred[0]
        B = np.eye(half_N) * std_pred[1]
        self.Q =  np.block([
                                [A,               np.zeros((half_N, half_N))],
                                [np.zeros((half_N, half_N)), B               ]
                            ])
        
     
        
        # R matrix
        A = np.eye(dim_z_1) * std_update[0]
        B = np.eye(dim_z_2) * std_update[1]
        self.R =  np.block([
                                [A,               np.zeros((dim_z_1, dim_z_2))],
                                [np.zeros((dim_z_2, dim_z_1)), B               ]
                            ])
        

    def predict_x(self, f_at, **f_kwargs):
        self.x = f_at(self.x, **f_kwargs)
        
        # min is 0.01x 
        #self.x = [max(0.01, self.x[i]) for i in range(len(self.x))]
        Rho = self.x[:21]
        Y = self.x[21:]
        
        # set the bounds on the variable
        Rho = [max(0.01, i) for i in Rho]
        Rho = [min(0.6, i) for i in Rho]
        Y = [max(-2, i) for i in Y]
        Y = [min(2, i) for i in Y]
        self.x = np.array(np.hstack([Rho,Y]))

    def predict(self, FJacobian, f_at, **f_kwargs):
        self.F = FJacobian(self.x, **f_kwargs)
        self.predict_x(f_at, **f_kwargs)

        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

def implement_kf(loops, dim_z_1, dim_z_2, rho, u,  config, **f_kwargs):
    std_Q = config['std_Q']
    std_R = config['std_R']
    init_rho = config['init_rho']
    init_u = config['init_u']
    init_P = config['init_P']
    
    para = f_kwargs['para']
    qhu_grn_sld = QHU_Green_Shield_augDerivative(para)
    p_arz = Partial_ARZ( qhu_grn_sld, para)

    lwrefk = ARZEKF(2*len(rho[:,0]),dim_z_1, dim_z_2,std_Q, std_R)
    lwrefk.x = np.hstack([init_rho, init_u])
    N = len(lwrefk.x)
    lwrefk.P = init_P
    X_pri = [lwrefk.x]
    X_pos = [lwrefk.x]
    Obs = []
    K = []
    P = []
    for i in range(1, rho.shape[1]):
        lwrefk.predict(Fjacobian, f_at, p_arz = p_arz)
        X_pri.append(lwrefk.x_prior)

        if i % 1 == 0:
            observe = [rho[k, i ] for k in loops] + [u[k, i ] for k in loops]
            Obs.append(observe)
            lwrefk.update(observe, Hjacobian, h_at, args=(N, loops),hx_args=(loops))
            #lwrefk.x = neibour_avg(lwrefk.x)
            lwrefk.x_post = lwrefk.x.copy()

        else:
            # X_pos.append(lwrefk.x_prior)
            pass
        K.append(lwrefk.K)
        P.append(lwrefk.P)
        X_pos.append(lwrefk.x_post)
    X_pos = np.vstack(X_pos).T
    X_pri = np.vstack(X_pri).T

    return X_pri, X_pos, K, P



path = "C:\\Users\\Zhaobin\\Desktop\\Columbia\\NGSIM\\Git\\NGSIM"
with open(os.path.join(path, "US101_Lane1to5_t1.5s30.pickle"), "rb") as f:
    data = pickle.load(f)
    
#with open( "US101_Lane1to5_t1.5s30.pickle", "rb") as f:
#    data = pickle.load(f)
    
# The real data provide the boundary and initial condition
u = data['vMat']
rho = data['rhoMat']
q = data['qMat']

# normalize the data
rhoM = rho.max()
uM = u.max()
#rho = rho / rhoM
#u = u /uM
#q = rho*u


t = data['t']
s = data['s']
dt = data['t'][1] - data['t'][0]
dx = data['s'][1] - data['s'][0]

# LF solver

PARA = {
    # general & green-shield flux
    "rho_jam": 1.13, # whatever, just a default value
    "u_free": 1.02,
    "tau":5,
    
    # three_para flux NOT USED in THE AZR EKF
    #"alpha": 0.1, # c in set_lwr_relation
    #"lambda":0.1, # stick to seibold's defition, not Kuang (1/lambda)
    #"p": 0.1, # b
    
    # h 
    #"beta": 0.1,
    #"gamma": 0.5,
    "is_h_original_with_beta": 0,
    "is_h_only_with_Ueq": 1,
    
    # grid
    'dt':dt,
    'dx':dx,
}

sum_h_swithces = PARA["is_h_original_with_beta"] + \
                    PARA["is_h_only_with_Ueq"]

# if sum_h_switthes == 0, by default use the h_only_with_Ueq
assert sum_h_swithces <= 1
    
    
    
# print the env variable
N = rho.shape[0]
T = rho.shape[1]
print('dt=', dt)
print('dx=', dx)
print('N=', N)
print('T=', T)
print('dx/dt', dx/dt)

LOOPS = {
    2: [0,239],\
    4: [0,80, 120, 239],\
    6: [0, 48, 96, 144, 192, 239],\
    10:[0, 26, 52, 78, 104, 130, 156, 182, 208, 239],\
    14:[0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180, 198, 216, 239]
        }

para_dict = {4: 
             {"loops": [0,80,120,239],
             "rho_max": 1.13,
             "u_max":1.02,
             "tau": 0.02}
            }

np.random.seed(43)

ReadRealPara = False
key = 4
loops = LOOPS[key]
print(loops)
# para
if ReadRealPara:
    #para = para_t30_s30
    pass
else:

    PARA['rho_jam'] = para_dict[key]['rho_max']
    PARA['u_free'] = para_dict[key]['u_max']
    PARA['tau'] = para_dict[key]['tau']
#loop

qhu_grn_sld = QHU_Green_Shield_augDerivative(PARA)
y = rho*(u - qhu_grn_sld.U(rho))
config = {'std_Q':[.02,.02],
              'std_R':[.3,.3],
              'init_rho': np.ones(rho.shape[0])*0.1,
              'init_u': np.ones(rho.shape[0])*0.5,
              'init_P': np.eye(rho.shape[0]+u.shape[0])*0.02}

X_pri, X_pos, K, P = implement_kf(loops, len(loops), len(loops), rho, y,  config, 
                            dx=dx,
                            dt=dt,
                            para=PARA)
Rho = X_pos[:21,:]
Y = X_pos[21:,:]
qhu_grn_sld = QHU_Green_Shield_augDerivative(PARA)
U = Y/Rho +qhu_grn_sld.U(Rho)
error_rho = np.linalg.norm(rho[:, :]-Rho ,2)/np.linalg.norm(rho[:, :],2)
errors_u = np.linalg.norm(u[:, :]-U,2)/np.linalg.norm(u[:, :],2)

