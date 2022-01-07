import scipy.io
import numpy as np

X_dim = 1 
Y_dim = 1
T_dim = 1
Z_dim = 1

# Noise level (in this noise free case is zero)
noise = 0.02
N_noise = 20 
N_u = 100000
N_loop = 8

data = scipy.io.loadmat('rho_bellshape_10grid_DS10_gn_eps005_solver2_ring.mat')

t = data['t'].flatten()[:,None]# 960 by 1
x = data['x'].flatten()[:,None]# 240 by 1
Exact_rho = np.real(data['rho']).T

print(len(x), len(x[0]))
print(N_loop-1)


X, T = np.meshgrid(x,t) #

X_repeat = X.flatten()[:,None].repeat(N_noise, axis=1)
T_repeat = T.flatten()[:,None].repeat(N_noise, axis=1)
# all points inside
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) # hstack is column wise stack, 960*240 by 2
index_x = np.where((X.flatten()[:,None] >= 0.4) & (X.flatten()[:,None] <=0.6))
index_t = np.where((T.flatten()[:,None] >= 0) & (T.flatten()[:,None] <=0.5))
print(index_x[0].shape,index_t[0].shape)
index_shock_wave = np.intersect1d(index_x[0], index_t[0])
print(index_shock_wave)
# data2 = np.array([0.0]*650)
idx2 = np.random.choice(range(960), 650, replace=False)
data2 = np.array([0.0]*960)
idx2 = np.array(list(range(960))) #np.random.choice(range(960), 650, replace=False)
# t2 = t[idx].T#np.array(idx2).T

# low boundary
X_star2 = np.hstack((data2.flatten()[:,None], t.flatten()[:,None]))
X_star2_repeat = np.repeat(X_star2,N_noise,axis = 0)

idx3 = np.random.choice(range(960*241),  N_u, replace=False)
X_star3 = X_star[idx3,:]
X_star3_repeat = np.repeat(X_star3,N_noise,axis = 0)


rho_star = Exact_rho.flatten()[:,None] # 960*240 by 1


# Doman bounds
lb = X_star.min(0) # [0, 0]
ub = X_star.max(0) # [1, 3] 

print(lb)
print(ub)

gap = int(240.0/(N_loop-1))
loopidx = [i*gap for i in range(N_loop-1)]
if N_loop == 4: loopidx[-1] = 120
loopidx.append(239)
span = np.array(loopidx)

idx = []
for i in range(960):
    seg = list(span+i*241)
    idx+= seg

print(len(idx), span)
print(len(idx),'the number of record on loops')
X_rho_train = X_star[idx,:]
# add shcock wave
# X_rho_shockwave = X_star[index_shock_wave,:]
# X_rho_train = np.vstack((X_rho_train,X_rho_shockwave))
##
X_rho_repeat = np.repeat(X_rho_train,N_noise,axis = 0)
print("X_rho_repeat:",X_rho_repeat.shape)
rho_train = rho_star[idx,:]
# add shcok wave
# rho_train_shockwave = rho_star[index_shock_wave,:]
# rho_train = np.vstack((rho_train,rho_train_shockwave))
##
print(rho_train.shape)
# rho_train = rho_train + noise*np.random.randn(rho_train.shape[0], rho_train.shape[1])
rho_train_repeat = rho_train + noise*np.random.randn(rho_train.shape[0], rho_train.shape[1])
for i in range(N_noise-1):
    rho_train_repeat = np.hstack((rho_train_repeat, rho_train + noise*np.random.randn(rho_train.shape[0], rho_train.shape[1])))
rho_train_repeat = rho_train_repeat.reshape(-1,1)
print("rho_train_repeat:",rho_train_repeat.shape)

# Model creation
layers_P = np.array([X_dim+T_dim+Z_dim,20,20,40,60,80,60,40,20,20,Y_dim])
layers_Q = np.array([X_dim+T_dim+Y_dim,20,20,40,60,80,60,40,20,20,Z_dim])  
layers_T = np.array([X_dim+T_dim+Y_dim,20,20,40,60,80,60,40,20,20,1])
