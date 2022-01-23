import numpy as np
import math

# function pool
# For 3 para, the definition of Q comes first
# For grn sld, the definition of U comes first
class QHU_3Para():
    def __init__(self, para):
        self.para = para
        self.para["u_free"] = self.init_u_free()
        
    def g(self, x):
        lamda = self.para['lambda']; p = self.para['p']
        temp = 1 + lamda ** 2 * (x - p) ** 2
        return np.sqrt(temp)
    
    def Q(self, rho):
        rho_max = self.para['rho_jam']; alpha = self.para['alpha']
        a = self.g(0)
        b = self.g(1)
        c = self.g(rho / rho_max)
        Q = alpha * (a + (b - a) * rho / rho_max - c)
        return Q
    
    def U(self, rho):
        para = self.para
        return self.Q(rho)/rho
    def h(self, rho):
        para = self.para
        if para["is_h_original_with_beta"] == 1:
            print('original h of 3 Para model')
            beta = para["beta"]; gamma = para["gamma"]; rho_jam = para["rho_jam"]
            temp = rho/rho_jam / ( 1-rho/rho_jam )
            #return beta*math.pow(temp, gamma)
            return beta * temp**gamma
        else:
            # by default use the simplest h function
            print('simplest h of 3 Para model')
            return para["u_free"] - self.U(rho)
        
    def init_u_free(self):
        para = self.para
        alpha = para["alpha"]; p = para["p"];
        lamda = para["lambda"]; rho_jam = para["rho_jam"]
        
        # calculate U0 by derivative of Q at rho=0
        lamda = 1/lamda # consistent with Kuang's definition
        U0 = alpha*( self.g(1) - self.g(0) + p/lamda**2/math.sqrt(1+p**2/lamda**2) ) / rho_jam
        return U0
        
class QHU_Green_Shield():
    def __init__(self, para):
        self.para = para
    def U(self, rho):
        para = self.para
        u_free = para["u_free"]; rho_jam = para["rho_jam"]
        return u_free*(1 - rho/rho_jam)
    def Q(self, rho):
        return rho*self.U(rho)
    def h(self, rho):
        para = self.para
        if para["is_h_original_with_beta"] == 1:
            print('original h of GSD model')
            beta = para["beta"]; gamma = para["gamma"]; rho_jam = para["rho_jam"]
            temp = rho/rho_jam / ( 1-rho/rho_jam )
            #return beta*math.pow(temp, gamma)
            return beta * temp**gamma
        else:
            print('simplest h of GSD model')
            return self.U(0) - self.U(rho)
        
            
            
class LF():
    def __init__(self, QHU, rho_init, u_init, rho_lb, rho_rb, u_lb, u_rb, dt, dx):
        """
        QHU: QHU class, either QHU_3para or QHU_green_shield
        rho_init, u_init: must be numpy array, of size L
        x_lb, x_rb: boundary condition, of size T
        dt, dx: float
        """
        self.QHU = QHU
        self.rho_init = rho_init.reshape(-1,1)
        self.u_init = u_init.reshape(-1,1)
        self.rho_lb = rho_lb
        self.rho_rb = rho_rb
        self.u_lb = u_lb
        self.u_rb = u_rb
        self.dt = dt
        self.dx = dx
        
        
        
        # intermediate value
        self.c = dt/dx
        self.lamda = dt/QHU.para["tau"] # this lamda is different from QHU.para['lambda']
        
        # CFL condition
        if self.c*QHU.para["u_free"] > 1:
            print("CFL condition not satisified")
        
    def rho_u_to_y(self, rho, u):
        return rho*( u + self.QHU.h(rho) )
    
    
    def simulate(self):
        assert len(self.rho_lb) == len(self.rho_rb) == \
                 len(self.u_lb) == len(self.u_rb)
        
        self.lamda = self.dt/self.QHU.para["tau"] # this lamda is different from QHU.para['lambda']
        
        T = len(self.rho_lb)
        rho = self.rho_init.reshape(-1,1)
        u = self.u_init.reshape(-1,1)
        # record
        Rho = [self.rho_init.reshape(-1,1)]
        U = [self.u_init.reshape(-1,1)]
        for i in range(T-1): # as there is no ground truth for the predction(should be t=T+1) at t=T
            rho_1, u_1 = self.onestep(rho, u,
                                     self.rho_lb[i], self.rho_rb[i],
                                     self.u_lb[i], self.u_rb[i])
            Rho.append(rho_1)
            U.append(u_1)
            
            rho = rho_1
            u = u_1
        Rho = np.hstack(Rho)
        U = np.hstack(U)
        return Rho, U
    
    def onestep(self, rho, u, rho_lb, rho_rb,  u_lb, u_rb):
        """
        Input:
        rho: current rho array
        u: current u array, must in the same size of rhos_0
        
        Output: 
        rho_1: next rho array
        u_1: next u array
        
        Auxiliary Input:
        rho_lb: left boundary of rho, approximated by rho_lb = rho[0]
        rho_rb: right boudary of rho, approximated by rho_ub =  rho[-1]
        """
        
        c = self.c; lamda = self.lamda
        
        assert (len(rho) == len(u)) 
        L = len(rho)
        # output
        rho_1 = np.ones((L+2, 1)) # 2 means the loc for aux input, for the consistency with input
        u_1 = np.ones((L+2, 1))   #, will delete later
        y_1 = np.ones((L+2, 1))
        
        # attach boundary value to the input
        rho = rho.reshape(-1,1) # column vector
        rho = np.vstack([rho_lb, rho, rho_rb])
        u = u.reshape(-1,1)
        u = np.vstack([u_lb, u, u_rb])
        y = self.rho_u_to_y(rho, u)
        
        # onestep in the solver
        for j in range(1, L+1):
            l = j-1; r = j+1
            # note currently the input and output are both of size L+2
            rho_1[j] = 0.5 * (rho[l] + rho[r]) - \
                0.5 * c * (rho[r] * u[r] - \
                rho[l] * u[l]);
            y_temp = 0.5 * (y[l] + y[r]) - \
                0.5 * c * (y[r] * u[r] - \
                y[l] * u[l])
            u_1[j] = (lamda * self.QHU.U(rho_1[j]) + \
                             y_temp / rho_1[j] - \
                             self.QHU.h(rho_1[j])) / (1 + lamda)
            
        # remove the auxiliary
        rho_1 = rho_1[1:-1]
        u_1 = u_1[1:-1]
        return rho_1, u_1
    