from torch import nn
import torch
import math
import numpy as np
from torch._C import device
from src.layers.fully_connected import get_fully_connected_layer
import sys

def get_mask(z_dim, n_transformation):
    mask = np.ones((n_transformation, z_dim), dtype=np.float32)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if (i + j) % 2 == 0:
                mask[i, j] = 0
    return mask


class RealNVP_lz(nn.Module):
    def __init__(self, z_dim, n_transformation, train, mean, std, device, s_args, t_args, s_kwargs, t_kwargs,
                z_miu_args,z_sigma_args,z_miu_kwargs,z_sigma_kwargs):
        super(RealNVP_lz, self).__init__()
        mask = get_mask(z_dim, n_transformation)
        mask_torch = torch.from_numpy(mask)
        self.mask = nn.Parameter(mask_torch, requires_grad=False)
        t = [get_fully_connected_layer(*t_args, **t_kwargs,mean=mean,std=std) for _ in range(mask.shape[0])]
        s = [get_fully_connected_layer(*s_args, **s_kwargs,mean=mean,std=std) for _ in range(mask.shape[0])]
        self.net_miu = torch.nn.ModuleList([get_fully_connected_layer(*z_miu_args, **z_miu_kwargs,mean=mean,std=std,NNz=True)])
        self.net_sigma = torch.nn.ModuleList([get_fully_connected_layer(*z_sigma_args, **z_sigma_kwargs,mean=mean,std=std,NNz=True)])

        # hardcode: force the first s-net to have a tanh activation function
        #s_kwargs["last_activation_type"] = "tanh"
        #s[0] = get_fully_connected_layer(*s_args, **s_kwargs)
        self.device = device
        self.t = torch.nn.ModuleList(t)
        self.s = torch.nn.ModuleList(s)
        self.prior = torch.distributions.MultivariateNormal(torch.zeros(z_dim, device=device),
                                                            torch.eye(z_dim, device=device))
        self.train = (train == "True")
        self.mean = mean # 4 dim
        self.std = std # 4 dim

    def g(self, z, c):
        # transform from z to x
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](
                torch.cat((x_, c), 1))
            t = self.t[i](
                torch.cat((x_, c), 1))
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x
    
    def f(self, x, c):
        # transform from x to z
        ## hard code for Ngsim normalization
#         self.mean = np.array([2.0758793e-01, 1.0194696e+01])
#         self.std = np.array([7.2862007e-02, 3.8798647e+00])

        if c.shape[1]==2:
            activation = {"x1": x[:, 0],
                      "x2": x[:, 1],
                      "c_1": c[:,0],
                      "c_2": c[:,1]}
        else:
            activation = {"x1": x[:, 0],
                      "x2": x[:, 1],
                      "c_1": c[:,0]}
        #x = (x - self.mean[:2]) / self.std[:2]
        z = torch.from_numpy(x.astype(np.float32)).float().to(self.device) # z = x
        if torch.is_tensor(c):
            c_ = c.to(self.device)
        else:
            c_ = torch.from_numpy(c.astype(np.float32)).float().to(self.device)
        
        log_det_J = z.new_zeros(x.shape[0])  # log_det_J = x.new_zeros(x.shape[0])

        for i in reversed(range(len(self.t))):
            st_id = np.where(self.mask[i].cpu().numpy()==0)[0][0] # because mask=1 means keep the same value; mask=0 means using affine_coupling_layer
            z_ = self.mask[i]* z  # z_ = (self.mask[i] * z)            
            s = (1 - self.mask[i]) * self.s[i](
                torch.cat((z_, c_), 1))
            s = torch.clamp(s, min=-5, max=5)
            t = (1 - self.mask[i]) * self.t[i](
                torch.cat((z_, c_), 1))
            t = torch.clamp(t, min=-5, max=5)
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)

            # save the activation            
            activation[f"s{i+1:d}"] = s[:, st_id]
            activation[f"t{i+1:d}"] = t[:, st_id]
            activation[f"z{i:d}_1"] = z[:, 0]
            activation[f"z{i:d}_2"] = z[:, 1]
                  
        return z, log_det_J, activation
    
    def NN_z(self, c):
        # c_ = torch.from_numpy(c).to(self.device)
        miu = self.net_miu[0](c)
        sigma = self.net_sigma[0](c)
        return miu, sigma

    def log_prob(self, x, c):
        z, log_p, activation = self.f(x, c)
        if torch.is_tensor(c):
            miu, sigma = self.NN_z(c.to(self.device))
        else:
            miu, sigma = self.NN_z(torch.from_numpy(c).to(self.device))

        # hard code here
        sigma = torch.sigmoid(sigma) + 0.5


        L = 0.5*torch.log(torch.tensor([2*math.pi], device=self.device))+torch.log(sigma)+torch.div(torch.mul((z-miu),(z-miu)),2*torch.mul(sigma,sigma))
        L = L[:,0:1]+L[:,1:2]

        activation["miu"] = miu
        activation["sigma"] = sigma
        activation["L"] = -L.squeeze()
        activation["log_p"] = log_p

        return -L.squeeze() + log_p , activation

    def eval(self, c):
        torch.manual_seed(1)
        z = self.prior.sample((c.shape[0], 1)).to(self.device)
        z = torch.squeeze(z)
        if torch.is_tensor(c):
            c_ = c.to(self.device)
        else:
            c_ = torch.from_numpy(c).to(self.device)
        # log_p = self.prior.log_prob(z, c)
        x = self.g(z, c_)
        x = x * torch.from_numpy(self.std[:2]).to(self.device) + torch.from_numpy(self.mean[:2]).to(self.device)
        activation = {"x1_eval": x[:, 0].cpu().detach().numpy(),
                      "x2_eval": x[:, 1].cpu().detach().numpy()}
        return activation

    def test(self, c):
        #torch.manual_seed(1)
        z = self.prior.sample((c.shape[0], 1)).to(self.device)
        z = torch.squeeze(z)
        miu,sigma = self.NN_z(c)
        #miu,_ = self.NN_z(c)
        #sigma=0.0

        z_cali = z*sigma + miu
        # log_p = self.prior.log_prob(z, c)
        x = self.g(z_cali, c)
        ## hard code for Ngsim normalization
        #x = x*torch.from_numpy(self.std[:2]).to(self.device)+torch.from_numpy(self.mean[:2]).to(self.device)
        return x[:, 0:1], x[:, 1:2]

class MO_RealNVP_lz(nn.Module):
    def __init__(self, z_dim, n_transformation, train, mean, std, device, s_args, t_args, s_kwargs, t_kwargs,
                z_miu_args,z_sigma_args,z_miu_kwargs,z_sigma_kwargs):
        super(MO_RealNVP_lz, self).__init__()
        mask = get_mask(z_dim, n_transformation)
        mask_torch = torch.from_numpy(mask)
        self.mask = nn.Parameter(mask_torch, requires_grad=False)
        t = [get_fully_connected_layer(*t_args, **t_kwargs,mean=mean,std=std) for _ in range(mask.shape[0])]
        s = [get_fully_connected_layer(*s_args, **s_kwargs,mean=mean,std=std) for _ in range(mask.shape[0])]
        self.net_miu = torch.nn.ModuleList([get_fully_connected_layer(*z_miu_args, **z_miu_kwargs,mean=mean,std=std,NNz=True)])
        self.net_sigma = torch.nn.ModuleList([get_fully_connected_layer(*z_sigma_args, **z_sigma_kwargs,mean=mean,std=std,NNz=True)])

        # hardcode: force the first s-net to have a tanh activation function
        #s_kwargs["last_activation_type"] = "tanh"
        #s[0] = get_fully_connected_layer(*s_args, **s_kwargs)
        self.device = device
        self.t = torch.nn.ModuleList(t)
        self.s = torch.nn.ModuleList(s)
        self.prior = torch.distributions.MultivariateNormal(torch.zeros(z_dim, device=device),
                                                            torch.eye(z_dim, device=device))
        self.train = (train == "True")
        self.mean = mean # 4 dim
        self.std = std # 4 dim

    def g(self, z, c):
        # transform from z to x
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](
                torch.cat((x_, c), 1))
            t = self.t[i](
                torch.cat((x_, c), 1))
            x = x_ + (1 - self.mask[i]) * (x * torch.sigmoid(s) + t)
        return x
    
    def f(self, x, c):
        # transform from x to z
        ## hard code for Ngsim normalization
#         self.mean = np.array([2.0758793e-01, 1.0194696e+01])
#         self.std = np.array([7.2862007e-02, 3.8798647e+00])

        if c.shape[1]==2:
            activation = {"x1": x[:, 0],
                      "x2": x[:, 1],
                      "c_1": c[:,0],
                      "c_2": c[:,1]}
        else:
            activation = {"x1": x[:, 0],
                      "x2": x[:, 1],
                      "c_1": c[:,0]}
        #x = (x - self.mean[:2]) / self.std[:2]
        z = torch.from_numpy(x.astype(np.float32)).float().to(self.device) # z = x
        if torch.is_tensor(c):
            c_ = c.to(self.device)
        else:
            c_ = torch.from_numpy(c.astype(np.float32)).float().to(self.device)
        
        log_det_J = z.new_zeros(x.shape[0])  # log_det_J = x.new_zeros(x.shape[0])

        for i in reversed(range(len(self.t))):
            st_id = np.where(self.mask[i].cpu().numpy()==0)[0][0] # because mask=1 means keep the same value; mask=0 means using affine_coupling_layer
            z_ = self.mask[i]* z  # z_ = (self.mask[i] * z)            
            s = (1 - self.mask[i]) * self.s[i](
                torch.cat((z_, c_), 1))
            s = torch.clamp(s, min=-5, max=5)
            t = (1 - self.mask[i]) * self.t[i](
                torch.cat((z_, c_), 1))
            t = torch.clamp(t, min=-5, max=5)
            z = (1 - self.mask[i]) * (z - t) / torch.sigmoid(s) + z_
            log_det_J -= torch.log(torch.abs(torch.sigmoid(s))).sum(dim=1)
            #log_det_J -= s.sum(dim=1)



            # save the activation            
            activation[f"s{i+1:d}"] = s[:, st_id]
            activation[f"t{i+1:d}"] = t[:, st_id]
            activation[f"z{i:d}_1"] = z[:, 0]
            activation[f"z{i:d}_2"] = z[:, 1]
                  
        return z, log_det_J, activation
    
    def NN_z(self, c):
        # c_ = torch.from_numpy(c).to(self.device)
        miu = self.net_miu[0](c)
        sigma = self.net_sigma[0](c)
        return miu, sigma

    def log_prob(self, x, c):
        z, log_p, activation = self.f(x, c)
        if torch.is_tensor(c):
            miu, sigma = self.NN_z(c.to(self.device))
        else:
            miu, sigma = self.NN_z(torch.from_numpy(c).to(self.device))

        # hard code here
        sigma = torch.sigmoid(sigma) + 0.5


        L = 0.5*torch.log(torch.tensor([2*math.pi], device=self.device))+torch.log(sigma)+torch.div(torch.mul((z-miu),(z-miu)),2*torch.mul(sigma,sigma))
        L = L[:,0:1]+L[:,1:2]

        activation["miu"] = miu
        activation["sigma"] = sigma
        activation["L"] = -L.squeeze()
        activation["log_p"] = log_p

        return -L.squeeze() + log_p , activation

    def eval(self, c):
        torch.manual_seed(1)
        z = self.prior.sample((c.shape[0], 1)).to(self.device)
        z = torch.squeeze(z)
        if torch.is_tensor(c):
            c_ = c.to(self.device)
        else:
            c_ = torch.from_numpy(c).to(self.device)
        # log_p = self.prior.log_prob(z, c)
        x = self.g(z, c_)
        x = x * torch.from_numpy(self.std[:2]).to(self.device) + torch.from_numpy(self.mean[:2]).to(self.device)
        activation = {"x1_eval": x[:, 0].cpu().detach().numpy(),
                      "x2_eval": x[:, 1].cpu().detach().numpy()}
        return activation

    def test(self, c):
        #torch.manual_seed(1)
        z = self.prior.sample((c.shape[0], 1)).to(self.device)
        z = torch.squeeze(z)
        miu,sigma = self.NN_z(c)
        #miu,_ = self.NN_z(c)
        #sigma=0.0

        z_cali = z*sigma + miu
        # log_p = self.prior.log_prob(z, c)
        x = self.g(z_cali, c)
        ## hard code for Ngsim normalization
        #x = x*torch.from_numpy(self.std[:2]).to(self.device)+torch.from_numpy(self.mean[:2]).to(self.device)
        return x[:, 0:1], x[:, 1:2]

### edited graph conditional flow for atoms in MoFlow:
class MOAffineCoupling(nn.Module):
    def __init__(self, n_node, in_dim, hidden_dim_dict, masked_row, affine=True):
        super(MOAffineCoupling, self).__init__()
        self.n_node = n_node
        self.in_dim = in_dim
        #self.hidden_dim_dict = hidden_dim_dict
        self.masked_row = masked_row
        self.affine = affine #True

        #self.hidden_dim_gnn = hidden_dim_dict['gnn']
        self.hidden_dim_linear = hidden_dim_dict['linear'] #number of hidden layer in each s or t

        """ ignore graph neural network
        self.net = nn.ModuleList()
        self.norm = nn.ModuleList()
        last_dim = in_dim
        for out_dim in self.hidden_dim_gnn:  # What if use only one gnn???
            #self.net.append(GraphConv(last_dim, out_dim))
            self.norm.append(nn.BatchNorm1d(n_node))  # , momentum=0.9 Change norm!!!
            # self.norm.append(ActNorm2D(in_dim=n_node, logdet=False))
            last_dim = out_dim
        """

        self.net_lin = nn.ModuleList()
        self.norm_lin = nn.ModuleList()
        for out_dim in self.hidden_dim_linear:  # What if use only one gnn???
            self.net_lin.append(nn.linear(last_dim, in_dim*2))#self.net_lin.append(GraphLinear(last_dim, in_dim*2))
            self.norm_lin.append(nn.BatchNorm1d(n_node))  # , momentum=0.9 Change norm!!!
            # self.norm_lin.append(ActNorm2D(in_dim=n_node, logdet=False))
            last_dim = out_dim

        if affine:
            self.net_lin.append(nn.linear(last_dim, in_dim*2))#self.net_lin.append(GraphLinear(last_dim, in_dim*2))
        else:
            self.net_lin.append(nn.linear(last_dim, in_dim))#self.net_lin.append(GraphLinear(last_dim, in_dim))

        self.scale = nn.Parameter(torch.zeros(1))  # nn.Parameter(torch.ones(1)) #
        mask = torch.ones(n_node, in_dim)
        mask[masked_row, :] = 0  # masked_row are kept same, and used for _s_t for updating the left rows
        self.register_buffer('mask', mask)

    def forward(self, adj, input): #from x to z
        masked_x = self.mask * input
        s, t = self._s_t_function(adj, masked_x)  # s must not equal to 0!!!
        if self.affine:
            out = masked_x + (1-self.mask) * (input + t) * s
            # out = masked_x + (1-self.mask) * (input * s + t)
            logdet = torch.sum(torch.log(torch.abs(s)).view(input.shape[0], -1), 1)  # possibly wrong answer
        else:  # add coupling
            out = masked_x + t*(1-self.mask)
            logdet = None
        return out, logdet

    def reverse(self, adj, output):
        masked_y = self.mask * output
        s, t = self._s_t_function(adj, masked_y)
        if self.affine:
            input = masked_y + (1 - self.mask) * (output / s - t)
            # input = masked_x + (1 - self.mask) * ((output-t) / s)
        else:
            input = masked_y + (1 - self.mask) * (output - t)
        return input

    def _s_t_function(self, adj, x):
        # adj: (2,4,9,9)  x: # (2,9,5)
        s = None
        h = x
        """ignore Graph neural network
        for i in range(len(self.net)):
            h = self.net[i](adj, h)  # (2,1,9,hidden_dim)
            h = self.norm[i](h)
            # h = torch.tanh(h)  # tanh may be more stable
            h = torch.relu(h)  # use relu!!!
        """

        for i in range(len(self.net_lin)-1):
            h = self.net_lin[i](h)  # (2,1,9,hidden_dim)
            h = self.norm_lin[i](h)
            # h = torch.tanh(h)
            h = torch.relu(h)

        h = self.net_lin[-1](h)
        # h =h * torch.exp(self.scale*2)

        if self.affine:
            log_s, t = h.chunk(2, dim=-1)
            #  x = sigmoid(log_x+bias): glow code Top 1 choice, keep s away from 0, s!!!!= 0  always safe!!!
            # And get the signal from added noise in the  input
            # s = torch.sigmoid(log_s + 2)
            s = torch.sigmoid(log_s)  # better validity + actnorm

            # s = torch.tanh(log_s)  # Not stable when s =0 for synthesis data, but works well for real data in best case....
            # s = torch.sign(s)

            # s = torch.sign(log_s)

            # s = F.softplus(log_s) # negative nll
            # s = torch.sigmoid(log_s)  # little worse than +2, # *self.scale #!!! # scale leads to nan results
            # s = torch.tanh(log_s+2) # not that good
            # s = torch.relu(log_s) # nan results
            # s = log_s  # nan results
            # s = torch.exp(log_s)  # nan results
        else:
            t = h
        return s, t

