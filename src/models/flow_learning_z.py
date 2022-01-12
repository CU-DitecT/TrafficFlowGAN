from torch import nn
import torch
import math
import numpy as np
from torch._C import device
from src.layers.fully_connected import get_fully_connected_layer


def get_mask(z_dim, n_transformation):
    mask = np.ones((n_transformation, z_dim), dtype=np.float32)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if (i + j) % 2 == 0:
                mask[i, j] = 0
    return mask


class RealNVP_lz(nn.Module):
    def __init__(self, z_dim, n_transformation, train, device, s_args, t_args, s_kwargs, t_kwargs,
                z_miu_args,z_sigma_args,z_miu_kwargs,z_sigma_kwargs):
        super(RealNVP_lz, self).__init__()
        mask = get_mask(z_dim, n_transformation)
        mask_torch = torch.from_numpy(mask)
        self.mask = nn.Parameter(mask_torch, requires_grad=False)
        t = [get_fully_connected_layer(*t_args, **t_kwargs) for _ in range(mask.shape[0])]
        s = [get_fully_connected_layer(*s_args, **s_kwargs) for _ in range(mask.shape[0])]
        self.net_miu = torch.nn.ModuleList([get_fully_connected_layer(*z_miu_args, **z_miu_kwargs)])
        self.net_sigma = torch.nn.ModuleList([get_fully_connected_layer(*z_sigma_args, **z_sigma_kwargs)])

        # hardcode: force the first s-net to have a tanh activation function
        #s_kwargs["last_activation_type"] = "tanh"
        #s[0] = get_fully_connected_layer(*s_args, **s_kwargs)
        self.device = device
        self.t = torch.nn.ModuleList(t)
        self.s = torch.nn.ModuleList(s)
        self.prior = torch.distributions.MultivariateNormal(torch.zeros(z_dim), torch.eye(z_dim)*0.05)
        self.train = (train == "True")

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
        activation = {"x1": x[:, 0],
                      "x2": x[:, 1],
                      "c_1": c[:,0],
                      "c_2": c[:,1]}
        z = torch.from_numpy(x).to(self.device) # z = x
        c_ = torch.from_numpy(c).to(self.device)
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
        miu, sigma = self.NN_z(torch.from_numpy(c).to(self.device))
        L = 0.5*torch.log(torch.Tensor([2*math.pi]))+torch.log(sigma)+torch.div(torch.mul((z-miu),(z-miu)),2*torch.mul(sigma,sigma))
        L = L[:,0:1]+L[:,1:2]

        return -L  + log_p , activation

    def eval(self, c):
        torch.manual_seed(1)
        z = self.prior.sample((c.shape[0], 1))
        z = torch.squeeze(z)
        c_ = torch.from_numpy(c).to(self.device)
        # log_p = self.prior.log_prob(z, c)
        x = self.g(z, c_)
        activation = {"x1_eval": x[:, 0],
                      "x2_eval": x[:, 1]}
        return activation

    def test(self, c):
        #torch.manual_seed(1)
        z = self.prior.sample((c.shape[0], 1))
        z = torch.squeeze(z)
        miu,sigma = self.NN_z(c)
        z_cali = z*sigma + miu
        # log_p = self.prior.log_prob(z, c)
        x = self.g(z_cali, c)        
        return x[:, 0:1], x[:, 1:2]
