from torch import nn
import torch
import numpy as np
from src.layers.fully_connected import get_fully_connected_layer


def get_mask(z_dim, n_transformation):
    mask = np.ones((n_transformation, z_dim))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if (i+j)%2 == 0:
                mask[i,j] = 0
    return mask


class RealNVP(nn.Module):
    def __init__(self, z_dim, n_transformation, s_args, t_args,  s_kwargs, t_kwargs):
        super(RealNVP, self).__init__()
        mask = get_mask(z_dim, n_transformation)
        self.mask = nn.Parameter(mask, requires_grad=False)
        t = [get_fully_connected_layer(*t_args, **t_kwargs) for _ in range(mask.shape[0])]
        s = [get_fully_connected_layer(*s_args, **s_kwargs) for _ in range(mask.shape[0])]
        self.t = torch.nn.ModuleList(t)
        self.s = torch.nn.ModuleList(s)

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
        z = x
        log_det_J = x.new_zeros(x.shape[0])
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = (1 - self.mask[i]) * self.s[i](
                torch.cat((z_, c), 1))
            t = (1 - self.mask[i]) * self.t[i](
                torch.cat((z_, c), 1))
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x, c):
        z, log_p = self.f(x, c)
        return self.prior.log_prob(z) + log_p

    def test(self, c):
        z = self.prior.sample((c.shape[0], 1))
        z = torch.squeeze(z)
        # log_p = self.prior.log_prob(z, c)
        x = self.g(z, c)
        return x[:,0:1], x[:,1:2]






