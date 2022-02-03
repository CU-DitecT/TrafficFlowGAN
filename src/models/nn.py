import torch.distributions
from torch import nn
import troch
import numpy as np
from torch._C import device
from src.layers.fully_connected import get_fully_connected_layer


class NN_lz(nn.Module):
    def __init(self, z_dim, train, mean, std, device, nn_args, nn_kwargs,
               z_miu_args, z_sigma_args, z_miu_kwargs, z_sigma_kwargs):
        super(NN, self).__init()
        self.nn = get_fully_connected_layer(*nn_args, **nn_kwargs)
        self.net_miu = get_fully_connected_layer(*z_miu_args, **z_miu_kwargs,
                                                 mean = mean,
                                                 std = std,
                                                 NNz = True)
        self.net_sigma = get_fully_connected_layer(*z_sigma_args, **z_sigma_kwargs,
                                                   mean = mean,
                                                   std = std,
                                                   NNz = True)

        self.device = device
        self.priot = torch.distributions.MultivariateNormal(torch.zeros(z_dim, device=device),
                                                            torch.eye(z_dim, device=device)*0.05)
        self.train = (train == "True")


    def NN_z(self, c):
        miu = self.net_miu(c)
        sigma = self.net_sigma(c)
        return miu, sigma

    def forward(self, c):
        z = self.prior.sample((c.shape[0], 1)).to(self.device)
        z = torch.squeeze(z)
        miu, sigma = self.NN_z(c)
        z_cali = z * sigma + miu
        # log_p = self.prior.log_prob(z, c)
        x = self.nn(torch.concat((c, z), 1))
        ## hard code for Ngsim normalization
        x = x * torch.from_numpy(np.array([2.0758793e-01, 1.0194696e+01])).to(self.device) + torch.from_numpy(
            np.array([7.2862007e-02, 3.8798647e+00])).to(self.device)
        return x[:, 0:1], x[:, 1:2]

