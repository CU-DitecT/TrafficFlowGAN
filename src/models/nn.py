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
    def __init__(self, z_dim, n_transformation, train, mean, std, device, s_args, t_args, s_kwargs, t_kwargs,
                z_miu_args,z_sigma_args,z_miu_kwargs,z_sigma_kwargs):
        super(RealNVP_lz, self).__init__()
        mask = get_mask(z_dim, n_transformation)
        mask_torch = torch.from_numpy(mask)
        self.mask = nn.Parameter(mask_torch, requires_grad=False)
        self.nn = get_fully_connected_layer(*t_args, **t_kwargs,mean=mean,std=std)
        self.net_miu = torch.nn.ModuleList([get_fully_connected_layer(*z_miu_args, **z_miu_kwargs,mean=mean,std=std,NNz=True)])
        self.net_sigma = torch.nn.ModuleList([get_fully_connected_layer(*z_sigma_args, **z_sigma_kwargs,mean=mean,std=std,NNz=True)])

        # hardcode: force the first s-net to have a tanh activation function
        #s_kwargs["last_activation_type"] = "tanh"
        #s[0] = get_fully_connected_layer(*s_args, **s_kwargs)
        self.device = device
        self.prior = torch.distributions.MultivariateNormal(torch.zeros(z_dim, device=device),
                                                            torch.eye(z_dim, device=device))
        self.train = (train == "True")
        self.model_D = torch.nn.Sequential(
            torch.nn.Linear(4,20),
            # torch.nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Tanh(),
            torch.nn.Linear(20,40),
            # torch.nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Tanh(),
            torch.nn.Linear(40,60),
            # torch.nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Tanh(),
            torch.nn.Linear(60,80),
            # torch.nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Tanh(),
            torch.nn.Linear(80,60),

            torch.nn.Tanh(),
            torch.nn.Linear(60,40),

            torch.nn.Tanh(),
            torch.nn.Linear(40,20),

            torch.nn.Tanh(),
            torch.nn.Linear(20, 1),
        )
        self.optimizer = torch.optim.Adam(self.model_D.parameters(), lr=0.001)
    def g(self, z, c):
        x = self.nn(torch.cat( (c, z), 1))
        return x
    
    def f(self, x, c):

        return None
    
    def NN_z(self, c):
        # c_ = torch.from_numpy(c).to(self.device)
        miu = self.net_miu[0](c)
        sigma = self.net_sigma[0](c)
        return miu, sigma

    def log_prob(self, x, c):
        miu, sigma = self.NN_z(torch.from_numpy(c).to(self.device))
        activation = {"miu": miu,
                      "sigma": sigma}
        return None, activation

    def eval(self, c):
        torch.manual_seed(1)
        z = self.prior.sample((c.shape[0], 1)).to(self.device)
        z = torch.squeeze(z)
        c_ = torch.from_numpy(c).to(self.device)
        # log_p = self.prior.log_prob(z, c)
        x = self.g(z, c_)
        activation = {"x1_eval": x[:, 0].cpu().detach().numpy(),
                      "x2_eval": x[:, 1].cpu().detach().numpy()}
        return activation

    def test(self, c):
        #torch.manual_seed(1)
        z = self.prior.sample((c.shape[0], 1)).to(self.device)
        z = torch.squeeze(z)
        miu,sigma = self.NN_z(c)
        # z_cali = z*sigma + miu
        z_cali = z
        # log_p = self.prior.log_prob(z, c)
        x = self.g(z_cali, c)
        ## hard code for Ngsim normalization
        # x = x*torch.from_numpy(np.array([2.0758793e-01, 1.0194696e+01])).to(self.device)+torch.from_numpy(np.array([7.2862007e-02, 3.8798647e+00])).to(self.device)
        return x[:, 0:1], x[:, 1:2]

    def training_gan(self, y, c, writer, epoch):
        c = torch.from_numpy(c).to(self.device)
        y = torch.from_numpy(y).to(self.device)
        for _ in range(1):
            self.optimizer.zero_grad()
            fake_y = self.test(c)
            real_y = y
            T_real = self.model_D(torch.cat((c,real_y),1))
            T_fake = self.model_D(torch.cat((c,fake_y[0],fake_y[1]),1))
            # loss_d = - (torch.log(1 - torch.sigmoid(T_real) + 1e-8) + torch.log(torch.sigmoid(T_fake) + 1e-8))
            loss_d = -T_real.mean() + T_fake.mean()
            loss_d.backward(retain_graph=True)
            self.optimizer.step()
            for p in self.model_D.parameters():
                p.data.clamp_(-0.01, 0.01)
            writer.add_scalar("loss/Discriminator_data_loss", loss_d.mean().cpu().detach().numpy(), epoch + 1)
        return -self.model_D(torch.cat((c,fake_y[0],fake_y[1]),1)).mean()



