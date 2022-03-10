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


class Normalization(nn.Module):
    def __init__(self, mean, std, device):
        super(Normalization, self).__init__()
        self.device = device
        self.mean = torch.from_numpy(mean).to(self.device)
        self.std = torch.from_numpy(std).to(self.device)

    def forward(self, tensors):
        norm_tensor = (tensors - self.mean) / self.std
        return norm_tensor

class RealNVP_lz(nn.Module):
    def __init__(self, z_dim, n_transformation, train, mean, std, device, s_args, t_args, s_kwargs, t_kwargs,
                z_miu_args,z_sigma_args,z_miu_kwargs,z_sigma_kwargs):
        super(RealNVP_lz, self).__init__()
        mask = get_mask(z_dim, n_transformation)
        mask_torch = torch.from_numpy(mask)
        self.mean = mean
        self.std = std
        self.mask = nn.Parameter(mask_torch, requires_grad=False)
        self.nn = get_fully_connected_layer(*t_args, **t_kwargs,mean=mean,std=std)
        self.net_miu = torch.nn.ModuleList([get_fully_connected_layer(*z_miu_args, **z_miu_kwargs,mean=mean,std=std,NNz=True)])
        self.net_sigma = torch.nn.ModuleList([get_fully_connected_layer(*z_sigma_args, **z_sigma_kwargs,mean=mean,std=std,NNz=True)])
        self.n_G = 0
        # hardcode: force the first s-net to have a tanh activation function
        #s_kwargs["last_activation_type"] = "tanh"
        #s[0] = get_fully_connected_layer(*s_args, **s_kwargs)
        self.device = device
        self.prior = torch.distributions.MultivariateNormal(torch.zeros(z_dim, device=device),
                                                            torch.eye(z_dim, device=device)*0.05)
        self.train = (train == "True")
        self.switch_epoch = 10000

        self.model_D = torch.nn.Sequential(
            torch.nn.Conv2d(2, 4, 3, 2, 2, bias=False),
            torch.nn.Tanh(),
            # torch.nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Conv2d(4, 8, 3, 2, 2, bias=False),
            torch.nn.BatchNorm2d(8),
            torch.nn.Tanh(),
            # torch.nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Conv2d(8, 16, 3, 2, 2, bias=False),
            torch.nn.BatchNorm2d(16),
            # torch.nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Tanh(),
            torch.nn.Conv2d(16, 24, 3, 2, 2, bias=False),
            torch.nn.BatchNorm2d(24),
            # torch.nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Tanh(),
            torch.nn.Conv2d(24, 1, (8, 3), 1, 0, bias=False),
        )
        def init_weights(m):
            if isinstance(m,torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.model_D.apply(init_weights)
        # for p in self.model_D.parameters():
        #     torch.nn.init.xavier_uniform_(p)
        self.optimizer = torch.optim.Adam(self.model_D.parameters(), lr=0.001)
    def g(self, z, c):
        if len(c.shape)>2:
            x = self.nn(torch.cat( (z,c), 1).permute(0, 3, 2, 1)).permute(0,3,2,1)
        else:
            x = self.nn(torch.cat((z,c), 1))
        return x
    
    def f(self, x, c):

        return None
    
    def NN_z(self, c):
        # c_ = torch.from_numpy(c).to(self.device)
        miu = self.net_miu[0](c)
        sigma = self.net_sigma[0](c)
        return miu, sigma

    def log_prob(self, x, c):
        miu, sigma = 0,0
        activation = {"miu": miu,
                      "sigma": sigma}
        return None, activation

    def eval(self, c):
        # torch.manual_seed(1)
        z = self.prior.sample((c.shape[0], c.shape[2], c.shape[3])).to(self.device)
        z = z.permute(0, 3, 1, 2)
        z = torch.squeeze(z)
        c_ = torch.from_numpy(c).to(self.device)
        # log_p = self.prior.log_prob(z, c)
        x = self.g(z, c_)
        activation = {"x1_eval": x[:, 0].cpu().detach().numpy(),
                      "x2_eval": x[:, 1].cpu().detach().numpy()}
        return activation

    def test(self, c):
        #torch.manual_seed(1)
        if len(c.shape)>2:
            z = self.prior.sample((c.shape[0], c.shape[2], c.shape[3])).to(self.device)
            z = z.permute(0, 3, 1, 2)
        else:
            z = self.prior.sample((c.shape[0],1)).to(self.device)
        z = torch.squeeze(z)
        # miu,sigma = self.NN_z(c)
        # z_cali = z*sigma + miu
        z_cali = z
        # log_p = self.prior.log_prob(z, c)
        x = self.g(z_cali, c)
        ## hard code for Ngsim normalization
        # x = x*torch.from_numpy(np.array([2.0758793e-01, 1.0194696e+01])).to(self.device)+torch.from_numpy(np.array([7.2862007e-02, 3.8798647e+00])).to(self.device)
        return x

    def training_gan(self, y, c, writer, epoch, train= None):

        c = torch.from_numpy(c).to(self.device)
        y = torch.from_numpy(y).to(self.device)
        if train:
            n_D = 0
            while True:
                n_D += 1
                self.optimizer.zero_grad()
                fake_y = self.test(c)
                real_y = y
                T_real = self.model_D(real_y)
                T_fake = self.model_D(fake_y.detach())
                if (T_fake.mean()>0.1 or n_D>5) and (epoch < self.switch_epoch):
                    break

                writer.add_scalar("loss/T_real", T_real.mean().cpu().detach().numpy(), epoch + 1)
                writer.add_scalar("loss/T_fake", T_fake.mean().cpu().detach().numpy(), epoch + 1)

                loss_d = - (torch.log(1 - torch.sigmoid(T_real) + 1e-8) + torch.log(torch.sigmoid(T_fake) + 1e-8))
                # loss_d = -T_real.mean() + T_fake.mean()
                loss_d.mean().backward(retain_graph=True)
                self.optimizer.step()
                # for p in self.model_D.parameters():
                #     p.data.clamp_(-0.01, 0.01)
                writer.add_scalar("loss/Discriminator_data_loss", loss_d.mean().cpu().detach().numpy(), epoch + 1)
                if epoch >= self.switch_epoch:
                    break
        else:
            fake_y = self.test(c)
        # wgan:
        return self.model_D(fake_y).mean()

        # vanilla gan
        # return self.model_D(torch.cat((fake_y[0],fake_y[1]),1)).mean()

        # mse
        # return torch.square(torch.cat((fake_y[0],fake_y[1]), 1) - real_y ).mean()

