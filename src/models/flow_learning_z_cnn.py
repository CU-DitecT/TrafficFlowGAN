from torch import nn
import torch
import math
import numpy as np
from torch._C import device
from src.layers.fully_connected import get_fully_connected_layer

class Normalization(nn.Module):
    def __init__(self, mean, std, device):
        super(Normalization, self).__init__()
        self.device = device
        self.mean = torch.from_numpy(mean).to(self.device)
        self.std = torch.from_numpy(std).to(self.device)

    def forward(self, tensors):
        norm_tensor = (tensors - self.mean) / self.std
        return norm_tensor

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
                                                            torch.eye(z_dim, device=device)*0.05)
        self.train = (train == "True")
        self.mean = mean
        self.std = std
        self.n_G = 0
        self.switch_epoch = 1
        self.model_D = torch.nn.Sequential(
            torch.nn.Conv2d(2, 4, 3, 2, 2, bias=False),
            torch.nn.Tanh(),
            # torch.nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Conv2d(4, 8, 3, 2, 2, bias=False),
            torch.nn.Tanh(),
            # torch.nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Conv2d(8, 16, 3, 2, 2, bias=False),
            # torch.nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Tanh(),
            torch.nn.Conv2d(16, 24, 3, 2, 2, bias=False),
            # torch.nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Tanh(),
            torch.nn.Conv2d(24, 1, (8, 3), 1, 0, bias=False),
        )

        def init_weights(m):
            if isinstance(m,torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.model_D.apply(init_weights)
        self.optimizer = torch.optim.Adam(self.model_D.parameters(), lr=0.001)
    def g(self, z, c):
        # transform from z to x
        if len(c.shape) > 2:
            x = z.permute(0, 3, 2, 1)
            c = c.permute(0, 3, 2, 1)
        else:
            x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](
                torch.cat((x_, c), -1))
            t = self.t[i](
                torch.cat((x_, c), -1))
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)

        return x
    
    def f(self, x, c):
        # transform from x to z

        # x = (x-self.mean[:2,])/self.std[:2,]
        activation = {"x1": x[:, 0],
                      "x2": x[:, 1],
                      "c_1": c[:,0],
                      "c_2": c[:,1]}
        z = torch.from_numpy(x.astype(np.float32)).float().permute(0, 3, 2, 1).to(self.device) # z = x
        c_ = torch.from_numpy(c.astype(np.float32)).float().permute(0, 3, 2, 1).to(self.device)
        log_det_J = z.new_zeros(z.shape[0],z.shape[1],z.shape[2])  # log_det_J = x.new_zeros(x.shape[0])
        for i in reversed(range(len(self.t))):
            st_id = np.where(self.mask[i].cpu().numpy()==0)[0][0] # because mask=1 means keep the same value; mask=0 means using affine_coupling_layer
            z_ = self.mask[i]* z  # z_ = (self.mask[i] * z)
            

            s = (1 - self.mask[i]) * self.s[i](
                torch.cat((z_, c_), -1))
            s = torch.clamp(s, min=-5, max=5)
            t = (1 - self.mask[i]) * self.t[i](
                torch.cat((z_, c_), -1))
            t = torch.clamp(t, min=-5, max=5)
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=-1)

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
        # miu, sigma = self.NN_z(torch.from_numpy(c).permute(0, 3, 2, 1).to(self.device))
        miu = torch.full(z.shape, 0)
        sigma = torch.full(z.shape, 1)
        L = 0.5*torch.log(torch.tensor([2*math.pi], device=self.device))+torch.log(sigma)+torch.div(torch.mul((z-miu),(z-miu)),2*torch.mul(sigma,sigma))
        L = L[:,:,:,0:1]+L[:,:,:,1:2]

        return -L.squeeze() + log_p , activation

    def eval(self, c):
        torch.manual_seed(1)
        z = self.prior.sample((c.shape[0], c.shape[2], c.shape[3])).to(self.device)
        z = z.permute(0, 3, 1, 2)
        z = torch.squeeze(z)


        c = torch.from_numpy(c).to(self.device)
        # log_p = self.prior.log_prob(z, c)
        # miu, sigma = self.NN_z(c)
        miu = torch.full(z.shape, 0)
        sigma = torch.full(z.shape, 1)
        z_cali = z * sigma + miu
        x = self.g(z_cali, c)
        ## hard code for Ngsim normalization
        # x = x * torch.from_numpy(self.std[:2, ]).to(self.device) + torch.from_numpy(self.mean[:2, ]).to(self.device)
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
        miu = torch.full(z.shape, 0).to(self.device)
        sigma = torch.full(z.shape, 1).to(self.device)
        z_cali = z*sigma + miu

        # log_p = self.prior.log_prob(z, c)
        x = self.g(z_cali, c)
        ## hard code for Ngsim normalization
        # x = x*torch.from_numpy(self.std[:2,]).to(self.device)+torch.from_numpy(self.mean[:2,]).to(self.device)
        if len(c.shape)>2:
            return x.permute(0, 3, 2, 1)
        else:
            return x

    def training_gan(self, y, c, writer, epoch, train=None):

        c = torch.from_numpy(c).to(self.device)
        y = torch.from_numpy(y).to(self.device)
        if train:
            n_D = 0
            while True:
                n_D += 1
                self.optimizer.zero_grad()
                fake_y = self.test(c)
                real_y = y
                loss_g_mse = torch.square(fake_y[:, :, :, [0, 8, 12, 15]]-
                                          real_y[:, :, :, [0, 8, 12, 15]]).mean()
                # T_real = self.model_D(torch.cat((real_y, c), 1))
                # T_fake = self.model_D(torch.cat((fake_y[0], fake_y[1], c), 1))
                T_real = self.model_D(real_y)
                T_fake = self.model_D(fake_y)
                # if (T_fake.mean() > 0.1 or n_D > 5) and (epoch < self.switch_epoch):
                #     break
                if (T_fake.mean() > 0.1 or n_D > 5) and (epoch < self.switch_epoch):
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
            real_y = y
            loss_g_mse = torch.square(fake_y[:, :, :, [0, 8, 12, 15]] -
                                      real_y[:, :, :, [0, 8, 12, 15]]).mean()
        writer.add_scalar("loss/loss_g_mse", loss_g_mse.mean().cpu().detach().numpy(), epoch + 1)
        # return self.model_D(torch.cat((fake_y[0], fake_y[1], c), 1)).mean()
        return self.model_D(fake_y).mean() + loss_g_mse


