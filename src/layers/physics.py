import torch
import math
import numpy as np
import time

class GaussianLWR(torch.nn.Module):
    def __init__(self, meta_params_value, meta_params_trainable, lower_bounds, upper_bounds, hypers,
                 train = False,
                 device=None):
        super(GaussianLWR, self).__init__()

        self.torch_meta_params = dict()
        self.meta_params_trainable =meta_params_trainable
        for k, v in meta_params_value.items():
            if meta_params_trainable[k] == "True":
                self.torch_meta_params[k] = torch.nn.Parameter(torch.tensor(v, dtype=torch.float32, device=device), requires_grad=True,
                                                               )
                self.torch_meta_params[k].retain_grad()
            else:
                self.torch_meta_params[k] = torch.nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=False,
                                                               )

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.randn = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.hypers = hypers
        self.train = train

    def caculate_residual(self, rho, x, t, Umax, RHOmax, Tau, model):
        drho_dt = torch.autograd.grad(rho, t, torch.ones([t.shape[0], 1]).to(model.device),
                                      retain_graph=True, create_graph=True)[0]
        # print(f"one autograd time: {time.time() - start_time:.5f}")
        drho_dx = torch.autograd.grad(rho, x, torch.ones([x.shape[0], 1]).to(model.device),
                                      retain_graph=True, create_graph=True)[0]
        drho_dxx = torch.autograd.grad(drho_dx, x, torch.ones([x.shape[0], 1]).to(model.device),
                                       retain_graph=True, create_graph=True)[0]

        # get params


        # eq_1 is: \rho * U_{eq}(\rho), where U_eq{\rho} = u_{max}(1 - \rho/\rho_{max})
        eq_1 = Umax * rho - 1 / RHOmax * Umax * rho * rho
        eq_2 = torch.autograd.grad(eq_1, x, torch.ones([eq_1.shape[0], eq_1.shape[1]]).to(model.device),
                                   retain_graph=True, create_graph=True)[0]

        r = drho_dt + eq_2 - Tau * drho_dxx
        # r = r.reshape(-1, self.hypers["n_repeat"])
        r = r.reshape(self.hypers["n_repeat"], -1).T

        r_mean = torch.square(torch.mean(r, dim=1))

        return r_mean, drho_dx, drho_dt,eq_2,r

    def get_residuals(self, model, x_unlabel):
        # get gradient
        batch_size = x_unlabel.shape[0]
        x = torch.tensor(x_unlabel[:, 0:1], requires_grad=True).float().to(model.device).repeat(self.hypers["n_repeat"],1)
        t = torch.tensor(x_unlabel[:, 1:2], requires_grad=True).float().to(model.device).repeat(self.hypers["n_repeat"],1)
        rho, u = model.test(torch.cat((x, t), 1))
        torch_params = self.sample_params(self.torch_meta_params, batch_size)

        r_mean, drho_dx, drho_dt, eq_2,r = self.caculate_residual(rho, x, t, torch_params["umax"], torch_params["rhomax"], torch_params["tau"],model)

        # Umax_line = np.linspace(0.1,2,50)
        # rhomax_line = np.linspace(0.1,2,50)
        # r_out = np.zeros((Umax_line.shape[0],rhomax_line.shape[0]))
        # for i in range(Umax_line.shape[0]):
        #     print('i:',i)
        #     for j in range(rhomax_line.shape[0]):
        #         r_mean, drho_dx, drho_dt, eq_2, r = self.caculate_residual(rho, x, t, torch.from_numpy(np.array(Umax_line[i])),
        #                                                                    torch.from_numpy(np.array(rhomax_line[j])), torch.tensor([0.005],dtype=torch.float32),
        #                                                                    model)
        #         r_out[i][j] = r_mean.mean().cpu().detach().numpy()
        # with open('/home/ubuntu/PhysFlow/test_21loop_100000.npy','wb') as f:
        #     np.save(f,r_out)


        gradient_hist = {"rho": rho.cpu().detach().numpy(),
                         "rho_dot_drhodx": (rho * drho_dx).cpu().detach().numpy(),
                         "drho_dt": drho_dt.cpu().detach().numpy(),
                         "drho_dx": drho_dx.cpu().detach().numpy(),
                         "dq_dx": eq_2.cpu().detach().numpy(),
                         "f1": r.cpu().detach().numpy()}

        for k in torch_params.keys():
            torch_params[k] = torch_params[k].cpu().detach().numpy()

        return r_mean, torch_params, gradient_hist

    def sample_params(self, torch_meta_params, batch_size):
        meta_pairs = [("mu_rhomax", "sigma_rhomax"),
                      ("mu_umax", "sigma_umax"),
                      ("mu_tau", "sigma_tau")]

        n_repeat = self.hypers["n_repeat"]
        torch_params = dict()
        for mu_key, sigma_key in meta_pairs:
            param_key = mu_key.split("_")[1]
            z = self.randn.sample(sample_shape=(1, n_repeat))[0]
            z = torch.repeat_interleave(z, batch_size, dim=0)
            torch_params[param_key] = torch_meta_params[mu_key] # + torch_meta_params[sigma_key] * z
            torch_params[param_key].retain_grad()
            #torch_params[param_key] = torch.clamp(torch_params[param_key], self.lower_bounds[mu_key],
            #                                        self.upper_bounds[mu_key])
        return torch_params

class GaussianARZ(torch.nn.Module):
    def __init__(self, meta_params_value, meta_params_trainable, lower_bounds, upper_bounds, hypers,
                 train = False,
                 device=None):
        super(GaussianARZ, self).__init__()
        self.meta_params_trainable = meta_params_trainable
        self.torch_meta_params = torch.nn.ParameterDict()
        for k, v in meta_params_value.items():
            if meta_params_trainable[k] == "True":
                self.torch_meta_params[k] = torch.nn.Parameter(torch.tensor(v, dtype=torch.float32, device=device), requires_grad=True,
                                                               )
                self.torch_meta_params[k].retain_grad()
            else:
                self.torch_meta_params[k] = torch.nn.Parameter(torch.tensor(v, dtype=torch.float32, device=device), requires_grad=False,
                                                               )

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.randn = torch.distributions.normal.Normal(torch.tensor([0.0],device=device), torch.tensor([1.0], device=device))
        self.hypers = hypers
        self.train = train
        self.device=device

    def caculate_residual(self, rho, u,  x, t, Umax, RHOmax, Tau, model):
        Tau=Tau/50.0

        drho_dt = torch.autograd.grad(rho, t, torch.ones([t.shape[0], 1], device=self.device).to(model.device),
                                      retain_graph=True, create_graph=True)[0]
        # print(f"one autograd time: {time.time() - start_time:.5f}")
        drho_dx = torch.autograd.grad(rho, x, torch.ones([x.shape[0], 1], device=self.device).to(model.device),
                                      retain_graph=True, create_graph=True)[0]
        du_dx = torch.autograd.grad(u, x, torch.ones([x.shape[0], 1]).to(model.device),
                                      retain_graph=True, create_graph=True)[0]
        # drho_dxx = torch.autograd.grad(drho_dx, x, torch.ones([x.shape[0], 1]).to(model.device),
        #                                retain_graph=True, create_graph=True)[0]

        U_eq = Umax*(1 - rho/RHOmax)
        h = Umax*rho/RHOmax

        ## f_rho
        drho_time_u_dx =drho_dx *u + du_dx*rho
        f_rho = drho_dt + drho_time_u_dx # - 0.005*drho_dxx

        ## f_u
        u_h = u+h
        du_h_dt = torch.autograd.grad(u_h, t, torch.ones([x.shape[0], 1]).to(model.device),
                                      retain_graph=True, create_graph=True)[0]
        du_h_dx = torch.autograd.grad(u_h, x, torch.ones([x.shape[0], 1]).to(model.device),
                                      retain_graph=True, create_graph=True)[0]
        f_u = Tau*(du_h_dt+ u*du_h_dx) -(U_eq-u)

        f_rho = f_rho.reshape(self.hypers["n_repeat"], -1).T

        f_rho_mean = torch.square(torch.mean(f_rho, dim=1))

        f_u = f_u.reshape(self.hypers["n_repeat"], -1).T

        f_u_mean = torch.square(torch.mean(f_u, dim=1))


        return f_rho_mean, f_u_mean, drho_dt

    def get_residuals(self, model, x_unlabel):
        # get gradient
        batch_size = x_unlabel.shape[0]
        x = torch.tensor(x_unlabel[:, 0:1], requires_grad=True, device=self.device).float().to(model.device).repeat(self.hypers["n_repeat"],1)
        t = torch.tensor(x_unlabel[:, 1:2], requires_grad=True, device=self.device).float().to(model.device).repeat(self.hypers["n_repeat"],1)
        rho, u = model.test(torch.cat((x, t), 1))
        torch_params = self.sample_params(self.torch_meta_params, batch_size)

        f_rho_mean, f_u_mean, drho_dt = self.caculate_residual(rho, u, x, t, torch_params["umax"], torch_params["rhomax"], torch_params["tau"],model)

        # Umax_line = np.linspace(0.1,2,50)
        # rhomax_line = np.linspace(0.1,2,50)
        # r_out = np.zeros((Umax_line.shape[0],rhomax_line.shape[0]))
        # for i in range(Umax_line.shape[0]):
        #     print('i:',i)
        #     for j in range(rhomax_line.shape[0]):
        #         r_mean, drho_dx, drho_dt, eq_2, r = self.caculate_residual(rho, x, t, torch.from_numpy(np.array(Umax_line[i])),
        #                                                                    torch.from_numpy(np.array(rhomax_line[j])), torch.tensor([0.005],dtype=torch.float32),
        #                                                                    model)
        #         r_out[i][j] = r_mean.mean().cpu().detach().numpy()
        # with open('/home/ubuntu/PhysFlow/test_21loop_100000.npy','wb') as f:
        #     np.save(f,r_out)
        loss_mean = self.hypers["alpha_u_rho"]*f_rho_mean +(1-self.hypers["alpha_u_rho"])*f_u_mean
        # loss_mean = f_rho_mean
        gradient_hist = {"rho": rho,
                         "drho_dt": drho_dt,
                         "f_rho_mean": f_rho_mean,
                         "f_u_mean": f_u_mean}

        return loss_mean, torch_params, gradient_hist

    def sample_params(self, torch_meta_params, batch_size):
        meta_pairs = [("mu_rhomax", "sigma_rhomax"),
                      ("mu_umax", "sigma_umax"),
                      ("mu_tau", "sigma_tau")]

        n_repeat = self.hypers["n_repeat"]
        torch_params = dict()
        for mu_key, sigma_key in meta_pairs:
            param_key = mu_key.split("_")[1]
            z = self.randn.sample(sample_shape=(1, n_repeat))[0]
            z = torch.repeat_interleave(z, batch_size, dim=0)
            torch_params[param_key] = torch_meta_params[mu_key] # + torch_meta_params[sigma_key] * z
            torch_params[param_key].retain_grad()
            #torch_params[param_key] = torch.clamp(torch_params[param_key], self.lower_bounds[mu_key],
            #                                        self.upper_bounds[mu_key])
        return torch_params


class GaussianBurgers(torch.nn.Module):
    def __init__(self, meta_params_value, meta_params_trainable, lower_bounds, upper_bounds, hypers,
                 train = False):
        super(GaussianBurgers, self).__init__()
        self.meta_params_trainable = meta_params_trainable
        self.torch_meta_params = dict()
        for k, v in meta_params_value.items():
            if meta_params_trainable[k] == "True":
                self.torch_meta_params[k] = torch.nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=True,
                                                               )
                self.torch_meta_params[k].retain_grad()
            else:
                self.torch_meta_params[k] = torch.nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=False,
                                                               )

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.randn = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.hypers = hypers
        self.train = train

    def caculate_residual(self, u, x, t, nu, model):
        nu = nu/100.0/math.pi
        u_t = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u).to(model.device),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u).to(model.device),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(u_x).to(model.device),
            retain_graph=True,
            create_graph=True
        )[0]

        f = u_t + u * u_x - nu * u_xx
        # r = r.reshape(-1, self.hypers["n_repeat"])
        f = f.reshape(self.hypers["n_repeat"], -1).T

        f_mean = torch.square(torch.mean(f, dim=1))

        return f_mean, u_t, u_x, u_xx, f

    def get_residuals(self, model, x_unlabel):
        # get gradient
        batch_size = x_unlabel.shape[0]
        x = torch.tensor(x_unlabel[:, 0:1], requires_grad=True).float().to(model.device).repeat(self.hypers["n_repeat"],1)
        t = torch.tensor(x_unlabel[:, 1:2], requires_grad=True).float().to(model.device).repeat(self.hypers["n_repeat"],1)
        u, _ = model.test(torch.cat((x, t), 1))
        torch_params = self.sample_params(self.torch_meta_params, batch_size)

        f_mean, u_t, u_x, u_xx, f = self.caculate_residual(u, x, t, torch_params["nu"],model)

        # Umax_line = np.linspace(0.1,2,50)
        # rhomax_line = np.linspace(0.1,2,50)
        # r_out = np.zeros((Umax_line.shape[0],rhomax_line.shape[0]))
        # for i in range(Umax_line.shape[0]):
        #     print('i:',i)
        #     for j in range(rhomax_line.shape[0]):
        #         r_mean, drho_dx, drho_dt, eq_2, r = self.caculate_residual(rho, x, t, torch.from_numpy(np.array(Umax_line[i])),
        #                                                                    torch.from_numpy(np.array(rhomax_line[j])), torch.tensor([0.005],dtype=torch.float32),
        #                                                                    model)
        #         r_out[i][j] = r_mean.mean().cpu().detach().numpy()
        # with open('/home/ubuntu/PhysFlow/test_21loop_100000.npy','wb') as f:
        #     np.save(f,r_out)


        gradient_hist = {"u": u.cpu().detach().numpy(),
                         "u_t": u_t.cpu().detach().numpy(),
                         "u_x": u_x.cpu().detach().numpy(),
                         "u_xx": u_xx.cpu().detach().numpy(),
                         "f_mean": f_mean.cpu().detach().numpy()}

        for k in torch_params.keys():
            torch_params[k] = torch_params[k].cpu().detach().numpy()

        return f_mean, torch_params, gradient_hist

    def sample_params(self, torch_meta_params, batch_size):
        meta_pairs = [("mu_nu", "sigma_nu")]

        n_repeat = self.hypers["n_repeat"]
        torch_params = dict()
        for mu_key, sigma_key in meta_pairs:
            param_key = mu_key.split("_")[1]
            z = self.randn.sample(sample_shape=(1, n_repeat))[0]
            z = torch.repeat_interleave(z, batch_size, dim=0)
            torch_params[param_key] = torch_meta_params[mu_key] # + torch_meta_params[sigma_key] * z
            torch_params[param_key].retain_grad()
            #torch_params[param_key] = torch.clamp(torch_params[param_key], self.lower_bounds[mu_key],
            #                                        self.upper_bounds[mu_key])
        return torch_params
