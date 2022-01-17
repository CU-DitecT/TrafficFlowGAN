import torch
import time

class GaussianLWR(torch.nn.Module):
    def __init__(self, meta_params_value, meta_params_trainable, lower_bounds, upper_bounds, hypers,
                 train = False):
        super(GaussianLWR, self).__init__()

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
        self.meta_params_trainable = meta_params_trainable

    def get_residuals(self, model, x_unlabel):
        # # clip the parameter

        # for k in self.torch_meta_params.keys():
        #     self.torch_meta_params[k] = self.torch_meta_params[k].clamp(self.lower_bounds[k], self.upper_bounds[k])

        # for k in self.torch_meta_params.keys():
        #     if self.meta_params_trainable[k] == "True":
        #         self.torch_meta_params[k].requires_grad = True


        # get gradient
        batch_size = x_unlabel.shape[0]
        x = torch.tensor(x_unlabel[:, 0:1], requires_grad=True).float().to(model.device).repeat(self.hypers["n_repeat"],1)
        t = torch.tensor(x_unlabel[:, 1:2], requires_grad=True).float().to(model.device).repeat(self.hypers["n_repeat"],1)
        rho, u = model.test(torch.cat((x, t), 1))


        # get the derivative
        start_time = time.time()
        drho_dt = torch.autograd.grad(rho, t, torch.ones([t.shape[0], 1]).to(model.device),
                                      retain_graph=True, create_graph=True)[0]
        drho_dt_time = time.time() - start_time
        print(f"drho_dt time: {drho_dt_time:.5f}")

        drho_dx = torch.autograd.grad(rho, x, torch.ones([x.shape[0], 1]).to(model.device),
                                      retain_graph=True, create_graph=True)[0]
        drho_dx_time = time.time() - start_time - drho_dt_time
        print(f"drho_dx time: {drho_dx_time:.5f}")

        drho_dxx_time = time.time() - start_time - drho_dx_time - drho_dt_time
        drho_dxx = torch.autograd.grad(drho_dx, x, torch.ones([x.shape[0], 1]).to(model.device),
                                      retain_graph=True, create_graph=True)[0]
        print(f"drho_dxx time: {drho_dxx_time:.5f}")

        # get params
        torch_params = self.sample_params(self.torch_meta_params, batch_size)

        # eq_1 is: \rho * U_{eq}(\rho), where U_eq{\rho} = u_{max}(1 - \rho/\rho_{max})
        eq_1 = torch_params["umax"] * rho - 1/torch_params["rhomax-inv"] * torch_params["umax"] * rho * rho
        eq_2 = torch.autograd.grad(eq_1, x, torch.ones([eq_1.shape[0], eq_1.shape[1]]).to(model.device),
                                      retain_graph=True, create_graph=True)[0]

        r = drho_dt + eq_2 - torch_params["tau"] * drho_dxx
        #r = r.reshape(-1, self.hypers["n_repeat"])
        r = r.reshape(self.hypers["n_repeat"], -1).T

        r_mean = torch.square(torch.mean(r, dim=1))
        #self.update_parameter(drho_dt,drho_dx,drho_dxx,rho)
        gradient_hist = {"rho": rho,
                         "rho_dot_drhodx": rho * drho_dx,
                         "drho_dt": drho_dt,
                         "drho_dx": drho_dx,
                         "dq_dx": eq_2,
                         "drho_dxdx": drho_dxx,
                         "f1": r}

        return r_mean, torch_params, gradient_hist

    def sample_params(self, torch_meta_params, batch_size):
        meta_pairs = [("mu_rhomax-inv", "sigma_rhomax-inv"),
                      ("mu_umax", "sigma_umax"),
                      ("mu_tau", "sigma_tau")]

        n_repeat = self.hypers["n_repeat"]
        torch_params = dict()
        for mu_key, sigma_key in meta_pairs:
            param_key = mu_key.split("_")[1]
            z = self.randn.sample(sample_shape=(1, n_repeat))[0]
            z = torch.repeat_interleave(z, batch_size, dim=0)
            torch_params[param_key] = torch_meta_params[mu_key] # + torch_meta_params[sigma_key] * z
            if self.meta_params_trainable[mu_key] == "True":
                torch_params[param_key].retain_grad()
            # torch_params[param_key] = torch.clamp(torch_params[param_key], self.lower_bounds[mu_key],
            #                                         self.upper_bounds[mu_key])
        return torch_params

    def update_parameter(self, drho_dt, drho_dx, drho_dxdx, rho):
        alpha = 0.01
        rhomax = self.torch_meta_params["mu_rhomax-inv"]
        umax = self.torch_meta_params["mu_umax"]
        tau = self.torch_meta_params["mu_tau"]
        q_x = drho_dx*umax - 2*rho*drho_dx*umax/rhomax
        C = -tau*drho_dxdx
        dphydum = 2*(drho_dt+q_x+C)*(drho_dx-2*rho*drho_dx/rhomax)
        #dphydum = dphydum.reshape(self.hypers["n_repeat"], -1).T
        #dphydum = torch.square(torch.mean(dphydum, dim=1))
        dphydum = dphydum.mean()
        dphydrhom = 2*(drho_dt+q_x+C)*(2*rho*drho_dx*umax/(rhomax**2))
        dphydrhom = dphydrhom.mean()
        #self.torch_meta_params["mu_rhomax-inv"].grad = dphydrhom
        #self.torch_meta_params["mu_umax"].grad = dphydum
        print("dphydum:", dphydum, "dphydrhom:", dphydrhom)
        with torch.no_grad():
            self.torch_meta_params["mu_rhomax-inv"] -= alpha * dphydrhom
            self.torch_meta_params["mu_umax"] -= alpha * dphydum






