import torch
import time

class GaussianLWR(torch.nn.Module):
    def __init__(self, meta_params_value, meta_params_trainable, lower_bounds, upper_bounds, hypers,
                 train = False):
        super(GaussianLWR, self).__init__()

        self.torch_meta_params = dict()
        for k, v in meta_params_value.items():
            self.torch_meta_params[k] = torch.nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=(meta_params_trainable[k] == "True"))

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.randn = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.hypers = hypers
        self.train = train

    def get_residuals(self, model, x_unlabel):
        # get gradient
        batch_size = x_unlabel.shape[0]
        x = torch.tensor(x_unlabel[:, 0:1], requires_grad=True).float().to(model.device).repeat(self.hypers["n_repeat"],1)
        t = torch.tensor(x_unlabel[:, 1:2], requires_grad=True).float().to(model.device).repeat(self.hypers["n_repeat"],1)
        rho, u = model.test(torch.cat((x, t), 1))


        # get the derivative
        start_time = time.time()
        drho_dt = torch.autograd.grad(rho, t, torch.ones([t.shape[0], 1]).to(model.device),
                                      retain_graph=True, create_graph=True)[0]
        print(f"one autograd time: {time.time() - start_time:.5f}")
        drho_dx = torch.autograd.grad(rho, x, torch.ones([x.shape[0], 1]).to(model.device),
                                      retain_graph=True, create_graph=True)[0]
        drho_dxx = torch.autograd.grad(drho_dx, x, torch.ones([x.shape[0], 1]).to(model.device),
                                      retain_graph=True, create_graph=True)[0]

        # get params
        torch_params = self.sample_params(self.torch_meta_params, batch_size)

        # eq_1 is: \rho * U_{eq}(\rho), where U_eq{\rho} = u_{max}(1 - \rho/\rho_{max})
        eq_1 = torch_params["umax"] * rho - torch_params["rhomax"] * torch_params["umax"] * rho * rho
        eq_2 = torch.autograd.grad(eq_1, x, torch.ones([eq_1.shape[0], eq_1.shape[1]]).to(model.device),
                                      retain_graph=True, create_graph=True)[0]

        r = drho_dt + eq_2 - torch_params["tau"] * drho_dxx
        r = r.reshape(-1, self.hypers["n_repeat"])

        r = torch.mean(r, dim=1)

        return r, torch_params

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
            torch_params[param_key] = torch_meta_params[mu_key] + \
                                      torch_meta_params[sigma_key] * z
            torch_params[param_key] = torch.clamp(torch_params[param_key], self.lower_bounds[mu_key],
                                                    self.upper_bounds[mu_key])
        return torch_params


