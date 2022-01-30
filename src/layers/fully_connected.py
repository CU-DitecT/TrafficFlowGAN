import numpy as np
from torch import nn
import torch


class Multiply(nn.Module):
    def __init__(self, scale):
        super(Multiply, self).__init__()
        self.scale = scale

    def forward(self, tensors):
        return self.scale * tensors


class Normalization(nn.Module):
    def __init__(self, mean, std, device, NNz):
        super(Normalization, self).__init__()
        self.device = device
        self.NNz = NNz
        self.mean = torch.from_numpy(mean).to(self.device)
        self.std = torch.from_numpy(std).to(self.device)


    def forward(self, tensors):
        if self.NNz:
            norm_tensor = (tensors - self.mean[2:4]) / self.std[2:4]
        else:
            norm_tensor = (tensors - torch.cat((torch.tensor([0,0]).to(self.device), self.mean[2:4]),dim=0)) / torch.cat((torch.tensor([1,1]).to(self.device), self.std[2:4]),dim=0)
        return norm_tensor

def instantiate_activation_function(function_name):
    function_dict = {
        "leaky_relu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "none": None
    }
    return function_dict[function_name]


def get_fully_connected_layer(input_dim, output_dim, n_hidden, hidden_dim,
                              activation_type="leaky_relu",
                              last_activation_type="tanh",
                              device=None,
                              mean = 0,
                              std = 1,
                              NNz= False):
    ## hard code without normalization
    # modules = [ Normalization(mean, std, device,NNz), nn.Linear(input_dim, hidden_dim, device=device)]
    modules = [nn.Linear(input_dim, hidden_dim, device=device)]
    activation = instantiate_activation_function(activation_type)
    if activation is not None:
        modules.append(activation)

    # add hidden layers
    if n_hidden > 1:
        for l in range(n_hidden-1):
            modules.append(nn.Linear(hidden_dim, hidden_dim, device=device))
            activation = instantiate_activation_function(activation_type)
            if activation is not None:
                modules.append(activation)

    # add the last layer

    modules.append(nn.Linear(hidden_dim, output_dim, device=device))
    last_activation = instantiate_activation_function(last_activation_type)

    modules.append(Multiply(0.5))
    if last_activation_type == "none":
        pass
    else:
        modules.append(last_activation)
    modules.append(Multiply(2))
    # the "mulltiply 2" is to stretch the range of the activation funtion (e.g. sigmoid) for 2 times long
    # the "multiply 0.5" is the stretch the x-axis accordingly.



    return nn.Sequential(*modules)
