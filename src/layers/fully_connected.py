from torch import nn
import torch


class Multiply(nn.Module):
    def __init__(self, scale):
        super(Multiply, self).__init__()
        self.scale = scale

    def forward(self, tensors):
        return self.scale * tensors


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
                              last_activation_type="tanh"):
    modules = [nn.Linear(input_dim, hidden_dim)]
    activation = instantiate_activation_function(activation_type)
    if activation is not None:
        modules.append(activation)

    # add hidden layers
    if n_hidden > 1:
        for l in range(n_hidden-1):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            activation = instantiate_activation_function(activation_type)
            if activation is not None:
                modules.append(activation)

    # add the last layer

    modules.append(nn.Linear(hidden_dim, output_dim))
    last_activation = instantiate_activation_function(last_activation_type)

    modules.append(Multiply(0.5))
    if last_activation_type == "none":
        pass
    else:
        modules.append(last_activation)
    modules.append(Multiply(2))



    return nn.Sequential(*modules)
