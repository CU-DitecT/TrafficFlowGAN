import torch
import numpy as np

class Discriminator(torch.nn.Module):
    def __init__(self, shape_input):
        super(Discriminator, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(int(np.prod(shape_input)),2048),
            torch.nn.LeakyReLU(0.2,inplace=False),
            torch.nn.Linear(2048, 1024),
            torch.nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Linear(1024, 512),
            torch.nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Linear(256,1),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, image):
        p = self.model(image)
        return p