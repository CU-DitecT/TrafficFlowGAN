import torch
import numpy as np

class Discriminator(torch.nn.Module):
    def __init__(self, shape_input):
        super(Discriminator, self).__init__()

        # self.model = torch.nn.Sequential(
        #     torch.nn.Flatten(),
        #     torch.nn.Linear(int(np.prod(shape_input)),2048),
        #     #torch.nn.LeakyReLU(0.2, inplace=False),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(2048, 1024),
        #     # torch.nn.LeakyReLU(0.2, inplace=False),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(1024, 512),
        #     # torch.nn.LeakyReLU(0.2, inplace=False),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(512, 256),
        #     # torch.nn.LeakyReLU(0.2, inplace=False),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(256,1),
        # )

        self.model_cnn = torch.nn.Sequential(
            torch.nn.Conv2d(2, 4, 3, 2, 2,bias=False),
            torch.nn.Tanh(),
            #torch.nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Conv2d(4, 8, 3, 2, 2,bias=False),
            torch.nn.BatchNorm2d(8),
            torch.nn.Tanh(),
            # torch.nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Conv2d(8, 16, 3, 2, 2,bias=False),
            torch.nn.BatchNorm2d(16),
            # torch.nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Tanh(),
            torch.nn.Conv2d(16, 24, 3, 2, 2,bias=False),
            torch.nn.BatchNorm2d(24),
            # torch.nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Tanh(),
            torch.nn.Conv2d(24, 1, (8,3), 1, 0, bias=False),
        )

        # self.model_x_t = torch.nn.Sequential(
        #     torch.nn.Linear(4,32),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(32, 64),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(64, 128),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(128, 64),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(64, 32),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(32, 1),
        # )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, image):
        p = self.model_cnn(image)
        return p