import torch
import numpy as np

class Discriminator(torch.nn.Module):
    def __init__(self, loop_number,  mean, std):
        super(Discriminator, self).__init__()
        self.loop = loop_number
        self.mean = mean
        self.std = std
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
        if self.loop == 3:
            self.model_cnn = torch.nn.Sequential(
                torch.nn.Conv2d(2, 4, 3, (3, 1), 1, bias=False),
                torch.nn.Tanh(),
                # torch.nn.LeakyReLU(0.2, inplace=False),
                torch.nn.Conv2d(4, 8, 3, (3, 1), 1, bias=False),
                torch.nn.BatchNorm2d(8),
                torch.nn.Tanh(),
                # torch.nn.LeakyReLU(0.2, inplace=False),
                torch.nn.Conv2d(8, 12, 3, (3, 1), 1, bias=False),
                torch.nn.BatchNorm2d(12),
                # torch.nn.LeakyReLU(0.2, inplace=False),
                torch.nn.Tanh(),

                torch.nn.Flatten(),
                torch.nn.Linear(144, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 1),
            )
        elif self.loop == 4:
            self.model_cnn = torch.nn.Sequential(
                torch.nn.Conv2d(2, 4, 3, (4, 2), 2,bias=False),
                torch.nn.Tanh(),
                # torch.nn.LeakyReLU(0.2, inplace=False),
                torch.nn.Conv2d(4, 8, 3, (4, 2), 2,bias=False),
                torch.nn.BatchNorm2d(8),
                torch.nn.Tanh(),
                # torch.nn.LeakyReLU(0.2, inplace=False),
                torch.nn.Conv2d(8, 16, 3, (4, 2), 2,bias=False),
                torch.nn.BatchNorm2d(16),
                # torch.nn.LeakyReLU(0.2, inplace=False),
                torch.nn.Tanh(),

                torch.nn.Flatten(),
                torch.nn.Linear(144, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 1),
            )
        elif self.loop == 6:
            self.model_cnn = torch.nn.Sequential(
                torch.nn.Conv2d(2, 4, 3, (3, 2), 2, bias=False),
                torch.nn.Tanh(),
                # torch.nn.LeakyReLU(0.2, inplace=False),
                torch.nn.Conv2d(4, 8, 3, (3, 2), 2, bias=False),
                torch.nn.BatchNorm2d(8),
                torch.nn.Tanh(),
                # torch.nn.LeakyReLU(0.2, inplace=False),
                torch.nn.Conv2d(8, 16, 3, (3, 2), 2, bias=False),
                torch.nn.BatchNorm2d(16),
                # torch.nn.LeakyReLU(0.2, inplace=False),
                torch.nn.Tanh(),

                torch.nn.Flatten(),
                torch.nn.Linear(240, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 1),
            )
        elif self.loop == 10:
            self.model_cnn = torch.nn.Sequential(
                torch.nn.Conv2d(2, 4, 3, (3, 2), 1, bias=False),
                torch.nn.Tanh(),
                # torch.nn.LeakyReLU(0.2, inplace=False),
                torch.nn.Conv2d(4, 8, 3, (3, 2), 1, bias=False),
                torch.nn.BatchNorm2d(8),
                torch.nn.Tanh(),
                # torch.nn.LeakyReLU(0.2, inplace=False),
                torch.nn.Conv2d(8, 12, 3, (3, 2), 1, bias=False),
                torch.nn.BatchNorm2d(12),
                # torch.nn.LeakyReLU(0.2, inplace=False),
                torch.nn.Tanh(),
                torch.nn.Flatten(),
                torch.nn.Linear(96, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 1)
            )
        elif self.loop == 14:
            self.model_cnn = torch.nn.Sequential(
                torch.nn.Conv2d(2, 4, 3, (3, 2), 1, bias=False),
                torch.nn.Tanh(),
                # torch.nn.LeakyReLU(0.2, inplace=False),
                torch.nn.Conv2d(4, 8, 3, (3, 2), 1, bias=False),
                torch.nn.BatchNorm2d(8),
                torch.nn.Tanh(),
                # torch.nn.LeakyReLU(0.2, inplace=False),
                torch.nn.Conv2d(8, 12, 3, (3, 2), 1, bias=False),
                torch.nn.BatchNorm2d(12),
                # torch.nn.LeakyReLU(0.2, inplace=False),
                torch.nn.Tanh(),

                torch.nn.Flatten(),
                torch.nn.Linear(96, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 1)
            )
        elif self.loop == 18:
            self.model_cnn = torch.nn.Sequential(
                torch.nn.Conv2d(2, 4, 3, (3, 2), 1, bias=False),
                torch.nn.Tanh(),
                # torch.nn.LeakyReLU(0.2, inplace=False),
                torch.nn.Conv2d(4, 8, 3, (3, 2), 1, bias=False),
                torch.nn.BatchNorm2d(8),
                torch.nn.Tanh(),
                # torch.nn.LeakyReLU(0.2, inplace=False),
                torch.nn.Conv2d(8, 12, 3, (3, 2), 1, bias=False),
                torch.nn.BatchNorm2d(12),
                # torch.nn.LeakyReLU(0.2, inplace=False),
                torch.nn.Tanh(),

                torch.nn.Flatten(),
                torch.nn.Linear(144, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 1)
            )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, image):
        image[:, 0, :, :] = (image[:, 0, :, :] - self.mean[0])/ self.std[0]
        image[:, 1, :, :] = (image[:, 1, :, :] - self.mean[1]) / self.std[1]
        p = self.model_cnn(image)

        return p
