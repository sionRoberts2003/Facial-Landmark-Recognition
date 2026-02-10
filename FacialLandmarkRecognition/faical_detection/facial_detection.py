import torch
from torch import nn

from config import FacialDetConfig


class FacialDetection(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=FacialDetConfig.IN_CHANNELS,
                      out_channels=FacialDetConfig.K_CHANNELS[0],
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )

        self.establish_cnn()

        kernel_layers = len(FacialDetConfig.K_CHANNELS)
        final_channel_count = FacialDetConfig.K_CHANNELS[-1]
        img_width = FacialDetConfig.IMG_WIDTH
        img_height = FacialDetConfig.IMG_HEIGHT

        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=(final_channel_count * img_width * img_height) // (2 ** (kernel_layers * 2)),
                      out_features=FacialDetConfig.LINEAR_DIMS[0]),
            nn.ReLU()
        )

        self.establish_fcl()

    def establish_fcl(self):
        for i, linear_dim in enumerate(FacialDetConfig.LINEAR_DIMS[:-1]):
            self.fully_connected.append(
                nn.Linear(in_features=linear_dim,
                          out_features=FacialDetConfig.LINEAR_DIMS[i + 1])
            )

            self.fully_connected.append(nn.ReLU())

        self.fully_connected.append(
            nn.Linear(in_features=FacialDetConfig.LINEAR_DIMS[-1],
                      out_features=FacialDetConfig.OUTPUT_DIM)
        )
        self.fully_connected.append(nn.ReLU())

    def establish_cnn(self):
        for i, k_channel in enumerate(FacialDetConfig.K_CHANNELS[:-1]):
            self.cnn.append(
                nn.Conv2d(in_channels=k_channel,
                          out_channels=FacialDetConfig.K_CHANNELS[i + 1],
                          kernel_size=3,
                          padding=1),
            )

            self.cnn.append(nn.ReLU())
            self.cnn.append(
                nn.MaxPool2d(kernel_size=2,
                             stride=2)
            )

            self.cnn.append(
                nn.BatchNorm2d(FacialDetConfig.K_CHANNELS[i + 1])
            )

        self.cnn.append(nn.Flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        return self.fully_connected(x)
