import torch
import torch.nn as nn

from utils.const import *


class Discriminator(torch.nn.Module):
    def __init__(self, sigmoid: bool):
        super(Discriminator, self).__init__()
        if sigmoid:
            self.main_module = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=NF, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(in_channels=NF, out_channels=NF * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_features=NF * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(in_channels=NF * 2, out_channels=NF * 4, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_features=NF * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(in_channels=NF * 4, out_channels=NF * 8, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_features=NF * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=NF * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
                nn.Sigmoid()
            )
        else:
            self.main_module = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=NF, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(in_channels=NF, out_channels=NF * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_features=NF * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(in_channels=NF * 2, out_channels=NF * 4, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_features=NF * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(in_channels=NF * 4, out_channels=NF * 8, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_features=NF * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=NF * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False))

    def forward(self, x):
        return self.main_module(x)
