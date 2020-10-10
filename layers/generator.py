import torch
import torch.nn as nn

from utils.const import *


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main_module = nn.Sequential(

            nn.ConvTranspose2d(in_channels=Z_VECTOR_SIZE, out_channels=NF * 8, kernel_size=4, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(num_features=NF * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=NF * 8, out_channels=NF * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=NF * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=NF * 4, out_channels=NF * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=NF * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=NF * 2, out_channels=NF, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=NF),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=NF, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main_module(x)
