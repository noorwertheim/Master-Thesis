import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
print(sys.executable)
sys.path.insert(1, '../src/')
from config import raw_data_path, univariate_data_path, processed_data_path, models_path
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import matplotlib.pyplot as plt

class ConvNeXtBlock1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Modify groups dynamically to match the input channels
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # Depthwise conv
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # Pointwise conv1 (expand)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)  # Pointwise conv2 (project)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)  # (B, C, L)
        x = x.permute(0, 2, 1)  # (B, L, C) for LayerNorm
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x + shortcut.permute(0, 2, 1)
        return x.permute(0, 2, 1)  # (B, C, L)

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class FCMAE(nn.Module):
    def __init__(self, in_channels=1, base_dim=64, num_blocks=3, kernel_size=7):
        super(FCMAE, self).__init__()

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        # Encoder blocks: reduce sequence length
        current_channels = in_channels
        for i in range(num_blocks):
            out_channels = base_dim * (2 ** i)
            self.encoder.append(nn.Conv1d(current_channels, out_channels, kernel_size=kernel_size, stride=2, padding=kernel_size // 2))
            self.encoder.append(nn.BatchNorm1d(out_channels))
            self.encoder.append(nn.ReLU())
            current_channels = out_channels

        # Decoder blocks: upsample back to original length
        for i in reversed(range(num_blocks)):
            out_channels = base_dim * (2 ** i)
            self.decoder.append(nn.ConvTranspose1d(current_channels, out_channels, kernel_size=kernel_size, stride=2, padding=kernel_size // 2, output_padding=1))
            self.decoder.append(nn.BatchNorm1d(out_channels))
            self.decoder.append(nn.ReLU())
            current_channels = out_channels

        # Final layer to return to original number of input channels
        self.output_layer = nn.Conv1d(current_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output_layer(x)
        return x