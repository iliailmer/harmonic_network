import torch.nn.functional as F
from torch import nn
import torch
import numpy as np


class HarmonicBlock(nn.Module):
    def __init__(self, input_channels, output_ch, bn=True,
                 kernel_size=4, lmbda=3, pad=0, stride=1):
        super(HarmonicBlock, self).__init__()

        self.bn = bn
        self.input_channels = input_channels
        self.output_ch = output_ch
        self.pad = pad
        self.stride = stride
        self.K = kernel_size
        # preferably to have N=K !! (to fully replicate the paper)
        # this is the convolution window size
        self.N = self.K
        self.PI = torch.as_tensor(np.pi)
        # self.lmbda = lmbda # limits the number of kernels
        self.conv = nn.Conv2d(in_channels=self.K**2,
                              out_channels=self.output_ch,
                              kernel_size=1,
                              padding=0,
                              stride=1)
        self.get_filter_bank()
        if self.bn:
            self.bnorm = nn.BatchNorm2d(self.K**2)

    def fltr(self, u, v, N, k):
        return torch.as_tensor([[
            torch.cos(
                torch.as_tensor(self.PI/N * (ii + 0.5) * v)) * torch.cos(
                torch.as_tensor(
                    self.PI/N*(jj+0.5)*u)) for ii in range(k)]
            for jj in range(k)])

    def get_filter_bank(self):
        self.filter_bank = torch.stack([torch.stack(
            [self.fltr(
                j,
                i,
                self.N,
                self.K) for i in range(self.K)]) for j in range(self.K)])
        self.filter_bank = self.filter_bank.reshape([1, -1, self.K, self.K])
        self.filter_bank = torch.cat(
            [self.filter_bank]*self.input_channels, dim=0)
        self.filter_bank = torch.transpose(self.filter_bank, 0, 1)
        self.filter_bank = self.filter_bank.to('cuda').to(
            torch.float32)  # without this, it does not get sent to cuda

    def forward(self, x):
        x = F.conv2d(x.to(torch.float32), weight=self.filter_bank.to(
            torch.float32), padding=self.pad, stride=self.stride)
        if self.bn:
            self.bnorm
            x = self.bnorm(x)
        x = self.conv(x)
        return x
