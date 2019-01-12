import torch.nn.functional as F
from torch import nn
import torch
import numpy as np


class HarmonicBlock(nn.Module):
    def __init__(self, input_channels, out_ch,
                 kernel_size=4, lmbda=3, pad=0, stride=1):
        super(HarmonicBlock, self).__init__()
        """
    :param input_channels: number of channels in the input
    :param kernel_size: size of the kernel in the filter bank
    :param pad: padding size
    :param stride: stride size
    :param lmbda: number of filters to be actually used (not implemented)
    """
        self.input_channels = input_channels
        self.out_ch = out_ch
        self.pad = pad
        self.stride = stride
        self.K = kernel_size
        # preferably to have N=K !! (to fully replicate the paper),
        # this is the convolution window size
        self.N = self.K
        self.PI = torch.as_tensor(np.pi)
        self.lmbda = lmbda  # limits the number of kernels
        self.conv = nn.Conv2d(in_channels=self.K**2, out_channels=self.out_ch,
                              kernel_size=1,
                              padding=self.pad,
                              stride=1)
        # output 1 because compresses into 1?
        # (see formula 2)
        self.get_filter_bank()

    def fltr(self, u, v, N, k):
        return torch.as_tensor([[torch.cos(torch.as_tensor(
                                v*self.PI/N*(ii+0.5)))
                                * torch.cos(
                                torch.as_tensor(u*self.PI/N*(jj+0.5)))
                                for ii in range(k)] for jj in range(k)])

    def get_filter_bank(self):
        self.filter_bank = torch.stack([torch.stack(
            [self.fltr(j, i, self.N, self.K)
             for i in range(self.K)]) for j in range(self.K)])
        self.filter_bank = self.filter_bank.reshape([-1, 1, self.K, self.K])
        self.filter_bank = torch.cat(
            [self.filter_bank]*self.input_channels, dim=1)
        # without this, it does not get sent to cuda
        self.filter_bank = self.filter_bank.to('cuda')

    def forward(self, x):

        # convolve all filters from the bank with the input
        # self.get_filter_bank()
        x = F.conv2d(x, weight=self.filter_bank,
                     padding=int(self.K/2), stride=self.stride)
        x = self.conv(x)
        return x
