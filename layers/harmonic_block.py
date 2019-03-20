import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
from matplotlib import pyplot as plt


class HarmonicBlock(nn.Module):
    def __init__(self, input_channels, output_ch, bn=True,
                 kernel_size=4, lmbda=None, diag=False, pad=0, stride=1):
        super(HarmonicBlock, self).__init__()
        """
        :param input_channels: number of channels in the input
        :param kernel_size: size of the kernel in the filter bank
        :param pad: padding size
        :param stride: stride size
        :param lmbda: number of filters to be actually used (feature not implemented)
        """
        self.bn = bn
        self.input_channels = input_channels
        self.output_ch = output_ch
        self.pad = pad
        self.stride = stride
        self.K = kernel_size
        self.diag = diag
        self.N = self.K  # preferably to have N=K !! (to fully replicate the paper), this is the convolution window size
        self.PI = torch.as_tensor(np.pi)
        if lmbda is not None:
            if lmbda > self.K ** 2:
                self.lmbda = self.K ** 2  # limits the number of kernels
            else:
                self.lmbda = lmbda
        else:
            self.lmbda = self.K ** 2
        self.diag = diag  # flag to select diagonal entries of the block
        self.filter_bank = self.get_filter_bank(N=self.N,  # ??? forgot what N is
                                                K=self.K,  # kernel size
                                                input_channels=self.input_channels,
                                                lmbda=self.lmbda,
                                                diag=self.diag).to('cuda')
        self.conv = nn.Conv2d(in_channels=self.filter_bank.shape[0], out_channels=self.output_ch,
                              kernel_size=1,
                              padding=0,
                              stride=1, bias=True)
        if self.bn:
            self.bnorm = nn.BatchNorm2d(self.filter_bank.shape[0])

    def fltr(self, u, v, N, k):
        return torch.as_tensor([[torch.cos(torch.as_tensor(self.PI / N * (ii + 0.5) * v)) * torch.cos(
            torch.as_tensor(self.PI / N * (jj + 0.5) * u)) for ii in range(k)] for jj in range(k)])

    '''def get_filter_bank(self):
      self.filter_bank = torch.stack([torch.stack([self.fltr(j, i, self.N, self.K) for i in range(self.K)]) for j in range(self.K)])
      self.filter_bank = self.filter_bank.reshape([1,-1,self.K,self.K])
      self.filter_bank = torch.cat([self.filter_bank]*self.input_channels, dim=0)
      self.filter_bank = torch.transpose(self.filter_bank,0,1)[:self.lmbda]
      self.filter_bank = self.filter_bank.to('cuda').to(torch.float32) # without this, it does not get sent to cuda
    '''

    def get_idx(self, K, l):
        out = []
        for i in range(K):
            for j in range(K):
                if i + j < l:
                    out.append(K * i + j)
        return tuple(out)

    def get_idx_diag(self, K):
        out = []
        for i in range(K):
            for j in range(K):
                if i == j:
                    out.append(i + j)
        return tuple(out)

    def draw_filters(self, fb_):
        fig, ax = plt.subplots(len(fb_), 1, figsize=(12, 4))
        j = 0
        for i in range(len(fb_)):
            ax[i].imshow(fb_[i, 0, :, :])
            ax[i].axis('off')
            ax[i].grid(False)

    def get_filter_bank(self, N, K, input_channels=3, lmbda=None, diag=False):
        filter_bank = torch.zeros((K, K, K, K))
        for i in range(K):
            for j in range(K):
                filter_bank[i, j, :, :] = self.fltr(i, j, N, K)
        if lmbda:
            ids = self.get_idx(K, lmbda)
            return torch.stack(tuple([filter_bank.view(-1, 1, K, K)] * input_channels), dim=1).view(
                (-1, input_channels, K, K))[ids, :, :, :]  # filter_bank.view(K**2,-1,K,K)
        if diag:
            ids = self.get_idx_diag(K)
            return torch.stack(tuple([filter_bank.view(-1, 1, K, K)] * input_channels), dim=1).view(
                (-1, input_channels, K, K))[ids, :, :, :]  # filter_bank.view(K**2,-1,K,K)[ids,:,:,:]
        return torch.stack(tuple([filter_bank.view(-1, 1, K, K)] * input_channels), dim=1).view(
            (-1, input_channels, K, K))

    def forward(self, x):
        x = F.conv2d(x.to(torch.float32), weight=self.filter_bank.to(torch.float32), padding=self.pad,
                     stride=self.stride)  # int(self.K/2)
        if self.bn:
            x = self.bnorm(x)
        x = self.conv(x)
        return x


'''
class HarmonicBlock(nn.Module):
    def __init__(self, input_channels, output_ch, bn=True,
                 kernel_size=4, lmbda=5, pad=0, stride=1):
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
        self.lmbda = min(self.K**2,lmbda) # limits the number of kernels
        self.conv = nn.Conv2d(in_channels=self.lmbda,
                              out_channels=self.output_ch,
                              kernel_size=1,
                              padding=0,
                              stride=1)
        self.get_filter_bank()
        if self.bn:
            self.bnorm = nn.BatchNorm2d(self.lmbda)

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
        self.filter_bank = torch.transpose(self.filter_bank, 0, 1)[:self.lmbda]
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
'''
