import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import hadamard, block_diag


class BasicBlock(nn.Module):
    def __init__(self, input_channels, output_ch, stride, dropRate=0.0, **kwargs):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, output_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (input_channels == output_ch)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(input_channels, output_ch, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class HarmonicBlock(nn.Module):
    def __init__(self, input_channels, output_ch,
                 kernel_size=3,
                 pad=1,
                 stride=1,
                 t=2,
                 alpha_root=1.0,
                 lmbda=None,
                 diag=False,
                 bn=True,
                 dropout=False,
                 bias=False,
                 use_res=True):
        super(HarmonicBlock, self).__init__()
        """
        :param input_channels: number of channels in the input
        :param kernel_size: size of the kernel in the filter bank
        :param pad: padding size
        :param stride: stride size
        :param lmbda: number of filters to be actually used (feature not implemented)
        """
        self.bn = bn
        self.drop = dropout
        self.input_channels = input_channels
        self.output_ch = output_ch
        self.stride = stride
        self.K = kernel_size
        if kernel_size % 2 == 0:
            self.pad = kernel_size // 2
        else:
            self.pad = pad
        self.diag = diag
        self.N = self.K  # preferably to have N=K !! (to fully replicate the paper), this is the convolution window size
        self.PI = torch.as_tensor(np.pi)
        self.use_res = use_res
        self.alpha_root = alpha_root
        if lmbda is not None:
            if lmbda > self.K ** 2:
                self.lmbda = self.K ** 2  # limits the number of kernels
            else:
                self.lmbda = lmbda
        else:
            self.lmbda = lmbda
        self.diag = diag  # flag to select diagonal entries of the block
        self.filter_bank = self.get_filter_bank(N=self.N,
                                                K=self.K,  # kernel size
                                                input_channels=self.input_channels,
                                                t=t,  # type of DCT
                                                lmbda=self.lmbda,
                                                diag=self.diag).float().cuda()
        self.conv = nn.Conv2d(in_channels=self.filter_bank.shape[0], out_channels=self.output_ch,
                              kernel_size=1,
                              padding=0,
                              stride=1,
                              bias=bias)
        if (stride != 2 or input_channels != output_ch):
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels,
                          output_ch,
                          kernel_size=2 if self.K % 2 == 0 else 1,
                          stride=stride,
                          padding=1 if self.K % 2 == 0 else 0,
                          bias=False),
                nn.BatchNorm2d(output_ch)
            )
        else:
            self.shortcut = nn.Sequential()

        if self.bn:
            self.bnorm = nn.BatchNorm2d(self.filter_bank.shape[0])
        if self.drop:
            self.dropout = nn.Dropout(0.5)

    @staticmethod
    def dct_matrix(t=1, N=32):
        if t == 1:
            # N- the size of the input
            # n is the column dummy index, k is the row dummy index
            res = np.zeros((N, N))
            res[:, 0] = 0.5
            for p in range(N):
                res[p, -1] = (-1) ** p
            for n in range(1, N - 1):
                for k in range(N):
                    res[k, n] = np.cos(np.pi / (N - 1) * n * k)
            return res
        if t == 2:
            res = np.zeros((N, N))
            for k in range(N):
                for n in range(N):
                    res[k, n] = np.cos(np.pi / (N) * (n + 0.5) * k)
            return res
        if t == 3:
            res = np.zeros((N, N))
            res[:, 0] = 0.5
            for n in range(1, N):
                for k in range(N):
                    res[k, n] = np.cos(np.pi / (N) * n * (k + 0.5))
            return res
        if t == 4:
            res = np.zeros((N, N))
            for k in range(N):
                for n in range(N):
                    res[k, n] = np.cos(np.pi / (N) * (n + 0.5) * (k + 0.5))
            return res

    def filter_from_dct_matrix(self, i, j, size, t=2):
        mat = self.dct_matrix(N=size, t=t)
        fltr = mat[i, :].reshape((-1, 1)).dot(mat[j, :].reshape(1, -1))
        return torch.as_tensor(fltr)

    def fltr(self, u, v, N, k):
        return torch.as_tensor([[torch.cos(torch.as_tensor(self.PI / N * (ii + 0.5) * v)) * torch.cos(
            torch.as_tensor(self.PI / N * (jj + 0.5) * u)) for ii in range(k)] for jj in range(k)])

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

    def draw_filters(self, fb_=None):
        if fb_ is None: fb_ = self.filter_bank
        fig, ax = plt.subplots(len(fb_), 1, figsize=(12, 4))
        j = 0
        for i in range(len(fb_)):
            ax[i].imshow(fb_[i, 0, :, :])
            ax[i].axis('off')
            ax[i].grid(False)

    def get_filter_bank(self, N, K, input_channels=3, t=2, lmbda=None, diag=False):
        filter_bank = torch.zeros((K, K, K, K))
        for i in range(K):
            for j in range(K):
                filter_bank[i, j, :, :] = self.filter_from_dct_matrix(i, j, K, t)  # self.fltr(i, j, N, K)
        if lmbda is not None:
            ids = self.get_idx(K, lmbda)
            return torch.stack(tuple([filter_bank.view(-1, 1, K, K)[ids, :, :, :]] * input_channels), dim=0).view(
                (-1, 1, K, K))
        if diag:
            ids = self.get_idx_diag(K)
            return torch.stack(tuple([filter_bank.view(-1, 1, K, K)[ids, :, :, :]] * input_channels), dim=0).view(
                (-1, 1, K, K))
        return torch.stack(tuple([filter_bank.view(-1, 1, K, K)] * input_channels), dim=0).view(
            (-1, 1, K, K))

    def alpha_rooting(self, x, alpha=1.0):
        if alpha is not None:
            return x.sign() * torch.abs(x).pow(alpha)
        else:
            return x

    def forward(self, x):
        in_ = x
        x = F.conv2d(x.float(),
                     weight=self.filter_bank,
                     padding=self.pad,
                     stride=self.stride,
                     groups=self.input_channels)  # int(self.K/2)
        x = self.alpha_rooting(x, alpha=self.alpha_root)
        if self.bn:
            x = F.relu(self.bnorm(x))
        else:
            x = F.relu(x)
        if self.drop:
            x = self.dropout(x)
        if self.use_res:
            x = self.conv(x) + self.shortcut(in_)
            # x = self.alpha_rooting(x, alpha=self.alpha_root) + self.shortcut(in_)
            # x = self.conv(x) + self.shortcut(in_)
        else:
            x = self.conv(x)
            # x = self.alpha_rooting(x, alpha=self.alpha_root)
        x = F.relu(x)
        return x


class HadamardBlock(nn.Module):
    def __init__(self, input_channels, output_ch,
                 bn=True,
                 dropout=False,
                 kernel_size=4,
                 pad=1,
                 stride=1,
                 alpha_root=1.0,
                 lmbda=None,
                 diag=False,
                 bias=False,
                 use_res=True):
        super(HadamardBlock, self).__init__()
        """
        :param input_channels: number of channels in the input
        :param kernel_size: size of the kernel in the filter bank
        :param pad: padding size
        :param stride: stride size
        :param lmbda: number of filters to be actually used (feature not implemented)
        """
        self.bn = bn
        self.drop = dropout
        self.input_channels = input_channels
        self.output_ch = output_ch
        self.pad = pad
        self.stride = stride
        self.K = kernel_size
        self.diag = diag
        self.use_res = use_res
        self.alpha_root = alpha_root
        self.walsh = False
        if lmbda is not None:
            # if lmbda > self.K ** 2:
            self.lmbda = min(lmbda, self.K ** 2)  # limits the number of kernels
            # else:
            #    self.lmbda = lmbda
        else:
            self.lmbda = lmbda
        self.diag = diag  # flag to select diagonal entries of the block
        self.filter_bank = self.get_filter_bank(K=self.K,  # kernel size
                                                input_channels=self.input_channels,
                                                lmbda=self.lmbda,
                                                diag=self.diag).float().cuda()
        self.conv = nn.Conv2d(in_channels=self.filter_bank.shape[0], out_channels=self.output_ch,
                              kernel_size=1,
                              padding=0,
                              stride=1, bias=bias)
        if (stride != 2 or input_channels != output_ch):
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels,
                          output_ch,
                          kernel_size=2,
                          stride=stride,
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(output_ch)
            )
        else:
            self.shortcut = nn.Sequential()

        if self.bn:
            self.bnorm = nn.BatchNorm2d(self.filter_bank.shape[0])
        if self.drop:
            self.dropout = nn.Dropout(0.5)

    def fltr(self, m, n, k):
        def paley(n):
            N = 2 ** n
            P_1 = np.array([1])
            P_2 = np.block([[np.kron(P_1, [1, 1])],
                            [np.kron(P_1, [1, -1])]])
            if N == 1:  # n=0
                return P_1
            elif N == 2:  # n=1
                return P_2
            else:
                i = 2
                while i >= 2 and i <= n:
                    P_1 = P_2
                    P_2 = np.block([[np.kron(P_1, [1, 1])],
                                    [np.kron(P_1, [1, -1])]])

                    i += 1

            return P_2

        if self.walsh:
            h = hadamard(min(k, 1024))  # /np.sqrt(32)
        else:
            assert (k & (k - 1) == 0), "Kernel size must be a power of 2."
            k = int(np.log2(k))
            h = paley(k)
        f = np.dot(h[m, :].reshape(-1, 1), h[n, :].reshape(1, -1))
        return torch.as_tensor(f)

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

    def draw_filters(self, fb_=None):
        if fb_ is None: fb_ = self.filter_bank
        fig, ax = plt.subplots(len(fb_), 1, figsize=(12, 4))
        j = 0
        for i in range(len(fb_)):
            ax[i].imshow(fb_[i, 0, :, :])
            ax[i].axis('off')
            ax[i].grid(False)

    def get_filter_bank(self, K, input_channels=3, lmbda=None, diag=False):
        filter_bank = torch.zeros((K, K, K, K))
        for i in range(K):
            for j in range(K):
                filter_bank[i, j, :, :] = self.fltr(i, j, K)
        if lmbda is not None:
            ids = self.get_idx(K, lmbda)
            return torch.stack(tuple([filter_bank.view(-1, 1, K, K)[ids, :, :, :]] * input_channels), dim=0).view(
                (-1, 1, K, K))
        if diag:
            ids = self.get_idx_diag(K)
            return torch.stack(tuple([filter_bank.view(-1, 1, K, K)[ids, :, :, :]] * input_channels), dim=0).view(
                (-1, 1, K, K))
        return torch.stack(tuple([filter_bank.view(-1, 1, K, K)] * input_channels), dim=0).view(
            (-1, 1, K, K))

    def alpha_rooting(self, x, alpha=1.0):
        if alpha is not None:
            return x.sign() * torch.abs(x).pow(alpha)
        else:
            return x

    def forward(self, x):
        in_ = x
        x = F.conv2d(x.float(),
                     weight=self.filter_bank,
                     padding=self.pad,
                     stride=self.stride,
                     groups=self.input_channels)

        x = self.alpha_rooting(x, alpha=self.alpha_root)

        if self.bn:
            x = F.relu(self.bnorm(x))
        else:
            x = F.relu(x)
        if self.drop:
            x = self.dropout(x)
        if self.use_res:  # and self.alpha_root is not None:
            x = self.conv(x) + self.shortcut(in_)
            # x = self.alpha_rooting(x, alpha=self.alpha_root) + self.shortcut(in_)
            # x = self.conv(x) + self.shortcut(in_)

        else:
            x = self.conv(x)
            # x = self.alpha_rooting(x, alpha=self.alpha_root)

        x = F.relu(x)
        # x = self.alpha_rooting(x, alpha=1.5)
        return x


class SlantBlock(nn.Module):
    def __init__(self, input_channels, output_ch,
                 bn=True,
                 dropout=False,
                 kernel_size=4,
                 pad=1,
                 stride=1,
                 alpha_root=1,
                 lmbda=None,
                 diag=False,
                 bias=False,
                 use_res=True):
        super(SlantBlock, self).__init__()
        """
        :param input_channels: number of channels in the input
        :param kernel_size: size of the kernel in the filter bank
        :param pad: padding size
        :param stride: stride size
        :param lmbda: number of filters to be actually used (feature not implemented)
        """
        self.bn = bn
        self.drop = dropout
        self.input_channels = input_channels
        self.output_ch = output_ch
        self.pad = pad
        self.stride = stride
        self.K = kernel_size
        self.diag = diag
        self.use_res = use_res
        self.alpha_root = alpha_root
        if lmbda is not None:
            if lmbda > self.K ** 2:
                self.lmbda = self.K ** 2  # limits the number of kernels
            else:
                self.lmbda = lmbda
        else:
            self.lmbda = lmbda
        self.diag = diag  # flag to select diagonal entries of the block
        self.filter_bank = self.get_filter_bank(K=self.K,  # kernel size
                                                input_channels=self.input_channels,
                                                lmbda=self.lmbda,
                                                diag=self.diag).float().cuda()
        self.conv = nn.Conv2d(in_channels=self.filter_bank.shape[0], out_channels=self.output_ch,
                              kernel_size=1,
                              padding=0,
                              stride=1, bias=bias)
        if (stride != 2 or input_channels != output_ch):
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels,
                          output_ch,
                          padding=1,
                          kernel_size=2,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(output_ch)
            )
        else:
            self.shortcut = nn.Sequential()
        if self.bn:
            self.bnorm = nn.BatchNorm2d(self.filter_bank.shape[0])
        if self.drop:
            self.dropout = nn.Dropout(0.5)

    def fltr(self, m, n, k):
        assert (k & (k - 1) == 0), "Kernel size must be a power of 2."
        k = int(np.log2(k))

        def slant(n):
            N = 2 ** n
            S_1 = 1 / np.sqrt(2) * np.array([[1, 1],
                                             [1, -1]])

            an = np.sqrt((2 * N ** 2) / (4 * N ** 2 - 1))  # a2
            bn = np.sqrt((N ** 2 - 1) / (4 * N ** 2 - 1))  # b2

            S_2 = 1 / np.sqrt(2) * np.array([[1, 0, 1, 0],
                                             [an, bn, -an, bn],
                                             [0, 1, 0, -1],
                                             [-bn, an, bn, an]])

            S_2 = np.matmul(S_2, block_diag(S_1, S_1))

            if N == 2:
                return S_1
            elif N == 4:
                return S_2
            else:
                S_prev = S_2
                i = 3
                while i >= 3 and i <= n:
                    N = 2 ** i
                    an = np.sqrt((3 * N ** 2) / (4 * N ** 2 - 1))  # a2
                    bn = np.sqrt((N ** 2 - 1) / (4 * N ** 2 - 1))
                    An1 = np.array([[1, 0],
                                    [an, bn]])
                    An2 = np.array([[1, 0],
                                    [-an, bn]])
                    Bn1 = np.array([[0, 1],
                                    [-bn, an]])
                    Bn2 = np.array([[0, -1],
                                    [bn, an]])
                    S_N = np.block([[An1, np.zeros((2, N // 2 - 2)), An2, np.zeros((2, N // 2 - 2))],

                                    [np.zeros((N // 2 - 2, 2)), np.eye(N // 2 - 2), np.zeros((N // 2 - 2, 2)),
                                     np.eye(N // 2 - 2)],

                                    [Bn1, np.zeros((2, N // 2 - 2)), Bn2, np.zeros((2, N // 2 - 2))],

                                    [np.zeros((N // 2 - 2, 2)), np.eye(N // 2 - 2), np.zeros((N // 2 - 2, 2)),
                                     -np.eye(N // 2 - 2)]])

                    S_N = 1 / np.sqrt(2) * np.matmul(S_N, block_diag(S_prev, S_prev))
                    S_prev = S_N
                    i += 1
                return S_prev

        s = slant(min(k, 8))  # /np.sqrt(32)
        f = np.dot(s[m, :].reshape(-1, 1), s[n, :].reshape(1, -1))
        return torch.as_tensor(f)

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

    def draw_filters(self, fb_=None):
        if fb_ is None: fb_ = self.filter_bank
        fig, ax = plt.subplots(len(fb_), 1, figsize=(24, 6))
        j = 0
        for i in range(len(fb_)):
            ax[i].imshow(fb_[i, 0, :, :])
            ax[i].axis('off')
            ax[i].grid(False)

    def get_filter_bank(self, K, input_channels=3, lmbda=None, diag=False):
        filter_bank = torch.zeros((K, K, K, K))
        for i in range(K):
            for j in range(K):
                filter_bank[i, j, :, :] = self.fltr(i, j, K)
        if lmbda is not None:
            ids = self.get_idx(K, lmbda)
            return torch.stack(tuple([filter_bank.view(-1, 1, K, K)[ids, :, :, :]] * input_channels), dim=0).view(
                (-1, 1, K, K))
        if diag:
            ids = self.get_idx_diag(K)
            return torch.stack(tuple([filter_bank.view(-1, 1, K, K)[ids, :, :, :]] * input_channels), dim=0).view(
                (-1, 1, K, K))
        return torch.stack(tuple([filter_bank.view(-1, 1, K, K)] * input_channels), dim=0).view(
            (-1, 1, K, K))

    def alpha_rooting(self, x, alpha=1.0):
        if alpha is not None:
            return x.sign() * torch.abs(x).pow(alpha)
        else:
            return x

    def forward(self, x):
        in_ = x
        x = F.conv2d(x.float(),
                     weight=self.filter_bank,
                     padding=self.pad,
                     stride=self.stride,
                     groups=self.input_channels)
        # alpha-rooting
        x = self.alpha_rooting(x, alpha=self.alpha_root)
        if self.bn:
            x = F.relu(self.bnorm(x))
        else:
            x = F.relu(x)
        if self.drop:
            x = self.dropout(x)
        if self.use_res:
            x = self.conv(x) + self.shortcut(in_)
            # x = self.alpha_rooting(x, alpha=self.alpha_root) + self.shortcut(in_)
            # x = self.conv(x) + self.shortcut(in_)
        else:
            x = self.conv(x)
            # x = self.alpha_rooting(x, alpha=self.alpha_root)

        x = F.relu(x)
        return x


class DCTPool(HarmonicBlock):
    def __init__(self, input_channels=1, lmbda=2, kernel_size=3, stride=1):
        super(DCTPool, self).__init__(input_channels=input_channels,
                                      output_ch=input_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      lmbda=lmbda,
                                      use_res=False, )
        self.alpha_root = None
        self.output_ch = self.input_channels


class HadamardPool(HadamardBlock):
    def __init__(self, input_channels=1, lmbda=2, kernel_size=4, stride=1):
        super(HadamardPool, self).__init__(input_channels=input_channels,
                                           output_ch=input_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           lmbda=lmbda,
                                           use_res=False)
        self.alpha_root = None
        self.walsh = False
        self.output_ch = self.input_channels


class SlantPool(SlantBlock):
    def __init__(self, input_channels=1, lmbda=2, kernel_size=4, stride=1):
        super(SlantPool, self).__init__(input_channels=input_channels,
                                        output_ch=input_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        lmbda=lmbda,
                                        use_res=False)
        self.alpha_root = None
        self.output_ch = self.input_channels
