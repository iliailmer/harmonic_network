import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
from additions import weight_rotate


class Isotonic(nn.Module):
    def __init__(self, input_channels, output_ch, bn=False,
                 kernel_size=5, pad=0, stride=1):
        super(Isotonic, self).__init__()

        self.bn = bn
        self.input_channels = input_channels
        self.output_ch = output_ch
        self.pad = pad
        self.stride = stride
        self.K = kernel_size
        self.conv = nn.Conv2d(in_channels=self.K**2,
                              out_channels=self.output_ch,
                              kernel_size=1,
                              padding=0,
                              stride=1)
        self.get_weights()
        if self.bn:
            self.bnorm = nn.BatchNorm2d(self.K**2)

    def get_weights(self):
        self.weights = np.random.randn(4, self.input_channels,
                                       self.K, self.K)
        self.weights = torch.as_tensor(self.weights)
        w0 = self.weights[0]
        w1 = self.weights[1]
        w2 = self.weights[2]
        w3 = self.weights[3]
        print(type(w0))
        self.weights = torch.stack([torch.stack([w0, w1, w2, w3]),
                                    torch.stack([weight_rotate(w3, 1),
                                                 weight_rotate(w0, 1),
                                                 weight_rotate(w1, 1),
                                                 weight_rotate(w2, 1)]),
                                    torch.stack([weight_rotate(w2, 2),
                                                 weight_rotate(w3, 2),
                                                 weight_rotate(w0, 2),
                                                 weight_rotate(w1, 2)]),
                                    torch.stack([weight_rotate(w1, 3),
                                                 weight_rotate(w2, 3),
                                                 weight_rotate(w3, 3),
                                                 weight_rotate(w0, 3)])]).view(
            (-1,
             self.input_channels,
             self.K,
             self.K))
        self.weights.requires_grad = True
        return self.weights

    def forward(self, x):
        x = F.conv2d(x.to(torch.float32), weight=self.weights.to(
            torch.float32), padding=self.pad, stride=self.stride)
        return x
