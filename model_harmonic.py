from harmonic_block import HarmonicBlock
import torch.nn as nn
from flatten import Flatten


class HarmonicNet(nn.Module):
    def __init__(self, in_ch,  out_ch, kernel_size,  stride, pad):
        super(HarmonicNet, self).__init__()
        self.in_channels = in_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.harmonic_block1 = HarmonicBlock(input_channels=self.in_channels,
                                             output_ch=out_ch,
                                             kernel_size=self.kernel_size,
                                             pad=self.pad,
                                             stride=self.stride)
        self.pooling = nn.MaxPool2d(3, 2)
        self.harmonic_block2 = HarmonicBlock(input_channels=out_ch,
                                             output_ch=64,
                                             kernel_size=3,
                                             pad=0,
                                             stride=2)
        self.harmonic_block3 = HarmonicBlock(input_channels=64,
                                             output_ch=128,
                                             kernel_size=3,
                                             pad=0,
                                             stride=2)

        self.flatten = Flatten()
        self.linear1 = nn.Linear(128*3*3, 1024)
        self.linear2 = nn.Linear(1024, 7)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.harmonic_block1(x)
        x = self.pooling(x)
        x = self.harmonic_block2(x)
        x = self.pooling(x)
        x = self.harmonic_block3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = x.view(-1, 7)
        return x
