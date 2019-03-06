from harmonic_block import HarmonicBlock
import torch.nn as nn
from flatten import Flatten
from torch import Tensor, float32


class HarmonicNet(nn.Module):
    def __init__(self, in_ch=3,  out_ch=32,
                 kernel_size=3,  stride=2, pad=1,
                 in_shape=(64, 64)):
        super(HarmonicNet, self).__init__()

        def calc_output_shape(in_shape, kernel, stride, pad):
            w, h = in_shape
            w_ = (w-kernel+2*pad)/stride + 1
            h_ = (h-kernel+2*pad)/stride + 1
            return int(w_), int(h_)

        self.input_channels = in_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.shapes = []
        '''***'''
        self.hb1_0 = HarmonicBlock(input_channels=self.input_channels,
                                   output_ch=3,  # self.input_channels,
                                   kernel_size=self.kernel_size,
                                   pad=self.pad,
                                   stride=self.stride)

        shape_hb1_0 = calc_output_shape(in_shape,
                                        self.hb1_0.K,
                                        self.hb1_0.stride,
                                        self.hb1_0.pad)
        self.shapes.append(shape_hb1_0)
        '''***'''
        self.harmonic_block1 = HarmonicBlock(
            input_channels=self.input_channels,
            output_ch=out_ch,
            kernel_size=self.kernel_size,
            pad=self.pad,
            stride=self.stride)

        shape_harmonic_block1 = calc_output_shape(shape_hb1_0,
                                                  self.harmonic_block1.K,
                                                  self.harmonic_block1.stride,
                                                  self.harmonic_block1.pad)
        self.shapes.append(shape_harmonic_block1)
        '''***'''
        self.hb1_1 = HarmonicBlock(input_channels=out_ch,
                                   output_ch=out_ch,
                                   kernel_size=3,
                                   pad=0,
                                   stride=2)

        shape_hb1_1 = calc_output_shape(shape_harmonic_block1,
                                        self.hb1_1.K,
                                        self.hb1_1.stride,
                                        self.hb1_1.pad)
        self.shapes.append(shape_hb1_1)
        '''***'''
        self.harmonic_block2 = HarmonicBlock(input_channels=out_ch,
                                             output_ch=64,
                                             kernel_size=3,
                                             pad=2,
                                             stride=1)
        shape_harmonic_block2 = calc_output_shape(shape_hb1_1,
                                                  self.harmonic_block2.K,
                                                  self.harmonic_block2.stride,
                                                  self.harmonic_block2.pad)
        self.shapes.append(shape_harmonic_block2)
        '''***'''
        self.hb1_2 = HarmonicBlock(input_channels=64,
                                   output_ch=64,
                                   kernel_size=3,
                                   pad=0,
                                   stride=2)
        shape_hb1_2 = calc_output_shape(shape_harmonic_block2,
                                        self.hb1_2.K,
                                        self.hb1_2.stride,
                                        self.hb1_2.pad)
        self.shapes.append(shape_hb1_2)
        '''***'''
        self.harmonic_block3 = HarmonicBlock(input_channels=64,
                                             output_ch=128,
                                             kernel_size=3,
                                             pad=2,
                                             stride=1)
        shape_harmonic_block3 = calc_output_shape(shape_hb1_2,
                                                  self.harmonic_block3.K,
                                                  self.harmonic_block3.stride,
                                                  self.harmonic_block3.pad)
        self.shapes.append(shape_harmonic_block3)
        '''***'''
        self.hb1_3 = HarmonicBlock(input_channels=128,
                                   output_ch=128,
                                   kernel_size=3,
                                   pad=1,
                                   stride=1)
        shape_hb1_3 = calc_output_shape(shape_harmonic_block3,
                                        self.hb1_3.K,
                                        self.hb1_3.stride,
                                        self.hb1_3.pad)
        self.shapes.append(shape_hb1_3)
        '''***'''
        self.pooling = nn.MaxPool2d(3, 2)
        shape_pool = calc_output_shape(shape_hb1_3, 3, 2, 0)
        self.shapes.append(shape_pool)
        '''***'''
        self.flatten = Flatten()
        self.linear1 = nn.Linear(128*shape_pool[0]*shape_pool[1], 1024)
        self.linear2 = nn.Linear(1024, 128)
        self.linear3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ReLU()
        self.softmax = nn.LogSoftmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def rescale_torch(self, image: Tensor) -> Tensor:
        return (1.*(image-image.min())/(image.max()-image.min())).type(float32)

    def calc_output_shape(in_shape, kernel, stride, pad):
        w, h = in_shape
        w_ = (w-kernel+2*pad)/stride + 1
        h_ = (h-kernel+2*pad)/stride + 1
        return int(w_), int(h_)

    def forward(self, x):
        x = self.hb1_0(x)
        x = self.activation(x)
        x = self.harmonic_block1(x)
        x = self.activation(x)
        x = self.hb1_1(x)
        x = self.activation(x)
        x = self.harmonic_block2(x)
        x = self.activation(x)
        x = self.hb1_2(x)
        x = self.activation(x)
        x = self.harmonic_block3(x)
        x = self.activation(x)
        x = self.hb1_3(x)
        x = self.activation(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear3(x)
        # x = torch.clamp( x.atan_(), 0, 1)
        # x = (self.tanh(x) + 1)/2
        # x = self.softmax(x)
        # x = torch.clamp(self.sigmoid(x)**(0.99), 0, 1)
        x = x.view(-1, 10)

        return x
