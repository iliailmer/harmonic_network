from layers.harmonic_block import HarmonicBlock
import torch.nn as nn
from layers.flatten import Flatten
from torch import Tensor, float32


class WideHarmonicResNet(nn.Module):
    def __init__(self, in_channels, depth, num_classes=10, widen_factor=1, lmbda=None, diag=False):
        super(WideHarmonicResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = HarmonicBlock
        self.lmbda = lmbda
        self.diag = diag
        self.conv1 = HarmonicBlock(input_channels=3, output_ch=nChannels[0],
                                   kernel_size=3,
                                   stride=1,
                                   pad=1, bias=False)
        self.drop = nn.Dropout(0.5)

        self.stack1 = self._make_layer(block,
                                       nb_layers=n,
                                       in_planes=nChannels[0],
                                       out_planes=nChannels[1],
                                       kernel_size=3,
                                       stride=1,
                                       pad=1)
        self.stack2 = self._make_layer(block,
                                       nb_layers=n,
                                       in_planes=nChannels[1],
                                       out_planes=nChannels[2],
                                       kernel_size=3,
                                       stride=2,
                                       pad=1)
        self.stack3 = self._make_layer(block,
                                       nb_layers=n,
                                       in_planes=nChannels[2],
                                       out_planes=nChannels[3],
                                       kernel_size=3,
                                       stride=2,
                                       pad=1)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, kernel_size, pad):
        strides = [stride] + [1] * (nb_layers - 1)
        stacking = []
        for st in strides:
            stacking.append(block(input_channels=in_planes,
                                  lmbda=self.lmbda,
                                  diag=self.diag,
                                  output_ch=out_planes,
                                  kernel_size=kernel_size,
                                  stride=st,
                                  pad=pad))
            if in_planes != out_planes:
                in_planes = out_planes
        return nn.Sequential(*stacking)

    def forward(self, x):
        out = self.conv1(x)
        out = self.drop(out)
        out = self.stack1(out)
        out = self.drop(out)
        out = self.stack2(out)
        out = self.drop(out)
        out = self.stack3(out)
        out = self.drop(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


'''
class HarmonicNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=32, lmbda=None, diag=False,
                 kernel_size=3, stride=2, pad=1,
                 in_shape=(64, 64)):
        super(HarmonicNet, self).__init__()

        def calc_output_shape(in_shape, kernel, stride, pad):
            w, h = in_shape
            w_ = (w - kernel + 2 * pad) / stride + 1
            h_ = (h - kernel + 2 * pad) / stride + 1
            return int(w_), int(h_)

        self.input_channels = in_ch
        self.output_channels = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.shapes = []
        self.lmbda = lmbda
        self.diag = diag
        x = (x - x.mean()) / x.std()
        x = x + self.hb1_0(x)
        x = self.activation(x)

        x = self.harmonic_block1(x)
        x = self.activation(x)

        x = self.hb1_1(x)
        x = self.activation(x)

        x = self.harmonic_block2(x)
        x = self.activation(x)

        x = x + self.hb1_2(x)
        x = self.activation(x)

        x = self.harmonic_block3(x)
        x = self.activation(x)

        x = x + self.hb1_3(x)
        x = self.activation(x)
        self.hb1_0 = HarmonicBlock(bn=True,
                                   lmbda=self.lmbda,
                                   diag=self.diag,
                                   input_channels=self.input_channels,
                                   output_ch=3,  # self.input_channels,
                                   kernel_size=3,  # self.kernel_size,
                                   pad=1,  # self.pad,
                                   stride=1)  # self.stride)

        shape_hb1_0 = calc_output_shape(in_shape,
                                        self.hb1_0.K,
                                        self.hb1_0.stride,
                                        self.hb1_0.pad)
        self.shapes.append(shape_hb1_0)
        ''''''
        self.harmonic_block1 = HarmonicBlock(bn=True,
                                             diag=self.diag,
                                             lmbda=self.lmbda,
                                             input_channels=3,
                                             output_ch=16,
                                             kernel_size=3,  # self.kernel_size,
                                             pad=1,  # self.pad,
                                             stride=1)  # self.stride)

        shape_harmonic_block1 = calc_output_shape(shape_hb1_0,
                                                  self.harmonic_block1.K,
                                                  self.harmonic_block1.stride,
                                                  self.harmonic_block1.pad)
        self.shapes.append(shape_harmonic_block1)
        ''''''
        self.hb1_1 = HarmonicBlock(bn=False, lmbda=self.lmbda,
                                   diag=self.diag,
                                   input_channels=16,
                                   output_ch=32,
                                   kernel_size=3,  # 3,
                                   pad=1,
                                   stride=2)

        shape_hb1_1 = calc_output_shape(shape_harmonic_block1,
                                        self.hb1_1.K,
                                        self.hb1_1.stride,
                                        self.hb1_1.pad)
        self.shapes.append(shape_hb1_1)
        ''''''
        self.harmonic_block2 = HarmonicBlock(bn=True, lmbda=self.lmbda,
                                             diag=self.diag,
                                             input_channels=32,
                                             output_ch=64,
                                             kernel_size=3,
                                             pad=1,
                                             stride=1)
        shape_harmonic_block2 = calc_output_shape(shape_hb1_1,
                                                  self.harmonic_block2.K,
                                                  self.harmonic_block2.stride,
                                                  self.harmonic_block2.pad)
        self.shapes.append(shape_harmonic_block2)
        ''''''
        self.hb1_2 = HarmonicBlock(bn=False, lmbda=self.lmbda,
                                   diag=self.diag,
                                   input_channels=64,
                                   output_ch=128,
                                   kernel_size=3,
                                   pad=1,
                                   stride=1)
        shape_hb1_2 = calc_output_shape(shape_harmonic_block2,
                                        self.hb1_2.K,
                                        self.hb1_2.stride,
                                        self.hb1_2.pad)
        self.shapes.append(shape_hb1_2)
        ''''''
        self.harmonic_block3 = HarmonicBlock(bn=False, lmbda=self.lmbda,
                                             diag=self.diag,
                                             input_channels=128,
                                             output_ch=256,
                                             kernel_size=3,
                                             pad=1,
                                             stride=2)
        shape_harmonic_block3 = calc_output_shape(shape_hb1_2,
                                                  self.harmonic_block3.K,
                                                  self.harmonic_block3.stride,
                                                  self.harmonic_block3.pad)
        self.shapes.append(shape_harmonic_block3)
        ''''''
        self.hb1_3 = HarmonicBlock(bn=False, lmbda=self.lmbda,
                                   diag=self.diag,
                                   input_channels=256,
                                   output_ch=512,
                                   kernel_size=3,
                                   pad=1,
                                   stride=1)
        shape_hb1_3 = calc_output_shape(shape_harmonic_block3,
                                        self.hb1_3.K,
                                        self.hb1_3.stride,
                                        self.hb1_3.pad)
        self.shapes.append(shape_hb1_3)

        
        self.pooling = nn.AvgPool2d(8)  # MaxPool2d(3, 2)
        shape_pool = calc_output_shape((8, 8), 8, 8, 0)
        self.shapes.append(shape_pool)
        
        self.flatten = Flatten()
        self.linear1 = nn.Linear(640 * shape_pool[0] * shape_pool[1], 10)#1024)
        self.linear2 = nn.Linear(1024, 128)
        self.linear3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.ReLU()
        self.softmax = nn.LogSoftmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def rescale_torch(self, image: Tensor) -> Tensor:
        return (1. * (image - image.min()) / (image.max() - image.min())).type(float32)

    def calc_output_shape(in_shape, kernel, stride, pad):
        w, h = in_shape
        w_ = (w - kernel + 2 * pad) / stride + 1
        h_ = (h - kernel + 2 * pad) / stride + 1
        return int(w_), int(h_)

    def forward(self, x: Tensor):
        # print(self.shapes)
        orig = x
        x = self.conv1(x) + self.convShortCut1(orig)
        x = self.activation(x)
        orig = x
        x = self.conv2(x) + self.convShortCut2(orig)
        x = self.activation(x)
        orig = x
        x = self.conv3(x) + self.convShortCut3(orig)
        x = self.activation(x)
        orig = x
        x = self.conv4(x) + self.convShortCut4(orig)
        x = self.activation(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = nn.functional.softmax(self.linear1(x))
        #x = self.activation(x)
        #x = self.dropout(x)
       # x = self.linear2(x)
        #x = self.activation(x)
        #x = self.dropout(x)
        #x = self.linear3(x)
        # x = torch.clamp( x.atan_(), 0, 1)
        # x = (self.tanh(x) + 1)/2
        # x = self.softmax(x)
        # x = torch.clamp(self.sigmoid(x)**(0.99), 0, 1)
        x = x.view(-1, 10)

        return x
'''

'''self.conv1 = HarmonicBlock(input_channels=self.input_channels,
                                   output_ch=16, bn=True, kernel_size=3,
                                   diag=self.diag,
                                   lmbda=self.lmbda, pad=1, stride=1)
        self.shapes.append(('conv1', calc_output_shape((32, 32), 3, 1, 1)))

        self.conv2 = [HarmonicBlock(input_channels=16,
                                    output_ch=160, bn=True, kernel_size=3,
                                    diag=self.diag,
                                    lmbda=self.lmbda, pad=1, stride=1)] + \
                     [HarmonicBlock(input_channels=160,
                                    output_ch=160, bn=True, kernel_size=3,
                                    diag=self.diag,
                                    lmbda=self.lmbda, pad=1, stride=1)] * 3
        self.conv2 = nn.Sequential(*self.conv2)
        self.shapes.append(('conv2', calc_output_shape((32, 32), 3, 1, 1)))

        self.conv3 = [HarmonicBlock(input_channels=160,
                                    output_ch=320, bn=True, kernel_size=3,
                                    diag=self.diag,
                                    lmbda=self.lmbda, pad=1, stride=2)] + \
                     [HarmonicBlock(input_channels=320,
                                    output_ch=320, bn=True, kernel_size=3,
                                    diag=self.diag,
                                    lmbda=self.lmbda, pad=1, stride=1)] * 3
        self.shapes.append(('conv3', calc_output_shape((32, 32), 3, 1, 2)))
        self.conv3 = nn.Sequential(*self.conv3)

        self.conv4 = [HarmonicBlock(input_channels=320,
                                    output_ch=640, bn=True, kernel_size=3,
                                    diag=self.diag,
                                    lmbda=self.lmbda, pad=1, stride=2)] + \
                     [HarmonicBlock(input_channels=640,
                                    output_ch=640, bn=True, kernel_size=3,
                                    diag=self.diag,
                                    lmbda=self.lmbda, pad=1, stride=1)] * 3
        self.shapes.append(('conv4', calc_output_shape((16, 16), 3, 1, 2)))
        self.conv4 = nn.Sequential(*self.conv4)

        self.convShortCut1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1)
        self.convShortCut2 = nn.Conv2d(in_channels=16, out_channels=160, kernel_size=1, stride=1)
        self.convShortCut3 = nn.Conv2d(in_channels=160, out_channels=320, kernel_size=1, stride=2)
        self.convShortCut4 = nn.Conv2d(in_channels=320, out_channels=640, kernel_size=1, stride=2)'''
'''x = (x - x.mean())/x.std()
        x = x+self.hb1_0(x)
        x = self.activation(x)

        x = self.harmonic_block1(x)
        x = self.activation(x)

        x = self.hb1_1(x)
        x = self.activation(x)

        x = self.harmonic_block2(x)
        x = self.activation(x)

        x = x+self.hb1_2(x)
        x = self.activation(x)

        x = self.harmonic_block3(x)
        x = self.activation(x)

        x = x+self.hb1_3(x)
        x = self.activation(x)
        self.hb1_0 = HarmonicBlock(bn=True,
                                   lmbda=self.lmbda,
                                   diag=self.diag,
                                   input_channels=self.input_channels,
                                   output_ch=3,  # self.input_channels,
                                   kernel_size=3,#self.kernel_size,
                                   pad=1,#self.pad,
                                   stride=1)#self.stride)

        shape_hb1_0 = calc_output_shape(in_shape,
                                        self.hb1_0.K,
                                        self.hb1_0.stride,
                                        self.hb1_0.pad)
        self.shapes.append(shape_hb1_0)
        ''''''
        self.harmonic_block1 = HarmonicBlock(bn=True,
                                             diag=self.diag,
                                             lmbda=self.lmbda,
                                             input_channels=3,
                                             output_ch=16,
                                             kernel_size=3,#self.kernel_size,
                                             pad=1,#self.pad,
                                             stride=1)#self.stride)

        shape_harmonic_block1 = calc_output_shape(shape_hb1_0,
                                                  self.harmonic_block1.K,
                                                  self.harmonic_block1.stride,
                                                  self.harmonic_block1.pad)
        self.shapes.append(shape_harmonic_block1)
        ''''''
        self.hb1_1 = HarmonicBlock(bn=False,lmbda=self.lmbda,
                                   diag=self.diag,
                                   input_channels=16,
                                   output_ch=32,
                                   kernel_size=3,#3,
                                   pad=1,
                                   stride=2)

        shape_hb1_1 = calc_output_shape(shape_harmonic_block1,
                                        self.hb1_1.K,
                                        self.hb1_1.stride,
                                        self.hb1_1.pad)
        self.shapes.append(shape_hb1_1)
        ''''''
        self.harmonic_block2 = HarmonicBlock(bn=True,lmbda=self.lmbda,
                                             diag=self.diag,
                                             input_channels=32,
                                             output_ch=64,
                                             kernel_size=3,
                                             pad=1,
                                             stride=1)
        shape_harmonic_block2 = calc_output_shape(shape_hb1_1,
                                                  self.harmonic_block2.K,
                                                  self.harmonic_block2.stride,
                                                  self.harmonic_block2.pad)
        self.shapes.append(shape_harmonic_block2)
        ''''''
        self.hb1_2 = HarmonicBlock(bn=False, lmbda=self.lmbda,
                                   diag=self.diag,
                                   input_channels=64,
                                   output_ch=128,
                                   kernel_size=3,
                                   pad=1,
                                   stride=1)
        shape_hb1_2 = calc_output_shape(shape_harmonic_block2,
                                        self.hb1_2.K,
                                        self.hb1_2.stride,
                                        self.hb1_2.pad)
        self.shapes.append(shape_hb1_2)
        ''''''
        self.harmonic_block3 = HarmonicBlock(bn=False, lmbda=self.lmbda,
                                             diag=self.diag,
                                             input_channels=128,
                                             output_ch=256,
                                             kernel_size=3,
                                             pad=1,
                                             stride=2)
        shape_harmonic_block3 = calc_output_shape(shape_hb1_2,
                                                  self.harmonic_block3.K,
                                                  self.harmonic_block3.stride,
                                                  self.harmonic_block3.pad)
        self.shapes.append(shape_harmonic_block3)
        ''''''
        self.hb1_3 = HarmonicBlock(bn=False, lmbda=self.lmbda,
                                   diag=self.diag,
                                   input_channels=256,
                                   output_ch=512,
                                   kernel_size=3,
                                   pad=1,
                                   stride=1)
        shape_hb1_3 = calc_output_shape(shape_harmonic_block3,
                                        self.hb1_3.K,
                                        self.hb1_3.stride,
                                        self.hb1_3.pad)
        self.shapes.append(shape_hb1_3)'''
