from layers.cyclic_layer import Cyclic, DeCyclic
from layers.isotonic_layer import Isotonic
from layers.flatten import Flatten
from additions import calc_output_shape
from torch import nn

class DREN(nn.Module):
    def __init__(self, in_ch=3,  out_ch=32,
                 kernel_size=3,  stride=2, pad=1,
                 in_shape=(64, 64)):
        super(DREN, self).__init__()
        self.input_channels = in_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.shapes = []
        '''***'''
        self.cyclic_0 = Cyclic(input_channels=self.input_channels,
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
        self.pooling = nn.MaxPool2d(3, 2)
        shape_pool = calc_output_shape(shape_hb1_3, 3, 2, 0)
        self.shapes.append(shape_pool)
        '''***'''
        self.flatten = Flatten()
        self.linear1 = nn.Linear(128*shape_pool[0]*shape_pool[1], 1024)
        self.linear2 = nn.Linear(1024, 128)
        self.linear3 = nn.Linear(128, 7)
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
        x = x.view(-1, 7)

        return x
