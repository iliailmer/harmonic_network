from torch import nn
from flatten import Flatten


class ConvNet(nn.Module):
    def __init__(self, in_ch, stride, pad):
        super(ConvNet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=32,
                      kernel_size=5, stride=stride,
                      padding=pad),
            nn.BatchNorm2d(32, eps=1e-4, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),  # 32x96x96
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 32x48x48
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),  # 64x48x48
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 64x24x24
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),  # 128x24x24
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 128x12x12
            Flatten(),
            nn.Linear(in_features=128*12*12, out_features=1024,
                      bias=False),  # 1024*12
            nn.Dropout2d(0.5),
            nn.Linear(in_features=1024, out_features=7, bias=False)
        )

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(-1, 7)
        return x
