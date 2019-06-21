from layers.blocks import HarmonicBlock, HadamardBlock, SlantBlock, BasicBlock
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        self.drop = nn.Dropout(0.1)

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
        if num_classes == 10 or num_classes == 100:
            self.fc = nn.Linear(nChannels[3], num_classes)
        else:
            self.fc = nn.Linear(nChannels[3] * 4, num_classes)
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
            # stacking.append(nn.Dropout(0.5))
            if in_planes != out_planes:
                in_planes = out_planes
        return nn.Sequential(*stacking)

    def forward(self, x):
        # print(x.shape)
        out = self.conv1(x)
        # print(out.shape)
        out = self.drop(out)
        # print(out.shape)
        out = self.stack1(out)
        # print(out.shape)
        out = self.drop(out)
        # print(out.shape)
        out = self.stack2(out)
        # print(out.shape)
        out = self.drop(out)
        # print(out.shape)
        out = self.stack3(out)
        # print(out.shape)
        out = self.drop(out)
        # print(out.shape)
        out = self.relu(self.bn1(out))
        # print(out.shape)
        out = F.avg_pool2d(out, 8)
        # print(out.shape)
        out = out.view(-1, self.nChannels * out.shape[2] * out.shape[3])
        # print(out.shape)
        return self.fc(out)


class WideOrthoResNet(nn.Module):
    def __init__(self, in_channels, kernel_size=None,
                 depth=10, num_classes=10, widen_factor=1,
                 block=HarmonicBlock, bn=True, drop=False, droprate=0.1,
                 alpha_root=None,
                 lmbda=None, diag=False):
        super(WideOrthoResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        self.lmbda = lmbda
        self.alpha_root = alpha_root
        self.diag = diag
        self.bn = bn
        self.dropout = drop  # extra dropout inside the block
        pad = kernel_size // 2
        self.conv1 = block(input_channels=in_channels, bn=self.bn,
                           dropout=self.dropout,
                           output_ch=nChannels[0],
                           kernel_size=kernel_size,
                           alpha_root=alpha_root,
                           stride=1,
                           pad=pad, bias=False)
        self.drop = nn.Dropout(droprate)  # ORIGINAL=0.1

        self.stack1 = self._make_layer(block,
                                       nb_layers=n,
                                       in_planes=nChannels[0],
                                       out_planes=nChannels[1],
                                       kernel_size=kernel_size,  # //2,
                                       stride=1,
                                       pad=pad)
        self.stack2 = self._make_layer(block,
                                       nb_layers=n,
                                       in_planes=nChannels[1],
                                       out_planes=nChannels[2],
                                       kernel_size=kernel_size,
                                       stride=2,
                                       pad=pad)
        self.stack3 = self._make_layer(block,
                                       nb_layers=n,
                                       in_planes=nChannels[2],
                                       out_planes=nChannels[3],
                                       kernel_size=kernel_size,
                                       stride=2,
                                       pad=pad)

        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.fc_64 = nn.Linear(nChannels[3] * 4, num_classes)
        self.nChannels = nChannels[3]

        # self.center = DecoderBlock(nChannels[3], nChannels[3], nChannels[3])
        # self.dec1 = DecoderBlock(nChannels[3]+nChannels[2], nChannels[1], 3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, kernel_size, pad):
        strides = [stride] + [1] * (nb_layers - 1)
        stacking = []
        for st in strides:
            stacking.append(block(input_channels=in_planes,
                                  output_ch=out_planes,
                                  lmbda=self.lmbda,
                                  diag=self.diag,
                                  bn=self.bn,
                                  dropout=self.dropout,  # extra dropout inside the block
                                  kernel_size=kernel_size,
                                  alpha_root=self.alpha_root,
                                  stride=st,
                                  pad=pad))
            # stacking.append(nn.Dropout(0.5))
            if in_planes != out_planes:
                in_planes = out_planes
        return nn.Sequential(*stacking)

    def _num_parameters(self, trainable=True):
        k = 0
        all_ = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                if m.weight.requires_grad:
                    all_ += m.weight.size().numel()
            if isinstance(m, HarmonicBlock) or isinstance(m, HadamardBlock) or isinstance(m, SlantBlock):
                all_ += m.filter_bank.size().numel()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                if m.weight.requires_grad:
                    k += m.weight.size().numel()
        return k, all_

    def forward(self, x):
        conv1 = self.drop(self.conv1(x))
        stack1 = self.drop(self.stack1(conv1))
        stack2 = self.drop(self.stack2(stack1))
        stack3 = self.drop(self.stack3(stack2))
        bn = self.relu(self.bn1(stack3))
        # center = self.center(bn)
        # dec1 = self.dec1(torch.cat([center, stack2],1))
        out = F.avg_pool2d(bn, bn.shape[-1])
        # if x.shape[-1] == 64:
        #    out = self.fc_64(out.view(-1, self.nChannels * 4))
        # else:
        out = self.fc(out.view(-1, self.nChannels))
        return out


class OrthoVGG(nn.Module):
    def __init__(self,
                 cfg,
                 block,
                 in_channels,
                 batch_norm=False,
                 dropout=False,
                 lmbda=None,
                 diag=False,
                 num_classes=10,
                 init_weights=True, **kwargs):
        super(OrthoVGG, self).__init__()
        self.configs = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                  'M'],
        }
        self.alpha_root = kwargs['alpha_root']
        self.features = self.make_layers(cfg=self.configs[cfg],
                                         in_channels=in_channels,
                                         batch_norm=batch_norm,
                                         dropout=dropout,
                                         lmbda=lmbda,
                                         diag=diag,
                                         block=block)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def make_layers(cfg, block, in_channels,
                    kernel_size,
                    batch_norm=False,
                    dropout=False,
                    lmbda=None,
                    diag=False):
        layers = []
        # in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = block(input_channels=in_channels,
                               output_ch=v,
                               use_res=False,
                               lmbda=lmbda,
                               diag=diag,
                               bn=batch_norm,
                               dropout=dropout,  # extra dropout inside the block
                               kernel_size=kernel_size,
                               stride=1,
                               pad=kernel_size // 2)
                layers += [conv2d]
                in_channels = v
        return nn.Sequential(*layers)
