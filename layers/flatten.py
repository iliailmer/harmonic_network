from torch import nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        self.output_shape = input.view(input.size(0), -1).shape
        return input.view(input.size(0), -1)
