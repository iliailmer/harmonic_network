from ..myloss import Loss, Variable
import torch

inputs = Variable(torch.random.randn((10,10)))
targets = Variable(torch.random.randn((10,10)))
B = Loss(1,1)
