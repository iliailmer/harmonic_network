from torch.nn import functional as F
from torch.autograd import Function


class Loss(Function):
    def __init__(self, a=1, b=0):
        super(Loss, self).__init__()
        self.a = a
        self.b = b

    def forward(self, inputs, targets):
        bce_loss = self.a * F.binary_cross_entropy_with_logits(input=inputs,
                                                               target=targets)
        margin_loss = self.b * F.multilabel_soft_margin_loss(input=inputs,
                                                             target=targets)
        self._Loss = bce_loss + margin_loss
        return self._Loss

    def backward(self):
        self._Loss.backward()

    def __call__(self, inputs, targets):
        self.forward(inputs, targets)
