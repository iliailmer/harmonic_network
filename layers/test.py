from isotonic_layer import Isotonic
import numpy as np
import torch
import pytest
from additions import rot90, weight_rotate


def test():
    weights = torch.from_numpy(np.random.randn(4, 1, 3, 3))
    assert torch.is_tensor(weights) is True
