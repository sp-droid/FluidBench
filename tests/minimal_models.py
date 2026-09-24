import torch
from torch import nn

from minimal_base import BaseRegressionModel

class BaselineZero(BaseRegressionModel):
    def __init__(self):
        super().__init__()

        self._dummy = torch.nn.Parameter(torch.zeros((), requires_grad=True))

    def forward(self, X, scalars=None):
        y = torch.zeros_like(X[1]) + self._dummy * 0.0
        return y

class BaselineSame(BaseRegressionModel):
    def __init__(self):
        super().__init__()

    def forward(self, X, scalars=None):
        y = X[1]
        return y
