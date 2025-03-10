import torch
from torch import nn
from torch.nn import functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class RBF(nn.Module):
    def __init__(self, dim, num_points, points=None, beta=1.0):
        super().__init__()
        if points is None:
            self.points = nn.Parameter(torch.randn(num_points, dim))
        else:
            self.points = nn.Parameter(points, requires_grad=False)
        if isinstance(beta, torch.Tensor):
            self.beta = beta.view(1, -1)
        else:
            self.beta = beta

    def __dist2__(self, x):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        points_norm = (self.points ** 2).sum(1).view(1, -1)
        d2 = x_norm + points_norm - 2.0 * torch.mm(x, self.points.transpose(0, 1))
        return d2.clamp(min=0.0)

    def forward(self, x):
        D2 = self.__dist2__(x)  # |x|-by-|points|
        val = torch.exp(-self.beta * D2)
        return val


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

    def forward(self, x, jacobian=False):
        val = super().forward(x)

        return val


class Reciprocal(nn.Module):
    def __init__(self, b=0.0):
        super().__init__()
        self.b = b

    def forward(self, x):
        val = 1.0 / (x + self.b)
        return val


class PosLinear(Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, jacobian=False):
        if self.bias is None:
            val = F.linear(x, F.softplus(self.weight))
        else:
            val = F.linear(x, F.softplus(self.weight), F.softplus(self.bias))

        return val


class Sqrt(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        val = torch.sqrt(x)
        return val
