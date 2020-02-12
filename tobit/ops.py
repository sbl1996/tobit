import math
import torch


def cdf(value):
    return 0.5 * (1 + torch.erf(value / math.sqrt(2)))


def pdf(value):
    return torch.exp(-0.5 * (value ** 2)) / math.sqrt(2 * math.pi)


def atleast_1d(t):
    t = torch.as_tensor(t)
    if t.dim() == 0:
        t = t.view(1)
    return t


def atleast_2d(t):
    t = torch.as_tensor(t)
    if t.dim() == 0:
        t = t.view(1, 1)
    elif t.dim() == 1:
        t = t[:, None]
    return t
