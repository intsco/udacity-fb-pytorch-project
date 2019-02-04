import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


class AdaptiveConcatPool2d(nn.Module):
    """ Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d` """

    def __init__(self, sz=None):
        """ Output will be 2*sz or 2 if sz is None """
        super().__init__()
        sz = sz or 1
        self.ap, self.mp = nn.AdaptiveAvgPool2d(sz), nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Lambda(nn.Module):
    """ An easy way to create a pytorch layer for a simple `func` """

    def __init__(self, func):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def Flatten(full=False):
    """ Flatten `x` to a single dimension, often used at the end of a model. `full` for rank-1 tensor """
    func = (lambda x: x.view(-1)) if full else (lambda x: x.view(x.size(0), -1))
    return Lambda(func)


flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if len(list(m.children())) > 0 else [m]


def freeze(model):
    flat_model = [nn.Sequential(*flatten_model(model))]

    for layer in flat_model:
        if not isinstance(layer, bn_types):
            for p in layer.parameters():
                p.requires_grad = False


def unfreeze(model):
    flat_model = [nn.Sequential(*flatten_model(model))]

    for layer in flat_model:
        for p in layer.parameters():
            p.requires_grad = True


def create_body(arch, pretrained=True, cut=-2):
    """ Cut off the body of a typically pretrained `model` at `cut` or as specified by `body_fn` """
    model = arch(pretrained)
    if pretrained:
        freeze(model)
    return nn.Sequential(*list(model.children())[:cut])


def bn_drop_lin(n_in, n_out, bn=True, p=0., actn=None):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0:
        layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None:
        layers.append(actn)
    return layers


def create_head(input_f, out_f, lin_ftrs=None, dropout_ps=0.5, bn_final=False):
    """ Model head that takes `input_f` features, runs through `lin_ftrs`, and about `out_f` classes """
    lin_ftrs = [input_f, 512, out_f] if lin_ftrs is None else [input_f] + lin_ftrs + [out_f]
    ps = [dropout_ps]
    if len(ps) == 1:
        ps = [ps[0] / 2] * (len(lin_ftrs) - 2) + ps

    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
    layers = [AdaptiveConcatPool2d(), Flatten()]
    for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += bn_drop_lin(ni, no, True, p, actn)
    if bn_final:
        layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)


def requires_grad(m):
    """ If `b` is not set `requires_grad` on all params in `m`, else return `requires_grad` of first param """
    ps = list(m.parameters())
    if not ps:
        return None
    return ps[0].requires_grad


def init_module_rec(m, func):
    if isinstance(m, nn.Module) and not isinstance(m, bn_types) and requires_grad(m):
        func(m)
    for ch in m.children():
        init_module_rec(ch, func)


def init_weight_bias(m, func=None):
    if func:
        if hasattr(m, 'weight'): func(m.weight)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'): m.bias.data.fill_(0.)
    return m
