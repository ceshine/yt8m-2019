from functools import partial

import torch
import numpy as np
import pretrainedmodels
from torch import nn
from torch.nn import functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def create_net(net_cls, pretrained: bool):
    net = net_cls(pretrained=pretrained)
    return net


def get_head(nf: int, n_classes):
    model = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        Flatten(),
        # nn.Dropout(p=0.25),
        nn.Linear(nf, n_classes)
    )
    return model


def init_weights(model):
    for i, module in enumerate(model):
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if module.weight is not None:
                nn.init.uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            if getattr(module, "weight_v", None) is not None:
                print("Initing linear with weight normalization")
                assert model[i].weight_g is not None
            else:
                nn.init.kaiming_normal_(module.weight)
                print("Initing linear")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    return model


def get_seresnet_model(arch: str = "se_resnext50_32x4d", n_classes: int = 10, pretrained: bool = False):
    full = pretrainedmodels.__dict__[arch](
        pretrained='imagenet' if pretrained else None)
    model = nn.Sequential(
        nn.Sequential(full.layer0, full.layer1, full.layer2, full.layer3[:3]),
        nn.Sequential(full.layer3[3:], full.layer4),
        get_head(2048, n_classes))
    print(" | ".join([
        "{:,d}".format(np.sum([p.numel() for p in x.parameters()])) for x in model]))
    if pretrained:
        init_weights(model[-1])
        return model
    return init_weights(model)


def get_densenet_model(arch: str = "densenet169", n_classes: int = 10, pretrained: bool = False):
    full = pretrainedmodels.__dict__[arch](
        pretrained='imagenet' if pretrained else None)
    print(len(full.features))
    model = nn.Sequential(
        nn.Sequential(*full.features[:8]),
        nn.Sequential(*full.features[8:]),
        get_head(full.features[-1].num_features, n_classes))
    print(" | ".join([
        "{:,d}".format(np.sum([p.numel() for p in x.parameters()])) for x in model]))
    if pretrained:
        init_weights(model[-1])
        return model
    return init_weights(model)


class Swish(nn.Module):
    def forward(self, x):
        """ Swish activation function """
        return x * torch.sigmoid(x)
