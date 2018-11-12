from typing import Callable, List, Dict

import torch
import torch.nn as nn


def opt_params(layer, learning_rate):
    return {'params': layer.parameters(), 'lr': learning_rate}


def setup_differential_learning_rates(
        optimizer_constructor: Callable[[List[Dict]], torch.optim.Optimizer],
        model: torch.nn.Module,
        lrs: List[float]) -> torch.optim.Optimizer:
    """Set up a optimizer with differential learning rates

    Reference: fast.ai v0.7

    Parameters
    ----------
    optimizer_constructor : Callable[[List[Dict]]]
        Optimizer constructor or a partial that returns an Optimizer object.
    model : torch.nn.Module
        The PyTorch model you want to optimize. Needs to have .get_layer_groups() method.
    lrs : List[float]
        A list of learning rates for each layer group.
    """
    layer_groups = model.get_layer_groups()
    assert len(layer_groups) == len(
        lrs), f'size mismatch, expected {len(layer_groups)} lrs, but got {len(lrs)}'
    optimizer = optimizer_constructor(
        [opt_params(*p) for p in zip(layer_groups, lrs)])
    return optimizer


def children(m):
    return m if isinstance(m, (list, tuple)) else list(m.children())


def set_trainable_attr(m, b):
    m.trainable = b
    for p in m.parameters():
        p.requires_grad = b


def apply_leaf(m, f):
    c = children(m)
    if isinstance(m, nn.Module):
        f(m)
    if len(c) > 0:
        for l in c:
            apply_leaf(l, f)


def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m, b))


def freeze_layers(layer_groups: List, freeze_flags: List[bool]):
    assert len(freeze_flags) == len(layer_groups)
    for layer, flag in zip(layer_groups, freeze_flags):
        set_trainable(layer, not flag)
