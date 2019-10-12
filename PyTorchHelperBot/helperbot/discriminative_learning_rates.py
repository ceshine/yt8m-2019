from typing import Callable, Sequence, Dict, Any, Union

import torch
import torch.nn as nn


LayerGroup = Union[nn.Module, nn.ModuleList]


def optimizer_with_layer_attributes(
        optimizer_constructor: Callable[[Sequence[Dict]], torch.optim.Optimizer],
        layer_groups: Sequence[LayerGroup],
        attributes: Dict[str, Sequence[Any]]) -> torch.optim.Optimizer:
    """Set up a optimizer with discriminative learning rates

    Reference: fast.ai v0.7

    Parameters
    ----------
    optimizer_constructor : Callable[[List[Dict]]]
        Optimizer constructor or a partial that returns an Optimizer object.
    layer_groups : Sequence[LayerGroup]
        A sequence of layer groups. Layer should have a `.parameters()` method.
    attribute : Dict[List[float]]
        A list of learning rates for each layer group. Example: {"lr": [0.01, 0.1]}.
    """
    assert attributes
    baseline_length = len(attributes[list(attributes.keys())[0]])
    for key in attributes.keys():
        assert len(attributes[key]) == baseline_length, \
            "All attribute list should have the same length."
    assert len(layer_groups) == baseline_length, \
        f'Size mismatch, expected {baseline_length} lrs, but got {len(layer_groups)}'
    layer_attributes = []
    for i in range(baseline_length):
        tmp_attr = {"params": layer_groups[i].parameters()}
        for key, item in attributes.items():
            tmp_attr[key] = item[i]
        layer_attributes.append(tmp_attr)
    optimizer = optimizer_constructor(layer_attributes)
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


def freeze_layers(layer_groups: Sequence[LayerGroup], freeze_flags: Sequence[bool]):
    assert len(freeze_flags) == len(layer_groups)
    for layer, flag in zip(layer_groups, freeze_flags):
        set_trainable(layer, not flag)
