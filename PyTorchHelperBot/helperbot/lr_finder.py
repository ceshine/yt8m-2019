"""LR Finder based on davidtvs/pytorch-lr-finder

Reference: https://github.com/davidtvs/pytorch-lr-finder
"""
import copy
import os
from typing import Optional, Union, Iterable

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

from .lr_scheduler import ExponentialLR, LinearLR


class LRFinder(object):
    """Learning rate range test.

    The learning rate range test increases the learning rate in a pre-training run
    between two boundaries in a linear or exponential manner. It provides valuable
    information on how well the network can be trained over a range of learning rates
    and what is the optimal learning rate.

    Example:
        >>> lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
        >>> lr_finder.range_test(dataloader, end_lr=100, num_iter=100)

    Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    fastai/lr_find: https://github.com/fastai/fastai
    """

    def __init__(
            self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module,
            device: Optional[Union[str, torch.device]] = None, memory_cache: bool = True,
            cache_dir: Optional[str] = None, use_amp: bool = False, clip_grad: float = 0):
        """Initialize LR Finder

        Parameters
        ----------
        model : nn.Module
            wrapped model.
        optimizer : torch.optim.Optimizer
            wrapped optimizer where the defined learning is assumed to be the lower boundary
            of the range test.
        criterion : nn.Module
            wrapped loss function.
        device : Optional[Union[str, torch.device]], optional
            a string ("cpu" or "cuda") with an optional ordinal for the device type (e.g.
            "cuda:X", where is the ordinal). Alternatively, can be an object representing
            the device on which the computation will take place. Default: None, uses the
            same device as `model`.
        memory_cache : bool, optional
            if this flag is set to True, `state_dict` of model and optimizer will be cached
            in memory. Otherwise, they will be saved to files under the `cache_dir`.
        cache_dir : Optional[str], optional
            path for storing temporary files. If no path is specified, system-wide temporary
            directory is used. Notice that this parameter will be ignored if `memory_cache` is True.
        use_amp: bool, optional
            Use Apex AMP.
        clip_grad: float, optional
            Clipping gradient norms.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.memory_cache = memory_cache
        self.cache_dir = cache_dir
        self.use_amp = use_amp
        self.clip_grad = clip_grad
        assert (self.use_amp and APEX_AVAILABLE) or (not self.use_amp)

        # Save the original state of the model and optimizer so they can be restored if
        # needed
        self.model_device = next(self.model.parameters()).device
        self.state_cacher = StateCacher(memory_cache, cache_dir=cache_dir)
        self.state_cacher.store('model', self.model.state_dict())
        self.state_cacher.store('optimizer', self.optimizer.state_dict())

        # If device is None, use the same as the model
        self.device = device if device else self.model_device

    @staticmethod
    def extract_prediction(output):
        return output

    def reset(self):
        """Restores the model and optimizer to their initial states."""
        self.model.load_state_dict(self.state_cacher.retrieve('model'))
        self.optimizer.load_state_dict(self.state_cacher.retrieve('optimizer'))
        self.model.to(self.model_device)

    def range_test(
            self, train_loader: Iterable, min_lr_ratio: float, total_steps: int,
            ma_decay: float = 0.95, stop_ratio: float = 10, linear_schedule: bool = False):
        """Performs the learning rate range test.

            Parameters
            ----------
            train_loader : Iterable
                the training set data laoder.
            min_lr_ratio : float
                min_lr_ratio * base_lr will be the starting learning rate.
            total_epochs : int
                the total number of "steps" in this run.
            ma_decay : float, optional
                Exponential moving average decay, by default 0.95
            stop_ratio : float, optional
                The test will stop after current loss exceeds
                (minimum_loss) * stop_ratio, by default 10.
        """
        # Reset test results
        self.history = {"lr": [], "loss": []}
        self.best_loss = None

        # Move the model to the proper device
        self.model.to(self.device)

        # Initialize the proper learning rate policy
        if linear_schedule:
            lr_scheduler = LinearLR(
                self.optimizer, min_lr_ratio=min_lr_ratio,
                total_epochs=total_steps)
        else:
            lr_scheduler = ExponentialLR(
                self.optimizer, min_lr_ratio=min_lr_ratio,
                total_epochs=total_steps)

        assert ma_decay > 0 and ma_decay < 1, "ma_decay is outside the range (0, 1)"

        # Create an iterator to get data batch by batch
        iterator = iter(train_loader)
        for step in tqdm(range(total_steps)):
            # Get a new set of inputs and labels
            try:
                *inputs, labels = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                *inputs, labels = next(iterator)

            # Train on batch and retrieve loss
            loss = self._train_batch(inputs, labels)

            # Update the learning rate
            lr_scheduler.step()
            self.history["lr"].append(lr_scheduler.get_lr()[-1])

            # Track the best loss and smooth it if smooth_f is specified
            if step == 0:
                self.best_loss = loss
            else:
                loss = (
                    (1 - ma_decay) * loss +
                    ma_decay * self.history["loss"][-1]
                )
                if loss < self.best_loss:
                    self.best_loss = loss

            # Check if the loss has diverged; if it has, stop the test
            self.history["loss"].append(loss)
            if loss >= stop_ratio * self.best_loss:
                print("Stopping early, the loss has diverged")
                break

    def _train_batch(self, inputs, labels):
        # Set model to training mode
        self.model.train()

        # Move data to the correct device
        inputs = [x.to(self.device) for x in inputs]
        labels = labels.to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.extract_prediction(self.model(*inputs))
        loss = self.criterion(outputs, labels)

        # Backward pass
        if self.use_amp:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if self.clip_grad > 0:
            if not self.use_amp:
                clip_grad_norm_(self.model.parameters(), self.clip_grad)
            else:
                clip_grad_norm_(amp.master_params(
                    self.optimizer), self.clip_grad)

        self.optimizer.step()

        return loss.item()

    def plot(self, skip_start=10, skip_end=5, log_lr=True, filepath: Optional[str] = None):
        """Plots the learning rate range test.

        Parameters
        ----------
        skip_start : int, optional
            number of batches to trim from the start, by default 10.
        skip_end : int, optional
            number of batches to trim from at the end, by default 5.
        log_lr : bool, optional
            True to plot the learning rate in a logarithmic scale; otherwise,
            plotted in a linear scale, by default True.
        filepath : Optional[str], optional
            If set, will save the figure to this file path instead.
        """
        assert skip_start >= 0
        assert skip_end >= 0

        # Get the data to plot from the history dictionary. Also, handle skip_end=0
        # properly so the behaviour is the expected
        lrs = self.history["lr"]
        losses = self.history["loss"]
        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]

        # Plot loss as a function of the learning rate
        plt.figure(figsize=(10, 5))
        plt.plot(lrs, losses)
        if log_lr:
            plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        if filepath:
            plt.savefig(filepath)
        else:
            plt.show()


class StateCacher(object):
    def __init__(self, in_memory, cache_dir=None):
        self.in_memory = in_memory
        self.cache_dir = cache_dir

        if self.cache_dir is None:
            import tempfile
            self.cache_dir = tempfile.gettempdir()
        else:
            if not os.path.isdir(self.cache_dir):
                raise ValueError('Given `cache_dir` is not a valid directory.')

        self.cached = {}

    def store(self, key, state_dict):
        if self.in_memory:
            self.cached.update({key: copy.deepcopy(state_dict)})
        else:
            fn = os.path.join(
                self.cache_dir, 'state_{}_{}.pt'.format(key, id(self)))
            self.cached.update({key: fn})
            torch.save(state_dict, fn)

    def retrieve(self, key):
        if key not in self.cached:
            raise KeyError('Target {} was not cached.'.format(key))

        if self.in_memory:
            return self.cached.get(key)
        else:
            fn = self.cached.get(key)
            if not os.path.exists(fn):
                raise RuntimeError(
                    'Failed to load state in {}. File does not exist anymore.'.format(fn))
            state_dict = torch.load(
                fn, map_location=lambda storage, location: storage)
            return state_dict

    def __del__(self):
        """Check whether there are unused cached files existing in `cache_dir` before
        this instance being destroyed."""
        if self.in_memory:
            return

        for k in self.cached:
            if os.path.exists(self.cached[k]):
                os.remove(self.cached[k])
