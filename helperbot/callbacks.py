import torch
import numpy as np


class Callback:
    def on_batch_inputs(self, bot, input_tensors, targets):
        return input_tensors, targets

    def on_epoch_ends(self, bot, epoch):
        return

    def on_step_ends(self, bot):
        return


class MixUpCallback(Callback):
    """Assumes the first dimension is batch.

    Reference: https://github.com/fastai/fastai/blob/master/fastai/callbacks/mixup.py
    """

    def __init__(self, alpha: float = 0.4):
        super().__init__()
        self.alpha = alpha

    def on_batch_inputs(self, bot, input_tensors, targets):
        batch = input_tensors[0]
        permuted_idx = torch.randperm(batch.size(0)).to(batch.device)
        lambd = np.random.beta(self.alpha, self.alpha, batch.size(0))
        lambd = np.concatenate(
            [lambd[:, np.newaxis], 1-lambd[:, np.newaxis]], axis=1
        ).max(axis=1)
        # Create the tensor and expand
        lambd_tensor = batch.new(lambd).view(
            -1, *[1 for _ in range(len(batch.size())-1)]
        ).expand(-1, *batch.shape[1:])
        new_batch = (batch * lambd_tensor + batch[permuted_idx] * (1-lambd))
        lambd_tensor = batch.new(lambd).view(
            -1, *[1 for _ in range(len(targets.size())-1)]
        ).expand(-1, *targets.shape[1:])
        new_labels = (targets * lambd_tensor +
                      targets[permuted_idx] * (1-lambd))
        input_tensors[0] = new_batch
        return input_tensors, new_labels


class LearningRateSchedulerCallback(Callback):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler

    def on_step_ends(self, bot):
        self.scheduler.step()
