from collections import deque, defaultdict
from typing import Dict, Tuple

import torch
import numpy as np

__all__ = [
    "Callback", "MixUpCallback", "LearningRateSchedulerCallback",
    "StepwiseLinearPropertySchedulerCallback", "MovingAverageStatsTrackerCallback"
]


class Callback:
    def on_batch_inputs(self, bot, input_tensors, targets):
        return input_tensors, targets

    def on_epoch_ends(self, bot, epoch):
        return

    def on_eval_ends(self, bot, metrics):
        return

    def on_step_ends(self, bot, train_loss, train_weight):
        return

    def reset(self):
        return


class MixUpCallback(Callback):
    """Assumes the first dimension is batch.

    Reference: https://github.com/fastai/fastai/blob/master/fastai/callbacks/mixup.py
    """

    def __init__(self, alpha: float = 0.4, softmax_target: bool = False):
        super().__init__()
        self.alpha = alpha
        self.softmax_target = softmax_target

    def on_batch_inputs(self, bot, input_tensors, targets):
        batch = input_tensors[0]
        permuted_idx = torch.randperm(batch.size(0)).to(batch.device)
        lambd = np.random.beta(self.alpha, self.alpha, batch.size(0))
        lambd = np.concatenate(
            [lambd[:, np.newaxis], 1-lambd[:, np.newaxis]], axis=1
        ).max(axis=1)
        # Create the tensor and expand (for batch inputs)
        lambd_tensor = batch.new(lambd).view(
            -1, *[1 for _ in range(len(batch.size())-1)]
        ).expand(-1, *batch.shape[1:])
        # Combine input batch
        new_batch = (batch * lambd_tensor +
                     batch[permuted_idx] * (1-lambd_tensor))
        # Create the tensor and expand (for target)
        lambd_tensor = batch.new(lambd).view(
            -1, *[1 for _ in range(len(targets.size())-1)]
        ).expand(-1, *targets.shape[1:])
        # Combine targets
        if self.softmax_target:
            new_targets = torch.stack([
                targets.float(), targets[permuted_idx].float(), lambd_tensor
            ], dim=1)
        else:
            new_targets = (
                targets * lambd_tensor +
                targets[permuted_idx] * (1-lambd_tensor)
            )
        input_tensors[0] = new_batch
        return input_tensors, new_targets


class LearningRateSchedulerCallback(Callback):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler

    def on_step_ends(self, bot, *args, **kwargs):
        self.scheduler.step()


class StepwiseLinearPropertySchedulerCallback(Callback):
    def __init__(self, target_obj, property_name, start_val, end_val, decay_start_step, decay):
        super().__init__()
        self.target_obj = target_obj
        self.property_name = property_name
        self.start_val = start_val
        self.end_val = end_val
        self.decay_start_step = decay_start_step
        self.decay = decay

    def on_step_ends(self, bot, *args, **kwargs):
        if bot.step % 200 == 0:
            bot.logger.info(
                "%s %s %.4f",
                self.target_obj.__class__.__name__,
                self.property_name,
                getattr(self.target_obj, self.property_name))
        new_val = self.get_value(bot)
        setattr(self.target_obj, self.property_name, new_val)

    def get_value(self, bot):
        if self.start_val == self.end_val or bot.step <= self.decay_start_step:
            return self.start_val
        change = (self.end_val - self.start_val) * min(
            ((bot.step - self.decay_start_step) * self.decay), 1
        )
        return self.start_val + change


class MovingAverageStatsTrackerCallback(Callback):
    """Keep moving average for training losses.

    Raw values for evaluation stats.
    """

    def __init__(self, avg_window: int, log_interval: int):
        super().__init__()
        self.avg_window = avg_window
        self.log_interval = log_interval
        self.reset()

    def on_step_ends(self, bot, train_loss, train_weight):
        self.train_losses.append(train_loss)
        self.train_weights.append(train_weight)
        if bot.step % self.log_interval == 0:
            train_loss_avg = np.average(
                self.train_losses, weights=self.train_weights)
            bot.logger.info(
                "Step %s: train %.6f lr: %.3e",
                bot.step, train_loss_avg, bot.optimizer.param_groups[-1]['lr'])
            bot.logger.tb_scalars(
                "lr", bot.optimizer.param_groups[0]['lr'], bot.step)
            bot.logger.tb_scalars(
                "loss", {"train": train_loss_avg}, bot.step)
            self.train_logs.append(train_loss_avg)

    def on_eval_ends(self, bot, metrics: Dict[str, Tuple[float, str]]):
        self.metrics["step"].append(bot.step)
        history_length = len(self.metrics["step"])
        bot.logger.info(f"Metrics at step {bot.step}:")
        for metric_name, (metric_value, metric_string) in metrics.items():
            self.metrics[metric_name].append(metric_value)
            assert history_length == len(
                self.metrics[metric_name]), "Inconsistent metric found!"
            bot.logger.info(f"{metric_name}: {metric_string}")
            bot.logger.tb_scalars(
                metric_name, {"val": metric_value}, bot.step)

    def reset(self):
        self.train_losses = deque(maxlen=self.avg_window)
        self.train_weights = deque(maxlen=self.avg_window)
        self.metrics = defaultdict(list)
        self.train_logs = []
