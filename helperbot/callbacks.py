from datetime import datetime
from collections import deque, defaultdict
from typing import Dict, Tuple, List
from pathlib import Path

import torch
import numpy as np

from .bot import BaseBot, StopTraining

__all__ = [
    "Callback", "MixUpCallback", "LearningRateSchedulerCallback",
    "StepwiseLinearPropertySchedulerCallback", "MovingAverageStatsTrackerCallback",
    "CheckpointCallback", "EarlyStoppingCallback"
]


class Callback:
    def on_batch_inputs(self, bot: BaseBot, input_tensors: torch.Tensor, targets: torch.Tensor):
        return input_tensors, targets

    def on_train_ends(self, bot: BaseBot):
        return

    def on_epoch_ends(self, bot: BaseBot, epoch: int):
        return

    def on_eval_ends(self, bot: BaseBot, metrics: Dict[str, Tuple[float, str]]):
        return

    def on_step_ends(self, bot: BaseBot, train_loss: float, train_weight: int):
        return

    def on_load_checkpoint(self, **kwargs):
        return

    def on_save_checkpoint(self):
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

    def on_batch_inputs(self, bot: BaseBot, input_tensors, targets):
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

    def on_step_ends(self, bot: BaseBot, train_loss, train_weight):
        self.scheduler.step()

    def on_load_checkpoint(self, **kwargs):
        self.scheduler.switch_optimizer(kwargs["optimizer"])

    def on_save_checkpoint(self):
        self.scheduler.clear_optimizer()


class StepwiseLinearPropertySchedulerCallback(Callback):
    def __init__(self, target_obj, property_name, start_val, end_val, decay_start_step, decay):
        super().__init__()
        self.target_obj = target_obj
        self.property_name = property_name
        self.start_val = start_val
        self.end_val = end_val
        self.decay_start_step = decay_start_step
        self.decay = decay

    def on_step_ends(self, bot: BaseBot, train_loss, train_weight):
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

    def on_step_ends(self, bot: BaseBot, train_loss, train_weight):
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

    def on_eval_ends(self, bot: BaseBot, metrics: Dict[str, Tuple[float, str]]):
        self.metrics["step"].append(bot.step)
        history_length = len(self.metrics["step"])
        bot.logger.info(f"Metrics at step {bot.step}:")
        for metric_name, (metric_value, metric_string) in metrics.items():
            self.metrics[metric_name].append((metric_value, metric_string))
            assert history_length == len(
                self.metrics[metric_name]), "Inconsistent metric found!"
            bot.logger.info(f"{metric_name}: {metric_string}")
            bot.logger.tb_scalars(
                metric_name, {"val": metric_value}, bot.step)

    def on_train_ends(self, bot: BaseBot):
        if self.metrics["step"]:
            bot.logger.info("Training finished. Best step(s):")
            for metric_name, metric_values in self.metrics.items():
                if metric_name == "step":
                    continue
                best_idx = np.argmin(
                    np.array([x[0] for x in metric_values]))
                bot.logger.info(
                    "%s: %s @ step %d",
                    metric_name, metric_values[best_idx][1],
                    self.metrics["step"][best_idx]
                )

    def reset(self):
        self.train_losses = deque(maxlen=self.avg_window)
        self.train_weights = deque(maxlen=self.avg_window)
        self.metrics = defaultdict(list)
        self.train_logs = []


class CheckpointCallback(Callback):
    """Save and manage checkpoints.
    """

    def __init__(
            self, keep_n_checkpoints: int = 1,
            checkpoint_dir: Path = Path("./data/cache/model_cache/"),
            monitor_metric: str = "loss"):
        super().__init__()
        self.keep_n_checkpoints = keep_n_checkpoints
        self.checkpoint_dir = checkpoint_dir
        self.monitor_metric = monitor_metric
        self.best_performers: List[Tuple[float, Path, int]] = []
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

    def on_eval_ends(self, bot: BaseBot, metrics: Dict[str, Tuple[float, str]]):
        target_value, target_string = metrics[self.monitor_metric]
        target_path = (
            self.checkpoint_dir /
            "ckpt_{}_{}_{}_{}.pth".format(
                bot.name, target_string, bot.step,
                datetime.now().strftime("%m%d%H%M"))
        )
        bot.logger.debug("Saving checkpoint %s...", target_path)
        self.best_performers.append((target_value, target_path, bot.step))
        torch.save(bot.state_dict(), target_path)
        assert Path(target_path).exists()
        self.remove_checkpoints(keep=self.keep_n_checkpoints)

    def remove_checkpoints(self, keep):
        self.best_performers = sorted(self.best_performers, key=lambda x: x[0])
        for checkpoint in np.unique([
                x[1] for x in self.best_performers[keep:]]):
            Path(checkpoint).unlink()
        self.best_performers = self.best_performers[:keep]

    def reset(self, ignore_previous=False):
        if ignore_previous:
            self.best_performers = []
        else:
            self.remove_checkpoints(0)


class EarlyStoppingCallback(Callback):
    def __init__(self, patience: int, min_improv: float, monitor_metric: str = "loss"):
        super().__init__()
        self.patience = patience
        self.min_improv = min_improv
        self.monitor_metric = monitor_metric
        self.reset()

    def on_eval_ends(self, bot: BaseBot, metrics: Dict[str, Tuple[float, str]]):
        target_value, _ = metrics[self.monitor_metric]
        if target_value < self.best - self.min_improv:
            bot.logger.info(
                "New low: %.6f improvement\n",
                self.best - target_value)
            self.best = target_value
            self.no_improv = 0
        else:
            self.no_improv += 1
        if self.no_improv > self.patience:
            raise StopTraining()

    def reset(self):
        self.no_improv = 0
        self.best = float('Inf')
