import os
import random
import logging
from pathlib import Path
from collections import deque
from typing import List, Tuple, Iterable, Optional, Union, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

from .logger import Logger
from .metrics import Metric

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

SEED = int(os.environ.get("SEED", 9293))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


@dataclass
class BaseBot:
    """Base Interface to Model Training and Inference"""
    train_loader: Iterable
    val_loader: Iterable
    avg_window: int
    criterion: object
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    name: str = "basebot"
    use_amp: bool = False
    clip_grad: float = 0
    batch_idx: int = 0
    checkpoint_dir: Path = Path("./data/cache/model_cache/")
    device: Union[str, torch.device] = "cuda:0"
    log_dir: Path = Path("./data/cache/logs/")
    log_level: int = logging.INFO
    loss_format: str = "%.8f"
    metric_format: Optional[str] = None
    use_tensorboard: bool = False
    gradient_accumulation_steps: int = 1
    echo: bool = True
    step: int = 0
    best_performers: List[Tuple] = field(init=False)
    train_losses: deque = field(init=False)
    train_weights: deque = field(init=False)
    metrics: Sequence = ()
    callbacks: Sequence = ()
    monitor_metric: str = "loss"
    pbar: bool = False

    def __post_init__(self):
        assert (self.use_amp and APEX_AVAILABLE) or (not self.use_amp)
        self.logger = Logger(
            self.name, str(self.log_dir), self.log_level,
            use_tensorboard=self.use_tensorboard, echo=self.echo)
        self.logger.info("SEED: %s", SEED)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.best_performers: List[Tuple] = []
        if self.metric_format is None:
            self.metric_format = self.loss_format
        self.train_losses = deque(maxlen=self.avg_window)
        self.train_weights = deque(maxlen=self.avg_window)
        self.count_model_parameters()

    def count_model_parameters(self):
        self.logger.info(
            "# of parameters: {:,d}".format(
                np.sum(list(p.numel() for p in self.model.parameters()))))
        self.logger.info(
            "# of trainable parameters: {:,d}".format(
                np.sum(list(p.numel() for p in self.model.parameters() if p.requires_grad))))

    def train_one_step(self, input_tensors, target):
        self.model.train()
        assert self.model.training
        output = self.model(*input_tensors)
        batch_loss = self.criterion(
            self.extract_prediction(output), target
        ) / self.gradient_accumulation_steps
        if self.use_amp:
            with amp.scale_loss(batch_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            batch_loss.backward()
        self.train_losses.append(
            batch_loss.data.cpu().numpy() * self.gradient_accumulation_steps)
        self.train_weights.append(input_tensors[0].size(self.batch_idx))
        if self.step % self.gradient_accumulation_steps == 0:
            if self.clip_grad > 0:
                if not self.use_amp:
                    clip_grad_norm_(self.model.parameters(), self.clip_grad)
                else:
                    clip_grad_norm_(amp.master_params(
                        self.optimizer), self.clip_grad)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def log_progress(self):
        train_loss_avg = np.average(
            self.train_losses, weights=self.train_weights)
        self.logger.info(
            "Step %s: train %.6f lr: %.3e",
            self.step, train_loss_avg, self.optimizer.param_groups[-1]['lr'])
        self.logger.tb_scalars(
            "lr", self.optimizer.param_groups[0]['lr'], self.step)
        self.logger.tb_scalars(
            "losses", {"train": train_loss_avg}, self.step)

    def snapshot(self):
        metrics = self.eval(self.val_loader)
        target_metric = metrics[self.monitor_metric]
        metric_str = self.metric_format % target_metric
        self.logger.info("Snapshot metric %s", metric_str)
        self.logger.tb_scalars(
            "losses", {"val": metrics["loss"]},  self.step)
        self.logger.tb_scalars(
            "monitor_metric", {"val": target_metric},  self.step)
        target_path = (
            self.checkpoint_dir /
            "snapshot_{}_{}_{}.pth".format(self.name, metric_str, self.step))
        self.best_performers.append((target_metric, target_path, self.step))
        self.best_performers = sorted(self.best_performers, key=lambda x: x[0])
        self.logger.info("Saving checkpoint %s...", target_path)
        torch.save(self.model.state_dict(), target_path)
        assert Path(target_path).exists()
        return target_metric

    @staticmethod
    def extract_prediction(output):
        """Assumes single output"""
        return output[:, 0]

    @staticmethod
    def transform_prediction(prediction):
        return prediction

    def run_batch_inputs_callbacks(self, input_tensors, targets):
        for callback in self.callbacks:
            input_tensors, targets = callback.on_batch_inputs(
                self, input_tensors, targets)
        return input_tensors, targets

    def run_step_ends_callbacks(self):
        for callback in self.callbacks:
            callback.on_step_ends(self)

    def run_epoch_ends_callbacks(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_ends(self, epoch)

    def train(
            self, n_steps, *, log_interval=50,
            early_stopping_cnt=0, min_improv=1e-4,
            snapshot_interval=2500, keep_n_snapshots=-1):
        self.optimizer.zero_grad()
        if self.val_loader is not None:
            best_val_loss = 100
        epoch = 0
        wo_improvement = 0
        self.best_performers = []
        self.logger.info(
            "Optimizer {}".format(str(self.optimizer)))
        try:
            self.logger.info("Batches per epoch: {}".format(
                len(self.train_loader)))
        except TypeError:
            # IterableDataset doesn't have length
            pass
        try:
            while self.step < n_steps:
                epoch += 1
                self.logger.info(
                    "=" * 20 + "Epoch %d" + "=" * 20, epoch)
                for *input_tensors, targets in self.train_loader:
                    input_tensors = [x.to(self.device) for x in input_tensors]
                    targets = targets.to(self.device)
                    input_tensors, targets = self.run_batch_inputs_callbacks(
                        input_tensors, targets)
                    self.step += 1
                    self.train_one_step(input_tensors, targets)
                    if self.step % log_interval == 0:
                        self.log_progress()
                    if ((callable(snapshot_interval) and snapshot_interval(self.step))
                            or (not callable(snapshot_interval) and self.step % snapshot_interval == 0)):
                        loss = self.snapshot()
                        if best_val_loss > loss + min_improv:
                            self.logger.info("New low\n")
                            best_val_loss = loss
                            wo_improvement = 0
                        else:
                            wo_improvement += 1
                        if keep_n_snapshots > 0:
                            self.remove_checkpoints(keep=keep_n_snapshots)
                    self.run_step_ends_callbacks()
                    if early_stopping_cnt and wo_improvement > early_stopping_cnt:
                        return
                    if self.step >= n_steps:
                        break
                self.run_epoch_ends_callbacks(epoch + 1)
        except KeyboardInterrupt:
            pass
        self.log_progress()

    def eval(self, loader):
        """Warning: Only support datasets whose predictions and labels fit in memory together."""
        self.model.eval()
        preds, ys = [], []
        losses, weights = [], []
        self.logger.debug("Evaluating...")
        with torch.set_grad_enabled(False):
            for *input_tensors, y_local in tqdm(loader, disable=not self.pbar):
                input_tensors = [x.to(self.device) for x in input_tensors]
                output = self.extract_prediction(self.model(*input_tensors))
                batch_loss = self.criterion(
                    output, y_local.to(self.device))
                losses.append(batch_loss.data.cpu().item())
                weights.append(y_local.size(self.batch_idx))
                # Save batch labels and predictions
                preds.append(output.cpu())
                ys.append(y_local.cpu())
        loss = np.average(losses, weights=weights)
        self.logger.info("Criterion loss: {}".format(self.loss_format % loss))
        metrics = {"loss": loss}
        global_ys, global_preds = torch.cat(ys), torch.cat(preds)
        for metric in self.metrics:
            metric_loss, metric_string = metric(global_ys, global_preds)
            metrics[metric.name] = metric_loss
            self.logger.info(f"{metric.name}: {metric_string}")
        return metrics

    def predict_batch(self, input_tensors):
        self.model.eval()
        tmp = self.model(*input_tensors)
        return self.extract_prediction(tmp)

    def predict_avg(self, loader, k=8):
        assert len(self.best_performers) >= k
        preds = []
        # Iterating through checkpoints
        for i in range(k):
            target = self.best_performers[i][1]
            self.logger.info("Loading %s", format(target))
            self.load_model(target)
            preds.append(self.predict(loader).unsqueeze(0))
        return torch.cat(preds, dim=0).mean(dim=0)

    def predict(self, loader, *, return_y=False):
        self.model.eval()
        outputs, y_global = [], []
        with torch.set_grad_enabled(False):
            for *input_tensors, y_local in tqdm(loader, disable=not self.pbar):
                input_tensors = [x.to(self.device) for x in input_tensors]
                outputs.append(self.predict_batch(input_tensors).cpu())
                if return_y:
                    y_global.append(y_local)
            outputs = torch.cat(outputs, dim=0)
        if return_y:
            y_global = torch.cat(y_global, dim=0)
            return outputs, y_global.cpu()
        return outputs

    def remove_checkpoints(self, keep=0):
        for checkpoint in np.unique([x[1] for x in self.best_performers[keep:]]):
            Path(checkpoint).unlink()
        self.best_performers = self.best_performers[:keep]

    def load_model(self, target_path):
        self.model.load_state_dict(torch.load(target_path))
