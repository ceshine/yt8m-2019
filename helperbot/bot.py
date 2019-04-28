import os
import random
import logging
from pathlib import Path
from collections import deque
from typing import List, Tuple, Iterable, Optional, Union
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

from .logger import Logger

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
    use_tensorboard: bool = False
    echo: bool = True
    step: int = 0
    best_performers: List[Tuple] = field(init=False)
    train_losses: deque = field(init=False)
    train_weights: deque = field(init=False)
    metrics: Tuple = ()
    monitor_metric: str = "loss"

    def __post_init__(self):
        assert (self.use_amp and APEX_AVAILABLE) or (not self.use_amp)
        self.logger = Logger(
            self.name, str(self.log_dir), self.log_level,
            use_tensorboard=self.use_tensorboard, echo=self.echo)
        self.logger.info("SEED: %s", SEED)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.best_performers: List[Tuple] = []
        self.train_losses = deque(maxlen=self.avg_window)
        self.train_weights = deque(maxlen=self.avg_window)
        self.count_model_parameters()

    def count_model_parameters(self):
        self.logger.info(
            "# of parameters: {:,d}".format(
                np.sum(p.numel() for p in self.model.parameters())))
        self.logger.info(
            "# of trainable parameters: {:,d}".format(
                np.sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

    def train_one_step(self, input_tensors, target):
        self.model.train()
        assert self.model.training
        self.optimizer.zero_grad()
        output = self.model(*input_tensors)
        batch_loss = self.criterion(self.extract_prediction(output), target)
        if self.use_amp:
            with amp.scale_loss(batch_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            batch_loss.backward()
        self.train_losses.append(batch_loss.data.cpu().numpy())
        self.train_weights.append(target.size(self.batch_idx))
        if self.clip_grad > 0:
            if not self.use_amp:
                clip_grad_norm_(self.model.parameters(), self.clip_grad)
            else:
                clip_grad_norm_(amp.master_params(
                    self.optimizer), self.clip_grad)
        self.optimizer.step()

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
        loss = self.eval(self.val_loader)
        loss_str = self.loss_format % loss
        self.logger.info("Snapshot metric %s", loss_str)
        self.logger.tb_scalars(
            "losses", {"val": loss},  self.step)
        target_path = (
            self.checkpoint_dir /
            "snapshot_{}_{}_{}.pth".format(self.name, loss_str, self.step))
        self.best_performers.append((loss, target_path, self.step))
        self.best_performers = sorted(self.best_performers, key=lambda x: x[0])
        self.logger.info("Saving checkpoint %s...", target_path)
        torch.save(self.model.state_dict(), target_path)
        assert Path(target_path).exists()
        return loss

    @staticmethod
    def extract_prediction(output):
        """Assumes single output"""
        return output[:, 0]

    @staticmethod
    def transform_prediction(prediction):
        return prediction

    def train(
            self, n_steps, *, log_interval=50,
            early_stopping_cnt=0, min_improv=1e-4,
            scheduler=None, snapshot_interval=2500, keep_n_snapshots=-1):
        if self.val_loader is not None:
            best_val_loss = 100
        epoch = 0
        wo_improvement = 0
        self.best_performers = []
        self.logger.info(
            "Optimizer {}".format(str(self.optimizer)))
        self.logger.info("Batches per epoch: {}".format(
            len(self.train_loader)))
        try:
            while self.step < n_steps:
                epoch += 1
                self.logger.info(
                    "=" * 20 + "Epoch %d" + "=" * 20, epoch)
                for *input_tensors, target in self.train_loader:
                    input_tensors = [x.to(self.device) for x in input_tensors]
                    self.train_one_step(input_tensors, target.to(self.device))
                    self.step += 1
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
                    if scheduler:
                        scheduler.step()
                    if early_stopping_cnt and wo_improvement > early_stopping_cnt:
                        return
                    if self.step >= n_steps:
                        break
        except KeyboardInterrupt:
            pass

    def eval(self, loader):
        self.model.eval()
        losses, weights = [], []
        with torch.set_grad_enabled(False):
            for *input_tensors, y_local in tqdm(loader):
                input_tensors = [x.to(self.device) for x in input_tensors]
                output = self.model(*input_tensors)
                batch_loss = self.criterion(
                    self.extract_prediction(output), y_local.to(self.device))
                losses.append(batch_loss.data.cpu().numpy())
                weights.append(y_local.size(self.batch_idx))
        loss = np.average(losses, weights=weights)
        return loss

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
            for *input_tensors, y_local in tqdm(loader):
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
