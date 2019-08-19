import os
import random
import logging
from pathlib import Path
from typing import List, Tuple, Iterable, Union, Sequence
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


class StopTraining(Exception):
    pass


@dataclass
class BaseBot:
    """Base Interface to Model Training and Inference"""
    train_loader: Iterable
    val_loader: Iterable
    criterion: object
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    name: str = "basebot"
    use_amp: bool = False
    clip_grad: float = 0
    batch_idx: int = 0
    device: Union[str, torch.device] = "cuda:0"
    log_dir: Path = Path("./data/cache/logs/")
    log_level: int = logging.INFO
    loss_format: str = "%.8f"
    use_tensorboard: bool = False
    gradient_accumulation_steps: int = 1
    echo: bool = True
    step: int = 0
    metrics: Sequence = ()
    callbacks: Sequence = ()
    pbar: bool = False

    def __post_init__(self):
        assert (self.use_amp and APEX_AVAILABLE) or (not self.use_amp)
        self.logger = Logger(
            self.name, str(self.log_dir), self.log_level,
            use_tensorboard=self.use_tensorboard, echo=self.echo)
        self.logger.info("SEED: %s", SEED)
        self.count_model_parameters()
        if APEX_AVAILABLE:
            if not self.use_amp and (hasattr(amp._amp_state, "opt_properties")):
                self.logger.warning(
                    "AMP initialization detected but use_amp = False. "
                    "Did you forget to set `use_amp = True`?")

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
        if self.step % self.gradient_accumulation_steps == 0:
            if self.clip_grad > 0:
                if not self.use_amp:
                    clip_grad_norm_(self.model.parameters(), self.clip_grad)
                else:
                    clip_grad_norm_(amp.master_params(
                        self.optimizer), self.clip_grad)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return (
            batch_loss.data.cpu().item() * self.gradient_accumulation_steps,
            input_tensors[0].size(self.batch_idx)
        )

    @staticmethod
    def extract_prediction(output):
        """Assumes multiple outputs"""
        return output

    @staticmethod
    def transform_prediction(prediction):
        return prediction

    def run_batch_inputs_callbacks(self, input_tensors, targets):
        for callback in self.callbacks:
            input_tensors, targets = callback.on_batch_inputs(
                self, input_tensors, targets)
        return input_tensors, targets

    def run_step_ends_callbacks(self, train_loss, train_weight):
        for callback in self.callbacks:
            callback.on_step_ends(self, train_loss, train_weight)

    def run_train_ends_callbacks(self):
        for callback in self.callbacks:
            callback.on_train_ends(self)

    def run_epoch_ends_callbacks(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_ends(self, epoch)

    def run_eval_ends_callbacks(self, metrics):
        for callback in self.callbacks:
            callback.on_eval_ends(self, metrics)

    def train(self, n_steps, *, checkpoint_interval):
        self.optimizer.zero_grad()
        epoch = 0
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
                    train_loss, train_weight = self.train_one_step(
                        input_tensors, targets)
                    self.run_step_ends_callbacks(train_loss, train_weight)
                    if (
                        (callable(checkpoint_interval) and checkpoint_interval(self.step)) or
                        (not callable(checkpoint_interval) and
                         self.step % checkpoint_interval == 0)
                    ):
                        metrics = self.eval(self.val_loader)
                        self.run_eval_ends_callbacks(metrics)
                    if self.step >= n_steps:
                        break
                self.run_epoch_ends_callbacks(epoch + 1)
        except (KeyboardInterrupt, StopTraining):
            pass
        finally:
            self.run_train_ends_callbacks()

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
        metrics = {"loss": (loss, self.loss_format % loss)}
        global_ys, global_preds = torch.cat(ys), torch.cat(preds)
        for metric in self.metrics:
            metric_loss, metric_string = metric(global_ys, global_preds)
            metrics[metric.name] = (metric_loss, metric_string)
        return metrics

    def predict_batch(self, input_tensors):
        self.model.eval()
        tmp = self.model(*input_tensors)
        return self.extract_prediction(tmp)

    # def predict_avg(self, loader, k=8):
    #     assert len(self.best_performers) >= k
    #     preds = []
    #     # Iterating through checkpoints
    #     for i in range(k):
    #         target = self.best_performers[i][1]
    #         self.logger.info("Loading %s", format(target))
    #         self.load_model(target)
    #         preds.append(self.predict(loader).unsqueeze(0))
    #     return torch.cat(preds, dim=0).mean(dim=0)

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

    def load_model(self, target_path):
        self.model.load_state_dict(torch.load(target_path))
