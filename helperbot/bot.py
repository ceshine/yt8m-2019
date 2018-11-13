import os
import random
import logging
from pathlib import Path
from collections import deque

import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

from .logger import Logger

AVERAGING_WINDOW = 300
SEED = int(os.environ.get("SEED", 9293))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class BaseBot:
    """Base Interface to Model Training and Inference"""

    name = "basebot"

    def __init__(
            self, model, train_loader, val_loader, *, optimizer, clip_grad=0,
            avg_window=AVERAGING_WINDOW, log_dir="./data/cache/logs/", log_level=logging.INFO,
            checkpoint_dir="./data/cache/model_cache/", batch_idx=0, echo=False,
            device="cuda:0", use_tensorboard=False):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.avg_window = avg_window
        self.clip_grad = clip_grad
        self.optimizer = optimizer
        self.model = model
        self.batch_idx = batch_idx
        self.logger = Logger(self.name, log_dir, log_level,
                             use_tensorboard=use_tensorboard, echo=echo)
        self.logger.info("SEED: %s", SEED)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.device = device
        self.best_performers = []
        self.step = 0
        self.train_losses = None
        self.train_weights = None
        # Should be overriden when necessary:
        self.criterion = torch.nn.MSELoss()
        self.loss_format = "%.8f"

        self.count_model_parameters()

    def count_model_parameters(self):
        self.logger.info(
            "# of paramters: {:,d}".format(
                np.sum(p.numel() for p in self.model.parameters())))
        self.logger.info(
            "# of trainable paramters: {:,d}".format(
                np.sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

    def train_one_step(self, input_tensors, target):
        self.model.train()
        assert self.model.training
        self.optimizer.zero_grad()
        output = self.model(*input_tensors)
        batch_loss = self.criterion(self.extract_prediction(output), target)
        batch_loss.backward()
        self.train_losses.append(batch_loss.data.cpu().numpy())
        self.train_weights.append(target.size(self.batch_idx))
        if self.clip_grad > 0:
            clip_grad_norm_(self.model.parameters(), self.clip_grad)
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
        self.logger.info("Snapshot loss %s", loss_str)
        self.logger.tb_scalars(
            "losses", {"val": loss},  self.step)
        target_path = (
            self.checkpoint_dir /
            "snapshot_{}_{}.pth".format(self.name, loss_str))
        self.best_performers.append((loss, target_path, self.step))
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
            scheduler=None, snapshot_interval=2500):
        self.train_losses = deque(maxlen=self.avg_window)
        self.train_weights = deque(maxlen=self.avg_window)
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
                    if self.step % snapshot_interval == 0:
                        loss = self.snapshot()
                        if best_val_loss > loss + min_improv:
                            self.logger.info("New low\n")
                            best_val_loss = loss
                            wo_improvement = 0
                        else:
                            wo_improvement += 1
                    if scheduler:
                        scheduler.step()
                    if early_stopping_cnt and wo_improvement > early_stopping_cnt:
                        return
                    if self.step >= n_steps:
                        break
        except KeyboardInterrupt:
            pass
        self.best_performers = sorted(self.best_performers, key=lambda x: x[0])

    def eval(self, loader):
        self.model.eval()
        losses, weights = [], []
        with torch.set_grad_enabled(False):
            for *input_tensors, y in loader:
                input_tensors = [x.to(self.device) for x in input_tensors]
                output = self.model(*input_tensors)
                batch_loss = self.criterion(
                    self.extract_prediction(output), y.to(self.device))
                losses.append(batch_loss.data.cpu().numpy())
                weights.append(y.size(self.batch_idx))
        loss = np.average(losses, weights=weights)
        return loss

    def predict_batch(self, input_tensors):
        self.model.eval()
        tmp = self.model(*input_tensors)
        return self.extract_prediction(tmp)

    def predict_avg(self, loader, k=8, *, is_test=False):
        assert len(self.best_performers) >= k
        preds = []
        self.logger.info("Predicting %s...", (
            "test" if is_test else "validation"))
        # Iterating through checkpoints
        for i in range(k):
            target = self.best_performers[i][1]
            self.logger.info("Loading %s", format(target))
            self.load_model(target)
            preds.append(self.predict(loader).unsqueeze(0))
        return torch.cat(preds, dim=0).mean(dim=0)

    def predict(self, loader):
        self.model.eval()
        outputs = []
        with torch.set_grad_enabled(False):
            for *input_tensors, _ in loader:
                input_tensors = [x.to(self.device) for x in input_tensors]
                outputs.append(self.predict_batch(input_tensors))
            res = torch.cat(outputs, dim=0)
        return res.data

    def remove_checkpoints(self, keep=0):
        for checkpoint in np.unique([x[1] for x in self.best_performers[keep:]]):
            Path(checkpoint).unlink()
        self.best_performers = self.best_performers[:keep]

    def load_model(self, target_path):
        self.model.load_state_dict(torch.load(target_path))
