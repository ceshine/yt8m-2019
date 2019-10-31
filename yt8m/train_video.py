import os
import glob
import argparse
from pathlib import Path
from typing import Tuple
from datetime import datetime
from dataclasses import dataclass

import torch
# from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import numpy as np
from helperbot import (
    BaseBot, WeightDecayOptimizerWrapper,
    LearningRateSchedulerCallback,
    MovingAverageStatsTrackerCallback,
    CheckpointCallback, MultiStageScheduler, LinearLR
)
from helperbot.metrics import Metric

from .dataloader import YoutubeVideoDataset, DataLoader, collate_videos
from .models import NeXtVLADModel, GatedDBoFModel, SampleFrameModelWrapper
from .telegram_tokens import BOT_TOKEN, CHAT_ID
from .telegram_sender import telegram_sender

CACHE_DIR = Path('./data/cache/video')
CACHE_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR = Path('./data/cache/video')
MODEL_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR_STR = './data/video/'
NO_DECAY = ['bias', 'LayerNorm.weight', 'BatchNorm.weight']


class Accuracy(Metric):
    """Multi-Label Classification Accuracy"""
    name = "accuracy"

    def __init__(self, cutoff=0.5):
        super().__init__()
        self.cutoff = cutoff

    def __call__(self, truth: torch.Tensor, pred: torch.Tensor) -> Tuple[float, str]:
        probs = torch.sigmoid(pred).cpu()
        pred_positives = (probs > self.cutoff).float()
        correct = torch.sum(
            pred_positives == truth.cpu()
        ).item()
        total = pred_positives.numel()
        accuracy = (correct / total)
        return accuracy * -1, f"{accuracy * 100:.4f}%"


class Recall(Metric):
    """Multi-Label Classification Recall"""
    name = "recall"

    def __init__(self, cutoff=0.5):
        super().__init__()
        self.cutoff = cutoff

    def __call__(self, truth: torch.Tensor, pred: torch.Tensor) -> Tuple[float, str]:
        probs = torch.sigmoid(pred).cpu()
        pred_positives = (probs > self.cutoff)
        correct = torch.sum(
            pred_positives & (truth == 1).cpu()
        ).item()
        # print((pred_positives == 1).sum(), (truth == 1).sum())
        total = (truth == 1).sum().float()
        recall = (correct / total)
        return recall * -1, f"{recall * 100:.2f}%"


class Precision(Metric):
    """Multi-Label Classification Precision"""
    name = "precision"

    def __init__(self, cutoff=0.5):
        super().__init__()
        self.cutoff = cutoff

    def __call__(self, truth: torch.Tensor, pred: torch.Tensor) -> Tuple[float, str]:
        probs = torch.sigmoid(pred).cpu()
        pred_positives = (probs > self.cutoff)
        correct = torch.sum(
            pred_positives & (truth == 1).cpu()
        ).item()
        total = (pred_positives).sum().float()
        precision = (correct / total)
        return precision * -1, f"{precision * 100:.2f}%"


@dataclass
class YoutubeVideoBot(BaseBot):
    checkpoint_dir: Path = CACHE_DIR / "model_cache/"
    log_dir: Path = MODEL_DIR / "logs/"

    def __post_init__(self):
        super().__post_init__()
        self.loss_format = "%.6f"
        self.metrics = (Accuracy(), Recall(), Precision())
        self.monitor_metric = "loss"

    def extract_prediction(self, x):
        return x


def collect_file_paths(base_folder="train"):
    return list(glob.glob(str(DATA_DIR_STR + f"{base_folder}/*.tfrecord")))


def get_loaders(config):
    file_paths = collect_file_paths()
    train_ds = YoutubeVideoDataset(
        file_paths, epochs=None, seed=int(os.environ.get("SEED", 42)))
    train_loader = DataLoader(
        train_ds, num_workers=4, batch_size=config['training']['batch_size'],
        collate_fn=collate_videos)
    file_paths = collect_file_paths(base_folder="valid")
    valid_ds = YoutubeVideoDataset(
        file_paths, epochs=None, max_examples=32*4000/4)
    valid_loader = DataLoader(
        valid_ds, num_workers=4,
        batch_size=config['training']['batch_size'],
        collate_fn=collate_videos)
    return train_loader, valid_loader


def resume_training(config, checkpoint_path, model, optimizer, train_loader, valid_loader):
    bot = YoutubeVideoBot.load_checkpoint(
        checkpoint_path, train_loader, valid_loader,
        model, optimizer
    )
    checkpoints = None
    for callback in bot.callbacks:
        if isinstance(callback, CheckpointCallback):
            checkpoints = callback
            break
    if checkpoints:
        # We could reset the checkpoints
        checkpoints.reset(ignore_previous=True)
    bot.train(checkpoint_interval=config['ckpt_interval'])
    if checkpoints:
        bot.load_model(checkpoints.best_performers[0][1])
        checkpoints.remove_checkpoints(keep=0)
    return bot


def train_from_start(config, model, optimizer, train_loader, valid_loader):
    n_steps = config['steps']
    break_points = [0, int(n_steps*0.2)]
    lr_durations = np.diff(break_points + [n_steps])
    checkpoints = CheckpointCallback(
        keep_n_checkpoints=1,
        checkpoint_dir=CACHE_DIR / "model_cache/",
        monitor_metric="loss"
    )
    bot = YoutubeVideoBot(
        model=model, train_loader=train_loader,
        valid_loader=valid_loader, clip_grad=10.,
        optimizer=optimizer, echo=True,
        criterion=torch.nn.BCEWithLogitsLoss(),
        callbacks=[
            LearningRateSchedulerCallback(
                MultiStageScheduler(
                    [
                        LinearLR(optimizer, 0.01, lr_durations[0]),
                        LinearLR(
                            optimizer, 0.01, lr_durations[1], upward=False)
                    ],
                    start_at_epochs=break_points
                )
            ),
            MovingAverageStatsTrackerCallback(
                avg_window=1000,
                log_interval=1000,
            ),
            checkpoints,
        ],
        pbar=True, use_tensorboard=False
    )
    bot.train(
        total_steps=n_steps, checkpoint_interval=config['ckpt_interval']
    )
    bot.load_model(checkpoints.best_performers[0][1])
    checkpoints.remove_checkpoints(keep=0)
    return bot


def create_video_model(model_config):
    if model_config["type"] == "dbof":
        model = SampleFrameModelWrapper(
            GatedDBoFModel(
                hidden_dim=model_config["hidden_dim"],
                p_drop=model_config["p_drop"],
                fcn_dim=model_config["fcn_dim"],
                num_mixtures=model_config["n_mixtures"],
                per_class=model_config["per_class"],
                frame_se_reduction=model_config["frame_se_reduction"],
                video_se_reduction=model_config["video_se_reduction"]
            ).cuda(),
            max_len=model_config["max_len"]
        )
    elif model_config["type"] == "nextvlad":
        model = SampleFrameModelWrapper(
            NeXtVLADModel(
                p_drop=model_config["p_drop"],
                fcn_dim=model_config["fcn_dim"],
                groups=model_config["groups"],
                expansion=2,
                n_clusters=model_config["n_clusters"],
                num_mixtures=model_config["n_mixtures"],
                per_class=model_config["per_class"],
                add_batchnorm=model_config["add_batchnorm"],
                se_reduction=model_config["se_reduction"]
            ).cuda(),
            max_len=model_config["max_len"]
        )
    else:
        raise ValueError("Unrecognized model: %s" % model_config["type"])
    return model


@telegram_sender(token=BOT_TOKEN, chat_id=CHAT_ID, name="Training on Video")
def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('config', type=str)
    arg('--from-checkpoint', type=str, default='')
    args = parser.parse_args()
    with open(args.config) as fin:
        config = yaml.safe_load(fin)
    train_loader, valid_loader = get_loaders(config["video"])

    model_config = config["video"]["model"]
    training_config = config["video"]["training"]
    model = create_video_model(model_config)
    print(model)

    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in NO_DECAY)],
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if any(nd in n for nd in NO_DECAY)],
        }
    ]
    optimizer = WeightDecayOptimizerWrapper(
        torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=float(training_config['lr']),
            eps=float(training_config['eps'])
        ),
        [float(training_config['weight_decay']), 0]
    )

    if args.from_checkpoint:
        bot = resume_training(
            training_config, args.from_checkpoint, model,
            optimizer, train_loader, valid_loader)
    else:
        bot = train_from_start(
            training_config, model, optimizer,
            train_loader, valid_loader)
    target_dir = (MODEL_DIR / datetime.now().strftime("%Y%m%d_%H%M"))
    target_dir.mkdir(parents=True)
    torch.save(bot.model.state_dict(), target_dir / "model.pth")
    with open(target_dir / "config.yaml", "w") as fout:
        fout.write(yaml.dump(config, default_flow_style=False))


if __name__ == "__main__":
    main()
