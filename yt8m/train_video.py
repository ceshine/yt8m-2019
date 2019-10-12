import os
import glob
import argparse
from pathlib import Path
from typing import Tuple
from dataclasses import dataclass

import torch
# from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from helperbot import (
    BaseBot, WeightDecayOptimizerWrapper,
    LearningRateSchedulerCallback,
    MovingAverageStatsTrackerCallback,
    CheckpointCallback, MultiStageScheduler, LinearLR
)
from helperbot.metrics import Metric
from sklearn.metrics import roc_auc_score

from .dataloader import YoutubeVideoDataset, DataLoader, collate_videos
from .models import BasicMoeModel, NetVladModel, NeXtVLADModel, DBoFModel, SampleFrameModelWrapper
from .telegram_tokens import BOT_TOKEN, CHAT_ID
from .telegram_sender import telegram_sender

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

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


def get_loaders(args):
    file_paths = collect_file_paths()
    train_ds = YoutubeVideoDataset(
        file_paths, epochs=None, seed=int(os.environ.get("SEED", 42)))
    train_loader = DataLoader(
        train_ds, num_workers=4, batch_size=args.batch_size, collate_fn=collate_videos)
    file_paths = collect_file_paths(base_folder="valid")
    valid_ds = YoutubeVideoDataset(
        file_paths, epochs=None, max_examples=32*4000/4)
    valid_loader = DataLoader(
        valid_ds, num_workers=4,
        batch_size=args.batch_size, collate_fn=collate_videos)
    return train_loader, valid_loader


def resume_training(args, model, optimizer, train_loader, valid_loader):
    bot = YoutubeVideoBot.load_checkpoint(
        args.from_checkpoint, train_loader, valid_loader,
        model, optimizer
    )
    checkpoints = None
    for callback in bot.callbacks:
        if isinstance(callback, CheckpointCallback):
            checkpoints = callback
            break
    # We could reset the checkpoints
    checkpoints.reset(ignore_previous=True)
    bot.train(checkpoint_interval=args.ckpt_interval)
    if checkpoints:
        bot.load_model(checkpoints.best_performers[0][1])
        torch.save(bot.model, MODEL_DIR / "baseline_model.pth")
        checkpoints.remove_checkpoints(keep=0)


def train_from_start(args, model, optimizer, train_loader, valid_loader):
    n_steps = args.steps
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
        pbar=True, use_tensorboard=False,
        use_amp=(args.amp != '')
    )
    bot.train(
        total_steps=n_steps, checkpoint_interval=args.ckpt_interval
    )
    bot.load_model(checkpoints.best_performers[0][1])
    torch.save(bot.model, MODEL_DIR / "baseline_model.pth")
    checkpoints.remove_checkpoints(keep=0)


@telegram_sender(token=BOT_TOKEN, chat_id=CHAT_ID, name="Training on Video")
def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--lr', type=float, default=5e-4)
    arg('--steps', type=int, default=80000)
    arg('--per-class', action="store_true")
    arg('--ckpt-interval', type=int, default=8000)
    arg('--from-checkpoint', type=str, default='')
    arg('--n-clusters', type=int, default=64)
    arg('--groups', type=int, default=8)
    arg('--amp', type=str, default='')
    arg('--batch-size', type=int, default=32)
    arg('--max-len', type=int, default=-1)
    args = parser.parse_args()
    train_loader, valid_loader = get_loaders(args)

    # model = BasicPoolingModel(hidden_dim=2048, p_drop=0.).cuda()
    # model = BasicMoeModel(
    #     hidden_dim=8192, p_drop=0.5,
    #     num_mixtures=4, per_class=False, se_reduction=4
    # ).cuda()
    model = SampleFrameModelWrapper(
        DBoFModel(
            hidden_dim=4096, p_drop=0.5, fcn_dim=2048,
            num_mixtures=4, per_class=False,
            frame_se_reduction=16, video_se_reduction=4,
        ).cuda(),
        max_len=args.max_len
    )
    # model = NetVladModel(
    #     fcn_dim=2048, p_drop=0.5,
    #     n_clusters=args.n_clusters,
    #     num_mixtures=2, per_class=args.per_class,
    #     add_batchnorm=True
    # ).cuda()
    # model = SampleFrameModelWrapper(
    #     NeXtVLADModel(
    #         fcn_dim=2048, p_drop=0.5,
    #         groups=args.groups, expansion=2,
    #         n_clusters=args.n_clusters,
    #         num_mixtures=4, per_class=args.per_class,
    #         add_batchnorm=True, se_reduction=4
    #     ).cuda(),
    #     max_len=args.max_len
    # )
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
        torch.optim.Adam(optimizer_grouped_parameters, lr=args.lr, eps=1e-6),
        0.1
    )
    if args.amp:
        if not APEX_AVAILABLE:
            raise ValueError("Apex is not installed!")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.amp
        )

    if args.from_checkpoint:
        resume_training(args, model, optimizer, train_loader, valid_loader)
    else:
        train_from_start(args, model, optimizer, train_loader, valid_loader)


if __name__ == "__main__":
    main()
