import os
import glob
import argparse
from pathlib import Path
from typing import Tuple
from dataclasses import dataclass
from datetime import datetime

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
from sklearn.model_selection import KFold

from .models import SampleFrameModelWrapper
from .segment_models import SegmentModelWrapper
from .dataloader import YoutubeSegmentDataset, DataLoader, collate_segments
from .loss import SampledCrossEntropyLoss
from .telegram_tokens import BOT_TOKEN, CHAT_ID
from .telegram_sender import telegram_sender
from .encoders import TimeFirstBatchNorm1d

CACHE_DIR = Path('./data/cache/segment/')
CACHE_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR = Path('./data/cache/segment/')
MODEL_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR_STR = './data/segment/'
NO_DECAY = ['bias', 'LayerNorm.weight', 'BatchNorm.weight']


class Accuracy(Metric):
    name = "accuracy"

    def __init__(self, cutoff=0.5):
        super().__init__()
        self.cutoff = cutoff

    def __call__(self, truth: torch.Tensor, pred: torch.Tensor) -> Tuple[float, str]:
        labels, true_positives = truth[:, 0], truth[:, 1]
        probs = torch.sigmoid(pred)
        pred_positives = (
            torch.gather(probs, 1, labels.unsqueeze(1)).squeeze(1)
            > self.cutoff).long()
        correct = torch.sum(
            pred_positives == true_positives
        ).item()
        total = pred_positives.size(0)
        accuracy = (correct / total)
        return accuracy * -1, f"{accuracy * 100:.2f}%"


class AUC(Metric):
    name = "roc_auc"

    def __call__(self, truth: torch.Tensor, pred: torch.Tensor) -> Tuple[float, str]:
        labels, true_positives = truth[:, 0], truth[:, 1]
        probs = torch.sigmoid(pred)
        pred_positives = (
            torch.gather(probs, 1, labels.unsqueeze(1)).squeeze(1)
        ).numpy()
        auc_score = roc_auc_score(
            true_positives.numpy(), pred_positives
        )
        return auc_score * -1, f"{auc_score * 100:.2f}"


class MaskedNegativeAccuracy(Metric):
    name = "mask_accuracy"

    def __init__(self, cutoff=0.5):
        super().__init__()
        self.cutoff = cutoff

    def __call__(self, truth: torch.Tensor, pred: torch.Tensor) -> Tuple[float, str]:
        mask = truth[:, 2:]
        probs = torch.sigmoid(pred)
        correct = torch.sum(
            (probs <= self.cutoff).long() * mask
        ).float().item()
        total = mask.sum().float().item()
        # print(probs.size(), mask.size(), mask.max())
        # print(correct, total)
        accuracy = (correct / total)
        return accuracy * -1, f"{accuracy * 100:.2f}%"


@dataclass
class YoutubeBot(BaseBot):
    checkpoint_dir: Path = CACHE_DIR / "model_cache/"
    log_dir: Path = MODEL_DIR / "logs/"

    def __post_init__(self):
        super().__post_init__()
        self.loss_format = "%.6f"
        self.metrics = (AUC(), Accuracy(), MaskedNegativeAccuracy())

    def extract_prediction(self, x):
        return x


def collect_file_paths():
    return list(glob.glob(str(DATA_DIR_STR + "train/*.tfrecord")))


def get_loaders(args, seed=42, offset=0):
    kfold = KFold(n_splits=8, shuffle=True, random_state=42)
    file_paths = np.array(collect_file_paths())
    for i, (train_idx, valid_idx) in enumerate(kfold.split(file_paths)):
        if i == args.fold:
            train_ds = YoutubeSegmentDataset(
                file_paths[train_idx], epochs=None, offset=offset, seed=seed)
            train_loader = DataLoader(
                train_ds, num_workers=1, batch_size=args.batch_size, collate_fn=collate_segments)
            valid_ds = YoutubeSegmentDataset(
                file_paths[valid_idx], epochs=1, offset=offset)
            valid_loader = DataLoader(
                valid_ds, num_workers=1, batch_size=args.batch_size, collate_fn=collate_segments)
            return train_loader, valid_loader
    raise ValueError("Shouldn't have reached here! KFold settings are off.")


def patch(model):
    for module in model.modules():
        if isinstance(module, TimeFirstBatchNorm1d):
            if "groups" not in module.__dict__:
                module.groups = None
    return model


def prepare_models(args):
    model_dir = Path(args.model_dir)
    segment_model = patch(torch.load(str(model_dir / args.segment_model)))
    if isinstance(segment_model, SampleFrameModelWrapper):
        segment_model = segment_model.model
    return SegmentModelWrapper(segment_model)


@telegram_sender(token=BOT_TOKEN, chat_id=CHAT_ID, name="Training on Segment")
def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('model_dir', type=str)
    arg('segment_model', type=str)
    arg('--batch-size', type=int, default=32)
    arg('--lr', type=float, default=3e-4)
    arg('--steps', type=int, default=30000)
    arg('--offset', type=int, default=0)
    arg('--ckpt-interval', type=int, default=4000)
    arg('--fold', type=int, default=0)
    arg('--name', type=str, default="model")
    args = parser.parse_args()

    train_loader, valid_loader = get_loaders(
        args, seed=int(os.environ.get("SEED", "9293")), offset=args.offset)

    model = prepare_models(args)
    print(model)
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in NO_DECAY)],
            'lr': args.lr
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if any(nd in n for nd in NO_DECAY)],
            'lr': args.lr
        }
    ]
    optimizer = WeightDecayOptimizerWrapper(
        torch.optim.Adam(optimizer_grouped_parameters, lr=args.lr, eps=1e-7),
        [0.02, 0]
    )
    # optimizer = torch.optim.Adam(
    #     optimizer_grouped_parameters, lr=args.lr, eps=1e-7)

    n_steps = args.steps
    checkpoints = CheckpointCallback(
        keep_n_checkpoints=1,
        checkpoint_dir=CACHE_DIR / "model_cache/",
        monitor_metric="roc_auc"
    )
    break_points = [0, int(n_steps*0.25)]
    lr_durations = np.diff(break_points + [n_steps])
    bot = YoutubeBot(
        model=model, train_loader=train_loader,
        valid_loader=valid_loader, clip_grad=10.,
        optimizer=optimizer, echo=True,
        criterion=SampledCrossEntropyLoss(),
        callbacks=[
            LearningRateSchedulerCallback(
                MultiStageScheduler(
                    [
                        LinearLR(optimizer, 0.01, lr_durations[0]),
                        LinearLR(
                            optimizer, 0.001,
                            lr_durations[1], upward=False)
                        # CosineAnnealingLR(optimizer, lr_durations[1])
                    ],
                    start_at_epochs=break_points
                )
            ),
            MovingAverageStatsTrackerCallback(
                avg_window=1200,
                log_interval=1000,
            ),
            checkpoints,
        ],
        pbar=True, use_tensorboard=False
    )
    bot.train(
        total_steps=n_steps, checkpoint_interval=args.ckpt_interval
    )
    bot.load_model(checkpoints.best_performers[0][1])
    checkpoints.remove_checkpoints(keep=0)

    torch.save(
        bot.model, MODEL_DIR /
        f"{args.name}_{args.fold}_{datetime.now().strftime('%Y%m%d-%H%M')}.pth"
    )


if __name__ == "__main__":
    main()
