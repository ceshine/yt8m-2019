import os
import glob
import argparse
from pathlib import Path
from typing import Tuple
from dataclasses import dataclass
from datetime import datetime

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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from .models import SampleFrameModelWrapper
from .segment_models import SegmentModelWrapper
from .dataloader import YoutubeSegmentDataset, DataLoader, collate_segments
from .loss import SampledCrossEntropyLoss
from .telegram_tokens import BOT_TOKEN, CHAT_ID
from .telegram_sender import telegram_sender
from .train_video import create_video_model

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


def get_loaders(batch_size, fold, seed=42, offset=0):
    kfold = KFold(n_splits=8, shuffle=True, random_state=42)
    file_paths = np.array(collect_file_paths())
    for i, (train_idx, valid_idx) in enumerate(kfold.split(file_paths)):
        if i == fold:
            train_ds = YoutubeSegmentDataset(
                file_paths[train_idx], epochs=None, offset=offset, seed=seed)
            train_loader = DataLoader(
                train_ds, num_workers=1, batch_size=batch_size, collate_fn=collate_segments)
            valid_ds = YoutubeSegmentDataset(
                file_paths[valid_idx], epochs=1, offset=offset)
            valid_loader = DataLoader(
                valid_ds, num_workers=1, batch_size=batch_size, collate_fn=collate_segments)
            return train_loader, valid_loader
    raise ValueError("Shouldn't have reached here! KFold settings are off.")


def prepare_models(config, state_dict=None):
    segment_model = create_video_model(config["video"]["model"])
    if state_dict is not None:
        segment_model.load_state_dict(state_dict)
    if isinstance(segment_model, SampleFrameModelWrapper):
        segment_model = segment_model.model
    return SegmentModelWrapper(segment_model)


@telegram_sender(token=BOT_TOKEN, chat_id=CHAT_ID, name="Training on Segment")
def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('config', type=str)
    arg('base_model_dir', type=str)
    arg('--steps', type=int, default=-1)
    arg('--fold', type=int, default=0)
    arg('--name', type=str, default="model")
    args = parser.parse_args()
    with open(args.config) as fin:
        config = yaml.load(fin)
    training_config = config["pure_segment"]["training"]
    train_loader, valid_loader = get_loaders(
        training_config["batch_size"], fold=args.fold,
        seed=int(os.environ.get("SEED", "9293")),
        offset=training_config["offset"])

    if args.steps > 0:
        # override
        training_config["steps"] = args.steps

    base_model_dir = Path(args.base_model_dir)
    with open(base_model_dir / "config.yaml") as fin:
        video_config = yaml.load(fin)
    config.update(video_config)
    state_dict = torch.load(str(base_model_dir / "model.pth"))
    model = prepare_models(config, state_dict=state_dict)

    print(model)
    lr = float(training_config["lr"])
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in NO_DECAY)],
            'lr': lr
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if any(nd in n for nd in NO_DECAY)],
            'lr': lr
        }
    ]
    optimizer = WeightDecayOptimizerWrapper(
        torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=lr, eps=float(training_config["eps"])),
        [training_config["weight_decay"], 0]
    )
    # optimizer = torch.optim.Adam(
    #     optimizer_grouped_parameters, lr=lr, eps=1e-7)

    n_steps = training_config["steps"]
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
        total_steps=n_steps, checkpoint_interval=training_config["ckpt_interval"]
    )
    bot.load_model(checkpoints.best_performers[0][1])
    checkpoints.remove_checkpoints(keep=0)

    # save the model
    target_dir = (MODEL_DIR /
                  f"{args.name}_{args.fold}_{datetime.now().strftime('%Y%m%d-%H%M')}")
    target_dir.mkdir(parents=True)
    torch.save(
        bot.model.state_dict(), target_dir / "model.pth"
    )
    with open(target_dir / "config.yaml", "w") as fout:
        fout.write(yaml.dump(config, default_flow_style=False))


if __name__ == "__main__":
    main()
