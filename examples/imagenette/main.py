import argparse
import json
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import torch
from torch import nn, cuda
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from helperbot import (
    BaseBot, WeightDecayOptimizerWrapper, TriangularLR,
    GradualWarmupScheduler, LearningRateSchedulerCallback,
    MixUpCallback, Top1Accuracy, TopKAccuracy,
    MovingAverageStatsTrackerCallback,
    CheckpointCallback, EarlyStoppingCallback
)
from helperbot.loss import MixUpSoftmaxLoss
from helperbot.lr_finder import LRFinder

from models import get_seresnet_model, get_densenet_model, get_efficientnet_model
from dataset import TrainDataset, N_CLASSES, DATA_ROOT, build_dataframe_from_folder
from transforms import train_transform, test_transform

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

CACHE_DIR = Path('./data/cache/')
CACHE_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR = Path('./data/cache/')
MODEL_DIR.mkdir(exist_ok=True, parents=True)

NO_DECAY = [
    'bias', 'bn1.weight', 'bn2.weight', 'bn3.weight'
]


def make_loader(args, ds_class, df: pd.DataFrame, image_transform, drop_last=False, shuffle=False) -> DataLoader:
    return DataLoader(
        ds_class(df, image_transform, debug=args.debug),
        shuffle=shuffle,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=drop_last
    )


@dataclass
class ImageClassificationBot(BaseBot):
    log_dir: Path = MODEL_DIR / "logs/"

    def __post_init__(self):
        super().__post_init__()
        self.loss_format = "%.6f"
        self.metrics = (Top1Accuracy(), TopKAccuracy(k=3))

    @staticmethod
    def extract_prediction(output):
        return output


def get_optimizer(model, lr):
    return WeightDecayOptimizerWrapper(
        torch.optim.Adam(
            [
                {
                    'params': [p for n, p in model.named_parameters()
                               if not any(nd in n for nd in NO_DECAY)],
                },
                {
                    'params': [p for n, p in model.named_parameters()
                               if any(nd in n for nd in NO_DECAY)],
                }
            ],
            weight_decay=0,
            lr=lr
        ),
        weight_decay=[1e-1, 0],
        change_with_lr=True
    )


def train_from_scratch(args, model, train_loader, valid_loader, criterion):
    n_steps = len(train_loader) * args.epochs
    optimizer = get_optimizer(model, args.lr)
    if args.debug:
        print(
            "No decay:",
            [n for n, p in model.named_parameters()
             if any(nd in n for nd in NO_DECAY)]
        )
    if args.amp:
        if not APEX_AVAILABLE:
            raise ValueError("Apex is not installed!")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.amp
        )

    checkpoints = CheckpointCallback(
        keep_n_checkpoints=1,
        checkpoint_dir=CACHE_DIR / "model_cache/",
        monitor_metric="accuracy"
    )
    callbacks = [
        MovingAverageStatsTrackerCallback(
            avg_window=len(train_loader) // 5,
            log_interval=len(train_loader) // 6
        ),
        LearningRateSchedulerCallback(
            # TriangularLR(
            #     optimizer, 100, ratio=4, steps_per_cycle=n_steps
            # )
            GradualWarmupScheduler(
                optimizer, 100, min(int(n_steps*0.25), len(train_loader)),
                after_scheduler=CosineAnnealingLR(
                    optimizer,
                    n_steps - min(int(n_steps*0.25), len(train_loader)),
                )
            )
        ),
        checkpoints,
        EarlyStoppingCallback(
            patience=6, min_improv=1e-2,
            monitor_metric="accuracy"
        )
    ]
    if args.mixup_alpha:
        callbacks.append(MixUpCallback(
            alpha=args.mixup_alpha, softmax_target=True))
    bot = ImageClassificationBot(
        model=model, train_loader=train_loader,
        val_loader=valid_loader, clip_grad=10.,
        optimizer=optimizer, echo=True,
        criterion=criterion,
        callbacks=callbacks,
        pbar=True, use_tensorboard=True,
        use_amp=(args.amp != '')
    )
    bot.train(
        n_steps,
        checkpoint_interval=len(train_loader) // 2
    )
    bot.load_model(checkpoints.best_performers[0][1])
    torch.save(bot.model.state_dict(), CACHE_DIR /
               f"final_weights.pth")
    checkpoints.remove_checkpoints(keep=0)


def find_lr(args, model, train_loader, criterion):
    n_steps = len(train_loader) * args.epochs
    optimizer = get_optimizer(model, args.lr)
    if args.amp:
        if not APEX_AVAILABLE:
            raise ValueError("Apex is not installed!")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.amp
        )
    finder = LRFinder(model, optimizer, criterion)
    finder.range_test(
        train_loader,
        min_lr_ratio=1e-4,
        total_steps=n_steps,
        ma_decay=0.9, stop_ratio=3
    )
    finder.plot(skip_start=int(n_steps*0.1), filepath="lr_find.png")
    print("Learning rate probing completed. Check `lr_find.png` for result.")


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--batch-size', type=int, default=32)
    arg('--lr', type=float, default=2e-3)
    arg('--workers', type=int, default=4)
    arg('--epochs', type=int, default=5)
    arg('--mixup-alpha', type=float, default=0)
    arg('--arch', type=str, default='seresnext50')
    arg('--amp', type=str, default='')
    arg('--debug', action='store_true')
    arg('--find-lr', action='store_true')
    args = parser.parse_args()

    train_dir = DATA_ROOT / 'train'
    valid_dir = DATA_ROOT / 'val'

    use_cuda = cuda.is_available()
    if args.arch == 'seresnext50':
        model = get_seresnet_model(
            arch="se_resnext50_32x4d",
            n_classes=N_CLASSES, pretrained=False)
    elif args.arch == 'seresnext101':
        model = get_seresnet_model(
            arch="se_resnext101_32x4d",
            n_classes=N_CLASSES, pretrained=False)
    elif args.arch.startswith("densenet"):
        model = get_densenet_model(arch=args.arch)
    elif args.arch.startswith("efficientnet"):
        model = get_efficientnet_model(
            arch=args.arch, pretrained=False)
    else:
        raise ValueError("No such model")
    if use_cuda:
        model = model.cuda()
    criterion = MixUpSoftmaxLoss(nn.CrossEntropyLoss())
    (CACHE_DIR / 'params.json').write_text(
        json.dumps(vars(args), indent=4, sort_keys=True))

    df_train, class_map = build_dataframe_from_folder(train_dir)
    df_valid = build_dataframe_from_folder(valid_dir, class_map)

    train_loader = make_loader(
        args, TrainDataset, df_train, train_transform, drop_last=True, shuffle=True)
    valid_loader = make_loader(
        args, TrainDataset, df_valid, test_transform, shuffle=False)

    print(f'{len(train_loader.dataset):,} items in train, '
          f'{len(valid_loader.dataset):,} in valid')
    if args.find_lr:
        find_lr(args, model, train_loader, criterion)
    else:
        train_from_scratch(args, model, train_loader, valid_loader, criterion)


if __name__ == '__main__':
    main()
