import os
import argparse
from pathlib import Path
from datetime import datetime

import torch
# from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from helperbot import (
    WeightDecayOptimizerWrapper,
    LearningRateSchedulerCallback,
    MovingAverageStatsTrackerCallback,
    CheckpointCallback, MultiStageScheduler, LinearLR
)

from .models import (
    NeXtVLADModel, GatedDBoFModel, SampleFrameModelWrapper
)
from .segment_models import (
    ContextualSegmentModel, NeXtVLADEncoder,
    GatedDBofContextEncoder
)
from .loss import SampledCrossEntropyLoss
from .telegram_tokens import BOT_TOKEN, CHAT_ID
from .telegram_sender import telegram_sender
from .train_pure_segment import (
    YoutubeBot, get_loaders,  patch
)

CACHE_DIR = Path('./data/cache/segment/')
CACHE_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR = Path('./data/cache/segment/')
MODEL_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR_STR = './data/segment/'
NO_DECAY = ['bias', 'LayerNorm.weight', 'BatchNorm.weight']


def prepare_models(args):
    model_dir = Path(args.model_dir)
    context_model = patch(torch.load(str(model_dir / args.context_model)))
    if isinstance(context_model, SampleFrameModelWrapper):
        context_model = context_model.model
    segment_model = patch(torch.load(str(model_dir / args.segment_model)))
    if isinstance(segment_model, SampleFrameModelWrapper):
        segment_model = segment_model.model
    if isinstance(segment_model, NeXtVLADModel):
        segment_dim = segment_model.intermediate_fc[0].out_features
        segment_model = NeXtVLADEncoder(
            segment_model, vlad_only=False, truncate_intermediate=True)
    elif isinstance(segment_model, GatedDBoFModel):
        # segment_dim = segment_model.intermediate_fc[0].num_features
        # segment_model = GatedDBofEncoder(segment_model)
        segment_dim = segment_model.expert_fc[-1].in_features
        segment_model = GatedDBofContextEncoder(segment_model)
    else:
        raise ValueError("Model not supported yet!")
    if isinstance(context_model, NeXtVLADModel):
        context_dim = context_model.intermediate_fc[0].out_features
        context_model = NeXtVLADEncoder(
            context_model, vlad_only=False, truncate_intermediate=False)
    elif isinstance(context_model, GatedDBoFModel):
        context_dim = context_model.expert_fc[-1].in_features
        context_model = GatedDBofContextEncoder(context_model)
    else:
        raise ValueError("Model not supported yet!")
    return ContextualSegmentModel(
        context_model, segment_model, context_dim, segment_dim,
        args.fcn_dim, args.drop, se_reduction=args.se_reduction,
        max_video_len=args.max_len, train_context=args.finetune_context,
        num_mixtures=2
    ).cuda()


@telegram_sender(token=BOT_TOKEN, chat_id=CHAT_ID, name="Training on Segment")
def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('model_dir', type=str)
    arg('context_model', type=str)
    arg('segment_model', type=str)
    arg('--batch-size', type=int, default=32)
    arg('--lr', type=float, default=3e-4)
    arg('--steps', type=int, default=30000)
    arg('--offset', type=int, default=0)
    arg('--ckpt-interval', type=int, default=4000)
    arg('--fold', type=int, default=0)
    arg('--drop', type=float, default=0.5)
    arg('--fcn-dim', type=int, default=512)
    arg('--max-len', type=int, default=-1)
    arg('--se-reduction', type=int, default=0)
    arg('--finetune-context', action="store_true")
    arg('--name', type=str, default="model")
    args = parser.parse_args()

    train_loader, valid_loader = get_loaders(
        args, seed=int(os.environ.get("SEED", "9293")), offset=args.offset)

    model = prepare_models(args)
    print(model)
    # optimizer_grouped_parameters = []
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.segment_model.named_parameters()
                       if not any(nd in n for nd in NO_DECAY)],
            'lr': args.lr / 2
        },
        {
            'params': [p for n, p in model.segment_model.named_parameters()
                       if any(nd in n for nd in NO_DECAY)],
            'lr': args.lr / 2
        }
    ]
    if args.finetune_context:
        optimizer_grouped_parameters += [
            {
                'params': [p for n, p in model.context_model.named_parameters()
                           if not any(nd in n for nd in NO_DECAY)],
                'lr': args.lr / 4
            },
            {
                'params': [p for n, p in model.context_model.named_parameters()
                           if any(nd in n for nd in NO_DECAY)],
                'lr': args.lr / 4
            }
        ]
    for module in (model.expert_fc, model.gating_fc, model.intermediate_fc):
        optimizer_grouped_parameters += [
            {
                'params': [p for n, p in module.named_parameters()
                           if not any(nd in n for nd in NO_DECAY)],
                'lr': args.lr
            },
            {
                'params': [p for n, p in module.named_parameters()
                           if any(nd in n for nd in NO_DECAY)],
                'lr': args.lr
            }
        ]
    optimizer = WeightDecayOptimizerWrapper(
        torch.optim.Adam(optimizer_grouped_parameters, lr=args.lr),
        [0.02, 0] * (len(optimizer_grouped_parameters) // 2)
    )

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
