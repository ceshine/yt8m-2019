import os
import argparse
from pathlib import Path
from datetime import datetime

import torch
# from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
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
from .train_video import create_video_model
from .train_pure_segment import YoutubeBot, get_loaders


CACHE_DIR = Path('./data/cache/segment/')
CACHE_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR = Path('./data/cache/segment/')
MODEL_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR_STR = './data/segment/'
NO_DECAY = ['bias', 'LayerNorm.weight', 'BatchNorm.weight']


def prepare_models(config, *, context_state_dict, segment_state_dict):
    # Restore the video model for the context encoder
    context_model = create_video_model(config["context_base"]["model"])
    if context_state_dict is not None:
        context_model.load_state_dict(context_state_dict)
    if isinstance(context_model, SampleFrameModelWrapper):
        context_model = context_model.model
    # Restore the video model for the segment encoder
    segment_model = create_video_model(config["segment_base"]["model"])
    if segment_state_dict is not None:
        segment_model.load_state_dict(segment_state_dict)
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
    model_config = config["segment_w_context"]["model"]
    return ContextualSegmentModel(
        context_model, segment_model, context_dim, segment_dim,
        model_config["fcn_dim"], model_config["p_drop"],
        se_reduction=model_config["se_reduction"],
        max_video_len=model_config["max_len"],
        train_context=model_config["finetune_context"],
        num_mixtures=model_config["n_mixture"]
    ).cuda()


@telegram_sender(token=BOT_TOKEN, chat_id=CHAT_ID, name="Training on Segment")
def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('config')
    arg('context_model_dir', type=str)
    arg('segment_model_dir', type=str)
    arg('--steps', type=int, default=-1)
    arg('--fold', type=int, default=0)
    arg('--name', type=str, default="context_model")
    args = parser.parse_args()
    with open(args.config) as fin:
        config = yaml.load(fin)
    training_config = config["segment_w_context"]["training"]
    train_loader, valid_loader = get_loaders(
        training_config["batch_size"], fold=args.fold,
        seed=int(os.environ.get("SEED", "9293")),
        offset=training_config["offset"])

    if args.steps > 0:
        # override
        training_config["steps"] = args.steps

    context_model_dir = Path(args.context_model_dir)
    with open(context_model_dir / "config.yaml") as fin:
        context_config = yaml.load(fin)
    config["context_base"] = context_config["video"]
    context_state_dict = torch.load(str(context_model_dir / "model.pth"))
    segment_model_dir = Path(args.segment_model_dir)
    with open(segment_model_dir / "config.yaml") as fin:
        segment_config = yaml.load(fin)
    config["segment_base"] = segment_config["video"]
    segment_state_dict = torch.load(str(segment_model_dir / "model.pth"))
    model = prepare_models(
        config,
        context_state_dict=context_state_dict,
        segment_state_dict=segment_state_dict)
    print(model)

    # optimizer_grouped_parameters = []
    lr = float(training_config["lr"])
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.segment_model.named_parameters()
                       if not any(nd in n for nd in NO_DECAY)],
            'lr': lr / 2
        },
        {
            'params': [p for n, p in model.segment_model.named_parameters()
                       if any(nd in n for nd in NO_DECAY)],
            'lr': lr / 2
        }
    ]
    if config["segment_w_context"]["model"]["finetune_context"]:
        optimizer_grouped_parameters += [
            {
                'params': [p for n, p in model.context_model.named_parameters()
                           if not any(nd in n for nd in NO_DECAY)],
                'lr': lr / 4
            },
            {
                'params': [p for n, p in model.context_model.named_parameters()
                           if any(nd in n for nd in NO_DECAY)],
                'lr': lr / 4
            }
        ]
    for module in (model.expert_fc, model.gating_fc, model.intermediate_fc):
        optimizer_grouped_parameters += [
            {
                'params': [p for n, p in module.named_parameters()
                           if not any(nd in n for nd in NO_DECAY)],
                'lr': lr
            },
            {
                'params': [p for n, p in module.named_parameters()
                           if any(nd in n for nd in NO_DECAY)],
                'lr': lr
            }
        ]
    optimizer = WeightDecayOptimizerWrapper(
        torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=lr, eps=float(training_config["eps"])),
        [training_config["weight_decay"], 0] * (
            len(optimizer_grouped_parameters) // 2)
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
