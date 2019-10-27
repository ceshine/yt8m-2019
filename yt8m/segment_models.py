import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import general_weight_initialization
from .encoders import SEModule, BNSEModule


def children(m):
    return m if isinstance(m, (list, tuple)) else list(m.children())


def set_trainable_attr(m, b):
    m.trainable = b
    for p in m.parameters():
        p.requires_grad = b


def apply_leaf(m, f):
    c = children(m)
    if isinstance(m, nn.Module):
        f(m)
    if len(c) > 0:
        for l in c:
            apply_leaf(l, f)


def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m, b))


class SegmentModelWrapper(nn.Module):
    def __init__(self, segment_model):
        super().__init__()
        self._model = segment_model

    def forward(self, video_features, video_masks, segment_features):
        return self._model(segment_features)


class ContextualSegmentModel(nn.Module):
    def __init__(
            self, context_model, segment_model, context_dim, segment_dim,
            fcn_dim, p_drop, n_classes=1000, num_mixtures=2, se_reduction=0,
            max_video_len=-1, train_context=False):
        super().__init__()
        self.context_model = context_model
        self.train_context = train_context
        if train_context is False:
            set_trainable(self.context_model, False)
        self.segment_model = segment_model
        self.num_mixtures = num_mixtures
        self.n_classes = n_classes
        self.p_drop = p_drop
        self.context_dim = context_dim
        self.segment_dim = segment_dim
        self.se_reduction = se_reduction
        assert (max_video_len > 10 or max_video_len < 0)
        self.max_video_len = max_video_len
        if se_reduction > 0:
            self.intermediate_fc = nn.Sequential(
                nn.BatchNorm1d(segment_dim + context_dim),
                nn.Dropout(p_drop),
                nn.Linear(
                    segment_dim + context_dim,
                    fcn_dim
                ),
                nn.ReLU(),
                BNSEModule(fcn_dim, se_reduction)
            )
        else:
            self.intermediate_fc = nn.Sequential(
                nn.BatchNorm1d(segment_dim + context_dim),
                nn.Dropout(p_drop),
                nn.Linear(
                    segment_dim + context_dim,
                    fcn_dim
                ),
                nn.ReLU()
            )
        self.expert_fc = nn.Sequential(
            nn.BatchNorm1d(fcn_dim),
            nn.Dropout(p_drop),
            nn.Linear(fcn_dim, n_classes * self.num_mixtures)
        )
        self.gating_fc = nn.Sequential(
            nn.BatchNorm1d(fcn_dim),
            nn.Dropout(p_drop),
            nn.Linear(fcn_dim, (self.num_mixtures + 1))
        )  # contains one gate for the dummy 'expert' (always predict none)
        self._init_weights()

    def _init_weights(self):
        for component in (self.intermediate_fc, self.expert_fc, self.gating_fc):
            for module in component.modules():
                general_weight_initialization(module)

    def forward(self, video_features, video_masks, segment_features):
        # truncate the video heads and tails
        video_features = video_features[:, 30:-10]
        video_masks = video_masks[:, 30:-10]
        if getattr(self, "max_video_len", -1) > 0:
            if video_features.size(1) > self.max_video_len:
                sampled_index = np.random.choice(
                    np.arange(video_features.size(1)),
                    self.max_video_len - 10
                )
                video_features = video_features[:, sampled_index]
                video_masks = video_masks[:, sampled_index]
        if ("train_context" not in self.__dict__) or self.train_context is False:
            with torch.no_grad():
                self.context_model.eval()
                # print(self.max_video_len, video_features.shape)
                # (batch, n_classes)
                context_vectors = self.context_model(
                    video_features, masks=video_masks
                ).detach()
        else:
            self.context_model.train()
            context_vectors = self.context_model(
                video_features, masks=video_masks
            )
        # (batch, segment_dim)
        segment_vectors = self.segment_model(segment_features)
        # shape (n_batch, fcn_dim)
        fcn_output = self.intermediate_fc(
            torch.cat([context_vectors, segment_vectors], dim=1)
        )
        # shape (n_batch, n_classes, num_mixtures)
        expert_logits = self.expert_fc(fcn_output).view(
            -1, self.n_classes, self.num_mixtures)
        expert_distributions = F.softmax(
            self.gating_fc(fcn_output), dim=-1
        ).unsqueeze(1)
        logits = (
            expert_logits * expert_distributions[..., :self.num_mixtures]
        ).sum(dim=-1)
        return logits


class GatedDBofEncoder(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.encoder = full_model.encoder
        self.pooler = full_model.pooler

    def forward(self, features, masks=None):
        """Encoder, Pool, Predit
            expected shape of 'features': (n_batch, seq_len, input_dim)
        """
        # shape (n_batch, hidden_dim, seq_len)
        encoded = self.encoder(features)
        pooled = self.pooler(encoded, masks=masks)
        return pooled


class GatedDBofContextEncoder(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.encoder = full_model.encoder
        self.intermediate_fc = full_model.intermediate_fc
        self.pooler = full_model.pooler

    def forward(self, features, masks=None):
        """Encoder, Pool, Predit
            expected shape of 'features': (n_batch, seq_len, input_dim)
        """
        # shape (n_batch, hidden_dim, seq_len)
        encoded = self.encoder(features)
        pooled = self.pooler(encoded, masks=masks)
        hidden = self.intermediate_fc(pooled)
        return hidden


class NeXtVLADEncoder(nn.Module):
    def __init__(self, full_model, vlad_only=False, truncate_intermediate=False):
        super().__init__()
        self.video_encoder = full_model.video_encoder
        self.audio_encoder = full_model.audio_encoder
        self.video_dim = full_model.video_dim
        self.audio_dim = full_model.audio_dim
        self.vlad_only = vlad_only
        if self.vlad_only is False:
            if truncate_intermediate:
                self.intermediate_fc = full_model.intermediate_fc[:1]
            else:
                self.intermediate_fc = full_model.intermediate_fc
            if "se_reduction" in full_model.__dict__ and truncate_intermediate is False:
                self.se_reduction = full_model.se_reduction
                if self.se_reduction > 0:
                    self.se_gating = full_model.se_gating

    def forward(self, features, masks=None):
        # shape (n_batch, dim * n_cluster)
        video_encoded = self.video_encoder(
            features[:, :, :self.video_dim], masks)
        audio_encoded = self.audio_encoder(
            features[:, :, self.video_dim:], masks)
        if self.vlad_only is False:
            # shape (n_batch, fcn_dim)
            fcn_output = self.intermediate_fc(
                torch.cat([video_encoded, audio_encoded], dim=1)
            )
            if "se_reduction" in self.__dict__ and self.se_reduction > 0:
                fcn_output = self.se_gating(fcn_output)
            return fcn_output
        else:
            return torch.cat([video_encoded, audio_encoded], dim=1)


class NeXtVLADCSModel(ContextualSegmentModel):
    def __init__(
            self, context_model, segment_model, fcn_dim, p_drop, n_classes=1000,
            num_mixtures=2, se_reduction=0, max_video_len=150):
        # segment_dim = (
        #     (segment_model.audio_encoder.num_clusters * segment_model.audio_dim +
        #      segment_model.video_encoder.num_clusters * segment_model.video_dim) *
        #     segment_model.expansion // segment_model.groups
        # )
        segment_dim = segment_model.intermediate_fc[0].out_features
        context_dim = context_model.intermediate_fc[0].out_features
        segment_model = NeXtVLADEncoder(
            segment_model, vlad_only=False, truncate_intermediate=True)
        # segment_model = NeXtVLADEncoder(
        #     segment_model, vlad_only=True, truncate_intermediate=False)
        context_model = NeXtVLADEncoder(context_model)
        super().__init__(
            context_model, segment_model, context_dim, segment_dim,
            fcn_dim, p_drop, n_classes,
            num_mixtures, se_reduction, max_video_len)
