import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .encoders import (
    NeXtVLAD, general_weight_initialization,
    BNSEModule, BNSE1dModule
)


class SampleFrameModelWrapper(nn.Module):
    def __init__(self, model, frame_starts=30, frame_ends=-10, max_len=-1):
        super().__init__()
        self.model = model
        self.frame_starts = frame_starts
        self.frame_ends = frame_ends
        self.max_len = max_len

    def forward(self, features, masks=None):
        # (batch, len, dim)
        features = features[:, self.frame_starts:self.frame_ends]
        if masks is not None:
            masks = masks[:, self.frame_starts:self.frame_ends]
        if getattr(self, "max_len", -1) > 0:
            if features.size(1) > self.max_len:
                sampled_index = np.random.choice(
                    np.arange(features.size(1)),
                    self.max_len - 10
                )
                features = features[:, sampled_index]
                if masks is not None:
                    masks = masks[:, sampled_index]
        return self.model(features, masks)


class BaseModel(nn.Module):
    def _init_weights(self):
        for module in self.modules():
            general_weight_initialization(module)


class AvgMaxPooler(nn.Module):
    """Average Pooling + Max Pooling

    Expected input shape: (n_batch, seq_len, hidden_dim)
    """

    def __init__(self):
        super().__init__()

    def forward(self, tensor, masks=None):
        if masks is not None:
            # max pooling:
            # shape (n_batch, hidden_dim):
            max_pooled, _ = torch.max(
                tensor * masks.unsqueeze(1),
                dim=2
            )
            # mean pooling:
            avg_pooled = torch.mean(
                tensor * masks.unsqueeze(1),
                dim=2
            )
        else:
            # shape (n_batch, hidden_dim)
            max_pooled = F.adaptive_max_pool1d(tensor, 1).squeeze(2)
            avg_pooled = F.adaptive_avg_pool1d(tensor, 1).squeeze(2)
        return torch.cat([max_pooled, avg_pooled], dim=1)


class TransposeLayer(nn.Module):
    def forward(self, tensor):
        return tensor.transpose(2, 1)


class GatedDBoFModel(BaseModel):
    def __init__(
            self, input_dim=1152, hidden_dim=1024, fcn_dim=1024,
            n_classes=1000, p_drop=0.25, num_mixtures=2, per_class=True,
            frame_se_reduction=0, video_se_reduction=0,
            class_se_reduction=0):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.n_classes = n_classes
        self.per_class = per_class
        self.fcn_dim = fcn_dim
        self.pooler = AvgMaxPooler()
        self.frame_se_reduction = frame_se_reduction
        self.video_se_reduction = video_se_reduction
        self.class_se_reduction = class_se_reduction
        if frame_se_reduction > 0:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                TransposeLayer(),
                BNSE1dModule(hidden_dim, frame_se_reduction)
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                TransposeLayer()
            )
        if video_se_reduction > 0:
            self.intermediate_fc = nn.Sequential(
                nn.BatchNorm1d(hidden_dim * 2),
                nn.Dropout(p_drop),
                nn.Linear(hidden_dim * 2, fcn_dim, bias=False),
                nn.ReLU(),
                BNSEModule(fcn_dim, video_se_reduction)
            )
        else:
            self.intermediate_fc = nn.Sequential(
                nn.BatchNorm1d(hidden_dim * 2),
                nn.Dropout(p_drop),
                nn.Linear(hidden_dim * 2, fcn_dim, bias=False),
                nn.ReLU(),
            )
        if class_se_reduction > 0:
            self.expert_fc = nn.Sequential(
                nn.BatchNorm1d(fcn_dim),
                nn.Dropout(p_drop),
                nn.Linear(fcn_dim, n_classes * self.num_mixtures),
                BNSEModule(n_classes * self.num_mixtures, class_se_reduction)
            )
        else:
            self.expert_fc = nn.Sequential(
                nn.BatchNorm1d(fcn_dim),
                nn.Dropout(p_drop),
                nn.Linear(fcn_dim, n_classes * self.num_mixtures)
            )
        if self.per_class:
            self.gating_fc = nn.Sequential(
                nn.BatchNorm1d(fcn_dim),
                nn.Dropout(p_drop),
                nn.Linear(fcn_dim, n_classes * (self.num_mixtures + 1))
            )  # contains one gate for the dummy 'export' (always predict none)
        else:
            self.gating_fc = nn.Sequential(
                nn.BatchNorm1d(fcn_dim),
                nn.Dropout(p_drop),
                nn.Linear(fcn_dim, (self.num_mixtures + 1))
            )  # contains one gate for the dummy 'export' (always predict none)
        self._init_weights()

    def forward(self, tensor, masks=None):
        """Encoder, Pool, Predict
            expected shape of 'features': (n_batch, seq_len, input_dim)
        """
        # shape (n_batch, seq_len, hidden_dim)
        encoded = self.encoder(tensor)
        # shape (n_batch, hidden_dim * 2)
        pooled = self.pooler(encoded, masks)
        # shape (n_batch, hidden_dim // 8)
        hidden = self.intermediate_fc(pooled)
        # shape (n_batch, n_classes, num_mixtures)
        expert_logits = self.expert_fc(hidden).view(
            -1, self.n_classes, self.num_mixtures)
        if self.per_class:
            expert_distributions = F.softmax(
                self.gating_fc(hidden).view(
                    -1, self.n_classes, self.num_mixtures + 1
                ), dim=-1
            )
        else:
            expert_distributions = F.softmax(
                self.gating_fc(hidden), dim=-1
            ).unsqueeze(1)
        logits = (
            expert_logits * expert_distributions[..., :self.num_mixtures]
        ).sum(dim=-1)
        return logits


class NeXtVLADModel(nn.Module):
    def __init__(
            self,  video_dim: int = 1024, audio_dim: int = 128, fcn_dim: int = 1024,
            n_clusters: int = 64, groups: int = 8, expansion: int = 2,
            n_classes: int = 1000, p_drop: int = 0.25, num_mixtures: int = 2,
            per_class: int = True, add_batchnorm: int = False, se_reduction: int = 0):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.n_classes = n_classes
        self.per_class = per_class
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.groups = groups
        self.expansion = expansion
        self.video_encoder = NeXtVLAD(
            num_clusters=n_clusters, dim=video_dim,
            groups=groups, expansion=expansion,
            alpha=100, p_drop=p_drop / 2,
            add_batchnorm=add_batchnorm)
        self.audio_encoder = NeXtVLAD(
            num_clusters=n_clusters // 2, dim=audio_dim,
            groups=groups, expansion=expansion,
            alpha=100, p_drop=p_drop / 2,
            add_batchnorm=add_batchnorm)
        self.intermediate_fc = nn.Sequential(
            nn.Linear(
                n_clusters * video_dim * expansion // groups +
                (n_clusters // 2) * audio_dim * expansion // groups,
                fcn_dim),
            nn.ReLU()
        )
        self.se_reduction = se_reduction
        if se_reduction > 0:
            self.se_gating = BNSEModule(fcn_dim, se_reduction)
        self.expert_fc = nn.Sequential(
            nn.BatchNorm1d(fcn_dim),
            nn.Dropout(p_drop),
            nn.Linear(fcn_dim, n_classes * self.num_mixtures)
        )
        if self.per_class:
            self.gating_fc = nn.Sequential(
                nn.BatchNorm1d(fcn_dim),
                nn.Dropout(p_drop),
                nn.Linear(fcn_dim, n_classes * (self.num_mixtures + 1))
            )  # contains one gate for the dummy 'expert' (always predict none)
        else:
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

    def forward(self, features, masks=None):
        """Encoder, Pool, Predit
            expected shape of 'features': (n_batch, 5, input_dim)
        """
        # shape (n_batch, dim * n_cluster)
        video_encoded = self.video_encoder(
            features[:, :, :self.video_dim], masks)
        audio_encoded = self.audio_encoder(
            features[:, :, self.video_dim:], masks)
        # shape (n_batch, fcn_dim)
        fcn_output = self.intermediate_fc(
            torch.cat([video_encoded, audio_encoded], dim=1)
        )
        if "se_reduction" in self.__dict__ and self.se_reduction > 0:
            fcn_output = self.se_gating(fcn_output)
        # shape (n_batch, n_classes, num_mixtures)
        expert_logits = self.expert_fc(fcn_output).view(
            -1, self.n_classes, self.num_mixtures)
        if self.per_class:
            expert_distributions = F.softmax(
                self.gating_fc(fcn_output).view(
                    -1, self.n_classes, self.num_mixtures + 1
                ), dim=-1
            )
        else:
            expert_distributions = F.softmax(
                self.gating_fc(fcn_output), dim=-1
            ).unsqueeze(1)
        logits = (
            expert_logits * expert_distributions[..., :self.num_mixtures]
        ).sum(dim=-1)
        return logits


def test_nextvlad():
    model = NeXtVLADModel(
        n_clusters=64, video_dim=128, audio_dim=16,
        groups=8, p_drop=0.5,
        num_mixtures=2, per_class=False,
        add_batchnorm=True, n_classes=1000
    )
    # shape (n_batch, len, dim)
    input_tensor = torch.rand(16, 300, 128+16)
    # shape (n_batch, 1000)
    output_tensor = model(input_tensor)
    assert output_tensor.size() == (16, 1000)


if __name__ == "__main__":
    test_nextvlad()
