import torch
import torch.nn as nn
import torch.nn.functional as F


class SampledCrossEntropyLoss(nn.Module):
    def forward(self, x, y):
        labels, positives = y[:, 0], y[:, 1].float()  # .clamp(1e-5, 1-1e-5)
        preds = torch.gather(
            x, 1, labels.unsqueeze(1)
        ).squeeze(1)
        positive_loss = F.binary_cross_entropy_with_logits(
            preds,
            positives,
            weight=positives+0.5
        )
        mask = y[:, 2:].float()
        negative_loss = (
            F.binary_cross_entropy_with_logits(
                x, torch.zeros_like(x), reduction="none"
            ) * mask
        ).sum() / mask.sum()
        # print("positive loss:", positive_loss)
        # print("negative loss:", negative_loss)
        return positive_loss + negative_loss * 0.5
