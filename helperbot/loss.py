import torch.nn as nn


class MixUpSoftmaxLoss(nn.Module):
    "Reference: https://github.com/fastai/fastai/blob/master/fastai/callbacks/mixup.py#L6"

    def __init__(self, crit, reduction='mean'):
        super().__init__()
        self.crit = crit
        setattr(self.crit, 'reduction', 'none')
        self.reduction = reduction

    def forward(self, output, target):
        if len(target.size()) == 2:
            loss1 = self.crit(output, target[:, 0].long())
            loss2 = self.crit(output, target[:, 1].long())
            lambda_ = target[:, 2]
            d = (loss1 * lambda_ + loss2 * (1-lambda_)).mean()
        else:
            # This handles the cases without MixUp for backward compatibility
            d = self.crit(output, target)
        if self.reduction == 'mean':
            return d.mean()
        elif self.reduction == 'sum':
            return d.sum()
        return d
