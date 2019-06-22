import warnings
from typing import Tuple

import torch
import numpy as np
from sklearn.metrics import fbeta_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning


class Metric:
    name = "metric"

    def __call__(self, truth: torch.Tensor, pred: torch.Tensor) -> Tuple[float, str]:
        raise NotImplementedError()


class FBeta(Metric):
    """FBeta for binary targets"""
    name = "fbeta"

    def __init__(self, step, beta=2, average="binary"):
        self.step = step
        self.beta = beta
        self.average = average

    def __call__(self, truth: torch.Tensor, pred: torch.Tensor) -> Tuple[float, str]:
        best_fbeta, best_thres = self.find_best_fbeta_threshold(
            truth.numpy(), torch.sigmoid(pred).numpy(),
            step=self.step, beta=self.beta)
        return best_fbeta * -1, f"{best_fbeta:.4f} @ {best_thres:.2f}"

    def find_best_fbeta_threshold(self, truth, probs, beta=2, step=0.05):
        best, best_thres = 0, -1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UndefinedMetricWarning)
            for thres in np.arange(step, 1, step):
                current = fbeta_score(
                    truth, (probs >= thres).astype("int8"),
                    beta=beta, average=self.average)
                if current > best:
                    best = current
                    best_thres = thres
        return best, best_thres


class AUC(Metric):
    """AUC for binary targets"""
    name = "auc"

    def __call__(self, truth: torch.Tensor, pred: torch.Tensor) -> Tuple[float, str]:
        auc_score = roc_auc_score(
            truth.numpy(), torch.sigmoid(pred).numpy())
        return auc_score * -1, f"{auc_score * 100:.2f}"


class Top1Accuracy(Metric):
    name = "accuracy"

    def __call__(self, truth: torch.Tensor, pred: torch.Tensor) -> Tuple[float, str]:
        correct = torch.sum(
            truth.view(-1) == torch.argmax(pred, dim=-1).view(-1)).item()
        total = truth.view(-1).size(0)
        accuracy = (correct / total)
        return accuracy * -1, f"{accuracy * 100:.2f}%"


class TopKAccuracy(Metric):
    def __init__(self, k=1):
        self.name = f"top_{k}_accuracy"
        self.k = k

    def __call__(self, truth: torch.Tensor, pred: torch.Tensor) -> Tuple[float, str]:
        with torch.no_grad():
            _, pred = pred.topk(self.k, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(
                truth.view(1, -1).expand_as(pred)
            ).view(-1).float().sum(0, keepdim=True)
            accuracy = correct.mul_(100.0 / truth.size(0)).item()
        return accuracy * -1, f"{accuracy:.2f}%"
