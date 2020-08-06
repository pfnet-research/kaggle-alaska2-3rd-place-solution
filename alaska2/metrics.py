import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics


# https://www.kaggle.com/anokas/weighted-auc-metric-updated
def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)

    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)

    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)
        if mask.sum() == 0:
            return np.nan

        x_padding = np.linspace(fpr[mask][-1], 1, 100)

        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min  # normalize such that curve starts at y=0
        score = metrics.auc(x, y)
        submetric = score * weight
        competition_metric += submetric

    return competition_metric / normalization


def alaska_weighted_auc_metric_fun(y_pred, y_true):
    y_pred = 1 - F.softmax(y_pred, dim=1).detach().numpy()[:, 0]
    y_true = (y_true.detach().numpy() != 0).astype(np.int32)
    return alaska_weighted_auc(y_true, y_pred)


def cross_entropy_loss(y_pred, y_true, reduction="mean"):
    if y_true.dtype == torch.float:
        loss = torch.sum(-y_true * F.log_softmax(y_pred, dim=1), dim=1)
        if reduction == "mean":
            return torch.mean(loss)
        elif reduction == "none":
            return loss
        else:
            raise ValueError
    else:
        return F.cross_entropy(y_pred, y_true, reduction=reduction)


def reduced_focal_loss(y_pred, y_true):
    ce = cross_entropy_loss(y_pred, y_true, reduction="none")
    pt = torch.exp(-ce)

    threshold = 0.5
    gamma = 2.0
    coef = ((1.0 - pt) / threshold).pow(gamma)
    coef[pt < threshold] = 1

    return torch.mean(coef * ce)
