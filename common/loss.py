import torch.nn.functional as F
import numpy as np
import torch
from torch import Tensor

def focal_loss(inputs, targets, object_mask=None, alpha=-1, gamma=4, reduction="mean"):
    """
    Original code: https://github.com/apple/ml-destseg/blob/main/model/losses.py#L13
    """
    inputs = inputs.float()
    targets = targets.float()
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Apply the object mask if provided
    if object_mask is not None:
        object_mask = object_mask.repeat(inputs.size(0) // object_mask.size(0), 1, 1, 1)
        loss = loss * object_mask.float()

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def calc_loss(
        anomaly_map : Tensor,
        gt_mask : Tensor,
        th: float,
        score,
        label,
        object_mask : Tensor = None
):
    gt_mask = gt_mask.to("cpu")
    anomaly_map = anomaly_map.to("cpu")
    score = score.to('cpu')
    label = label.to('cpu')
    if object_mask is not None:
        object_mask = object_mask.to('cpu')

    normal_scores = anomaly_map[gt_mask == 0]
    anomalous_scores = anomaly_map[gt_mask > 0]
    true_loss = torch.clip(normal_scores + th, min=0)
    fake_loss = torch.clip(-anomalous_scores + th, min=0)

    if len(true_loss):
        true_loss = true_loss.mean()
    else:
        true_loss = 0
    if len(fake_loss):
        fake_loss = fake_loss.mean()
    else:
        fake_loss = 0

    pred_mask = anomaly_map.squeeze() >= th

    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()

    jaccard_index = intersection / union if union != 0 else 0
    jac_los = 1 - jaccard_index

    loss = (
        true_loss
        + fake_loss
        + focal_loss(torch.sigmoid(anomaly_map), gt_mask, object_mask)
        + focal_loss(torch.sigmoid(score), label, None)
        + jac_los
    )

    return loss