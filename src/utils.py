import torch


def calculate_iou(pred, target, threshold=0.5, eps=1e-7):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    target = (target > 0.5).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    if union.item() == 0:
        return 1.0
    return ((intersection + eps) / (union + eps)).item()
