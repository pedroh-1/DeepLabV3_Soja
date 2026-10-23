import torch
from tqdm import tqdm
from src.utils import calculate_iou


def _amp_context(device):
    enabled = str(device).startswith("cuda") and torch.cuda.is_available()
    return torch.amp.autocast("cuda", enabled=enabled)


def train_one_epoch(model, loader, optimizer, criterion, device, threshold=0.5):
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda") and torch.cuda.is_available())
    running_loss = 0.0
    total_iou = 0.0

    loop = tqdm(loader, desc="Treinando", leave=False)
    for images, masks in loop:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        total_iou += calculate_iou(outputs.detach(), masks, threshold=threshold)
        loop.set_postfix(loss=loss.item())

    return running_loss / len(loader), total_iou / len(loader)


def validate_one_epoch(model, loader, criterion, device, threshold=0.5, threshold_candidates=None):
    model.eval()
    running_loss = 0.0
    total_iou = 0.0
    threshold_candidates = threshold_candidates or [threshold]
    threshold_totals = {candidate: 0.0 for candidate in threshold_candidates}

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            with _amp_context(device):
                outputs = model(images)
                loss = criterion(outputs, masks)

            running_loss += loss.item()
            total_iou += calculate_iou(outputs, masks, threshold=threshold)
            for candidate in threshold_candidates:
                threshold_totals[candidate] += calculate_iou(outputs, masks, threshold=candidate)

    batches = len(loader)
    threshold_scores = {candidate: score / batches for candidate, score in threshold_totals.items()}
    best_threshold, best_threshold_iou = max(threshold_scores.items(), key=lambda item: item[1])

    return running_loss / batches, total_iou / batches, best_threshold, best_threshold_iou
