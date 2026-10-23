import json
import os
import random

import albumentations as A
import numpy as np
import torch
import wandb
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from src.dataset import SojaDataset
from src.losses import BCEDiceLoss
from src.model_factory import get_deeplab_model
from src.train import train_one_epoch, validate_one_epoch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
EPOCHS = int(os.getenv("EPOCHS", "100"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
LR = float(os.getenv("LR", "1e-4"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "384"))
BACKBONE = os.getenv("BACKBONE", "mobilenet_v2")
DEFAULT_THRESHOLD = 0.5
THRESHOLD_CANDIDATES = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
TRAIN_RATIO = 0.8
WEIGHTS_PATH = "weights/best_deeplab_soja.pth"
SPLIT_PATH = "splits/train_val_split_seed42.json"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def check_gpu():
    print(f"PyTorch version: {torch.__version__}")
    gpu_available = torch.cuda.is_available()
    print(f"GPU disponivel? {gpu_available}")

    if gpu_available:
        print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        if hasattr(props, "gcnArchName"):
            print(f"ID da arquitetura: {props.gcnArchName}")
    else:
        print("ALERTA: O codigo esta rodando na CPU!")


def build_transforms(image_size):
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    eval_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return train_transform, eval_transform


def build_or_load_split(images_dir, split_path, train_ratio, seed):
    image_names = sorted(os.listdir(images_dir))
    os.makedirs(os.path.dirname(split_path), exist_ok=True)

    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
        return split["train"], split["val"]

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(image_names), generator=generator).tolist()
    train_size = int(train_ratio * len(image_names))
    train_names = sorted(image_names[i] for i in permutation[:train_size])
    val_names = sorted(image_names[i] for i in permutation[train_size:])

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "train": train_names, "val": val_names}, f, indent=2)

    return train_names, val_names


def main():
    set_seed(SEED)
    check_gpu()
    os.makedirs("weights", exist_ok=True)

    wandb.init(
        project="soja-segmentation-deeplab",
        config={
            "learning_rate": LR,
            "architecture": "DeepLabV3+",
            "backbone": BACKBONE,
            "dataset": "Soybean_Leaves",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "image_size": IMAGE_SIZE,
            "seed": SEED,
            "loss": "0.5*BCEWithLogitsLoss + 0.5*DiceLoss",
            "default_threshold": DEFAULT_THRESHOLD,
            "threshold_candidates": THRESHOLD_CANDIDATES,
            "split_path": SPLIT_PATH,
        },
    )

    train_transform, val_transform = build_transforms(IMAGE_SIZE)
    train_names, val_names = build_or_load_split("data/images", SPLIT_PATH, TRAIN_RATIO, SEED)

    train_ds = SojaDataset("data/images", "data/masks", transform=train_transform, image_names=train_names)
    val_ds = SojaDataset("data/images", "data/masks", transform=val_transform, image_names=val_names)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    model = get_deeplab_model(encoder_name=BACKBONE).to(DEVICE)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_iou = 0.0
    best_threshold = DEFAULT_THRESHOLD
    for epoch in range(EPOCHS):
        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, threshold=DEFAULT_THRESHOLD
        )
        val_loss, val_iou, epoch_best_threshold, epoch_best_iou = validate_one_epoch(
            model,
            val_loader,
            criterion,
            DEVICE,
            threshold=DEFAULT_THRESHOLD,
            threshold_candidates=THRESHOLD_CANDIDATES,
        )
        scheduler.step(epoch_best_iou)

        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_iou_threshold_0_5": val_iou,
            "val_iou_best_threshold": epoch_best_iou,
            "best_threshold_epoch": epoch_best_threshold,
            "learning_rate": current_lr,
        })

        print(
            f"Epoch {epoch + 1} Finalizada. "
            f"IoU Val @0.5: {val_iou:.4f} | "
            f"Melhor IoU Val: {epoch_best_iou:.4f} @ threshold {epoch_best_threshold:.2f}"
        )

        if epoch_best_iou > best_iou:
            best_iou = epoch_best_iou
            best_threshold = epoch_best_threshold
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_iou": best_iou,
                "best_threshold": best_threshold,
                "image_size": IMAGE_SIZE,
                "backbone": BACKBONE,
                "seed": SEED,
                "split_path": SPLIT_PATH,
            }, WEIGHTS_PATH)

    wandb.summary["best_val_iou"] = best_iou
    wandb.summary["best_threshold"] = best_threshold
    wandb.finish()


if __name__ == "__main__":
    main()
