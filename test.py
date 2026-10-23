import os

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SojaDataset
from src.model_factory import get_deeplab_model


MODEL_PATH = "weights/best_deeplab_soja.pth"
TEST_IMGS_DIR = "data/test/images"
TEST_MASKS_DIR = "data/test/masks"
DEFAULT_IMAGE_SIZE = 384
DEFAULT_THRESHOLD = 0.5
DEFAULT_BACKBONE = "mobilenet_v2"


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint, checkpoint["model_state_dict"]
    return {}, checkpoint


def calculate_metrics(mask_pred, mask_true, threshold):
    pred = (mask_pred > threshold).float().detach()
    target = (mask_true > 0.5).float().detach()

    intersection = torch.sum(pred * target).item()
    pred_sum = torch.sum(pred).item()
    target_sum = torch.sum(target).item()
    union = pred_sum + target_sum - intersection

    if union == 0:
        return 1.0, 1.0

    iou = intersection / (union + 1e-8)
    dice = (2 * intersection) / (pred_sum + target_sum + 1e-8)
    return iou, dice


def run_test():
    print("Iniciando teste oficial do DeepLabV3+...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("resultados_finais", exist_ok=True)

    metadata, state_dict = load_checkpoint(MODEL_PATH, device)
    image_size = int(metadata.get("image_size", DEFAULT_IMAGE_SIZE))
    threshold = float(metadata.get("best_threshold", DEFAULT_THRESHOLD))
    backbone = metadata.get("backbone", DEFAULT_BACKBONE)

    print(f"Carregando pesos de: {MODEL_PATH}")
    print(f"Backbone: {backbone} | Resolucao: {image_size} | Threshold: {threshold:.2f}")

    model = get_deeplab_model(encoder_name=backbone).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    test_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    test_set = SojaDataset(TEST_IMGS_DIR, TEST_MASKS_DIR, transform=test_transform)
    loader = DataLoader(test_set, batch_size=1, shuffle=False)

    ious, dices = [], []
    print(f"Avaliando {len(test_set)} imagens de teste...")
    with torch.no_grad():
        for images, masks in tqdm(loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            mask_pred = torch.sigmoid(outputs)

            iou, dice = calculate_metrics(mask_pred, masks, threshold=threshold)
            ious.append(iou)
            dices.append(dice)

    mean_iou = float(np.mean(ious))
    mean_dice = float(np.mean(dices))

    print("\n" + "=" * 40)
    print("RESULTADOS FINAIS DO TESTE")
    print("=" * 40)
    print(f"Media IoU:  {mean_iou:.4f} ({mean_iou * 100:.2f}%)")
    print(f"Media Dice: {mean_dice:.4f} ({mean_dice * 100:.2f}%)")
    print("=" * 40)

    plt.figure(figsize=(8, 6))
    metrics = ["IoU", "Dice (F1-Score)"]
    values = [mean_iou, mean_dice]
    bars = plt.bar(metrics, values, color=["#2ca02c", "#d62728"])
    plt.ylim(0, 1.05)
    plt.title("Desempenho Geral no Conjunto de Teste")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{yval:.4f}", ha="center", fontweight="bold")
    plt.savefig("resultados_finais/metricas_medias.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(ious, label="IoU por Imagem", color="green", marker="o", alpha=0.6)
    plt.axhline(y=mean_iou, color="black", linestyle="--", label=f"Media ({mean_iou:.2f})")
    plt.xlabel("ID da Imagem")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Variabilidade do Acerto por Imagem")
    plt.savefig("resultados_finais/dispersao_testes.png")
    plt.close()

    print("Teste finalizado. Resultados salvos em resultados_finais/.")


if __name__ == "__main__":
    run_test()
