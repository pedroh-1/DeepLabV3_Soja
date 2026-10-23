import cv2
import albumentations as A
import matplotlib.pyplot as plt
import torch
from albumentations.pytorch import ToTensorV2

from src.model_factory import get_deeplab_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "weights/best_deeplab_soja.pth"
IMAGE_PATH = "data/test/images/IMG_20260322_132113.jpg"
DEFAULT_IMAGE_SIZE = 384
DEFAULT_THRESHOLD = 0.5
DEFAULT_BACKBONE = "mobilenet_v2"


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint, checkpoint["model_state_dict"]
    return {}, checkpoint


def predict():
    print(f"Carregando modelo no dispositivo: {DEVICE}...")
    metadata, state_dict = load_checkpoint(MODEL_PATH, DEVICE)
    image_size = int(metadata.get("image_size", DEFAULT_IMAGE_SIZE))
    threshold = float(metadata.get("best_threshold", DEFAULT_THRESHOLD))
    backbone = metadata.get("backbone", DEFAULT_BACKBONE)

    model = get_deeplab_model(encoder_name=backbone).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Erro: Arquivo {IMAGE_PATH} nao encontrado.")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = image_rgb.shape[:2]
    input_tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(DEVICE)

    print(f"Processando imagem em {image_size}x{image_size} com threshold {threshold:.2f}...")
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)
        mask = (probs > threshold).float().cpu().numpy()[0, 0]
        heatmap = probs.cpu().numpy()[0, 0]

    mask_resized = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(image_rgb)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Probabilidades")
    plt.imshow(heatmap, cmap="jet")
    plt.colorbar()
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"Segmentacao Final (Threshold {threshold:.2f})")
    plt.imshow(image_rgb)
    plt.imshow(mask_resized, alpha=0.5, cmap="Greens")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    predict()
