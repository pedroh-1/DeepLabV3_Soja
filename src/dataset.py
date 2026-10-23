import os
import cv2
import numpy as np
from torch.utils.data import Dataset
class SojaDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, image_names=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted(image_names) if image_names is not None else sorted(os.listdir(images_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        # Assume que a mascara tem o mesmo nome da imagem, mas extensao .png
        mask_path = os.path.join(self.masks_dir, os.path.splitext(self.images[idx])[0] + ".png")
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Imagem nao encontrada ou invalida: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mascara nao encontrada ou invalida: {mask_path}")
        
        # Normaliza a mascara para 0 e 1 se necessario
        mask = np.where(mask > 0, 1.0, 0.0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.unsqueeze(0)  # Adiciona canal de cor na mascara [1, H, W]
