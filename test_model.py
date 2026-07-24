from pathlib import Path

import timm
import torch
from PIL import Image
from torchvision import transforms

IMAGE_PATH = Path("test.jpg")
MODEL_NAME = "hf-hub:BVRA/MegaDescriptor-S-224"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError("Proje klasörüne test.jpg isimli bir hayvan fotoğrafı koy.")

    device = get_device()
    print(f"Kullanılan cihaz: {device}")

    model = timm.create_model(MODEL_NAME, pretrained=True)
    model = model.eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

    image = Image.open(IMAGE_PATH).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        embedding = model(tensor)
        embedding = torch.nn.functional.normalize(embedding, dim=1)

    print(f"Embedding boyutu: {embedding.shape}")
    print("Model başarıyla çalıştı.")


if __name__ == "__main__":
    main()
