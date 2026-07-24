# scripts/prepare_seed.py
# Oxford-IIIT Pet Dataset'ten seed data üretir: 37 sınıf x 3 fotoğraf.
# Çalıştır: python scripts/prepare_seed.py
from pathlib import Path

from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

out_dir = Path("tests/seed_data")
out_dir.mkdir(parents=True, exist_ok=True)

transform = transforms.Compose([transforms.Resize((256, 256))])

dataset = OxfordIIITPet(root="./tmp", download=True, transform=transform)

# Her sınıftan 3 örnek al (37 sınıf x 3 = 111 fotoğraf)
by_class = {}
for img, label in dataset:
    label = int(label)
    if label not in by_class:
        by_class[label] = []
    if len(by_class[label]) < 3:
        by_class[label].append(img)
    if len(by_class) >= 37 and all(len(v) >= 3 for v in by_class.values()):
        break

for label, imgs in by_class.items():
    label_dir = out_dir / f"class_{label:02d}"
    label_dir.mkdir(exist_ok=True)
    for i, img in enumerate(imgs):
        img.save(label_dir / f"{i}.jpg")

print(f"Seed data hazır: {out_dir} ({len(by_class)} sınıf)")
