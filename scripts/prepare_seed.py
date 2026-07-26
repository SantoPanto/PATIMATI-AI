# scripts/prepare_seed.py
# Oxford-IIIT Pet Dataset'ten seed data üretir: 37 sınıf x 3 fotoğraf.
# Ayrıca siniflar.json yazar (class_NN -> ırk adı + tür) — cins doğruluk ölçümü
# bunu kullanır, bkz. scripts/measure_breed.py
# Çalıştır: python scripts/prepare_seed.py
import json
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

# --- Irk adı + tür eşlemesi -------------------------------------------------
# Tür bilgisi ham annotation dosyasından okunur: <dosya> <CLASS-ID> <SPECIES 1=Cat 2=Dog> <BREED-ID>
list_txt = Path("./tmp/oxford-iiit-pet/annotations/list.txt")
tur_by_class = {}
for satir in list_txt.read_text(encoding="utf-8").splitlines():
    if satir.startswith("#") or not satir.strip():
        continue
    _ad, class_id, species_id, _breed_id = satir.split()
    tur_by_class.setdefault(int(class_id) - 1, "cat" if species_id == "1" else "dog")

siniflar = {
    f"class_{label:02d}": {"breed": dataset.classes[label], "species": tur_by_class[label]}
    for label in sorted(by_class)
}
(out_dir / "siniflar.json").write_text(
    json.dumps(siniflar, ensure_ascii=False, indent=2), encoding="utf-8")

kedi = sum(1 for v in siniflar.values() if v["species"] == "cat")
print(f"Seed data hazır: {out_dir} ({len(by_class)} sınıf — {kedi} kedi, "
      f"{len(siniflar) - kedi} köpek)")
print(f"Irk eşlemesi yazıldı: {out_dir / 'siniflar.json'}")
