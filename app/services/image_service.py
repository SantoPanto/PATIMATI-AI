# app/services/image_service.py
"""scripts/train.py için eğitim veri klasörlerini tarar ve görüntüleri RGB olarak açar.

Beklenen klasör yapısı (`scan_animal_dataset`'e verilen kök altında):

    <train-dir>/
        <animal_id_1>/
            foto1.jpg
            foto2.png
        <animal_id_2>/
            foto1.jpg
        ...

Her alt klasör bir hayvanın kimliğidir (`animal_id`) ve klasör adı olarak
kullanılır; içindeki dosyalar o hayvana ait fotoğraflardır (tek seviye —
animal_id klasörünün altındaki alt klasörler taranmaz). Kök klasörün doğrudan
içindeki dosyalar (bir animal_id klasörüne ait olmayanlar) yoksayılır ve
`stray_files` altında raporlanır; gizli dosya/klasörler (adı "." ile başlayan)
tamamen atlanır.

`app.embedder.goruntu_ac` BİLEREK kullanılmıyor: o modül import edilir edilmez
üretim tekil nesnesini (`embedder = PetEmbedder()`) oluşturuyor ve `KIMLIK_MODEL`
ayarına göre CLIP (~400 MB) ya da SigLIP2 (~1,4 GB) ağırlıklarını indirip
belleğe yüklüyor. Sadece bir klasör taramak/görüntü açmak için bunu tetiklemek
yanlış bağlantı olurdu; bu yüzden `load_rgb_image` kendi asgari açma mantığını
taşıyor (EXIF döndürme, şeffaflık düzeltmesi, asgari boyut kontrolü). Tarama
sırasında yapılan kontrol ise ayrıca ucuz bir `Image.verify()`'dır; tam decode
yalnızca `load_rgb_image` çağrıldığında, yani eğitim gerçekten o görüntüyü
kullandığında yapılır.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

# Bundan küçük görüntüler anlamlı bir vektör üretmez — app/embedder.py'deki
# ASGARI_KENAR ile aynı eşik, bilerek aynı tutuluyor.
MIN_EDGE = 32

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ScannedImage:
    """Bir hayvana ait, geçerliliği doğrulanmış tek bir eğitim fotoğrafı."""

    path: Path
    animal_id: str


@dataclass
class DatasetScan:
    """`scan_animal_dataset` sonucu: geçerli görüntüler + atlanan her şeyin dökümü."""

    directory: Path
    images: list[ScannedImage] = field(default_factory=list)
    stray_files: list[Path] = field(default_factory=list)
    unreadable_files: list[Path] = field(default_factory=list)
    empty_animal_dirs: list[str] = field(default_factory=list)


def _is_probably_image(path: Path) -> bool:
    """Ucuz bir bozukluk kontrolü. `Image.verify()` tam decode yapmaz; asıl
    doğrulama (EXIF, şeffaflık, asgari boyut) `load_rgb_image` çağrıldığında olur."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def scan_animal_dataset(directory: Path) -> DatasetScan:
    """`directory` altındaki `<animal_id>/<foto>` yapısını tarar.

    Tanınmayan uzantılar, bozuk dosyalar ve hiç geçerli görüntüsü olmayan
    animal_id klasörleri hata fırlatmadan `DatasetScan` içinde raporlanır
    (bkz. `report_scan_issues`); yalnızca `directory`'nin kendisi yoksa/klasör
    değilse `FileNotFoundError` fırlatılır.
    """
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Veri klasörü bulunamadı: {directory}")

    scan = DatasetScan(directory=directory)

    for entry in sorted(directory.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_file():
            scan.stray_files.append(entry)
            continue
        if not entry.is_dir():
            continue

        animal_id = entry.name
        found_for_animal = False
        for candidate in sorted(entry.iterdir()):
            if candidate.name.startswith(".") or not candidate.is_file():
                continue
            if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
                scan.stray_files.append(candidate)
                continue
            if not _is_probably_image(candidate):
                scan.unreadable_files.append(candidate)
                continue
            scan.images.append(ScannedImage(path=candidate, animal_id=animal_id))
            found_for_animal = True

        if not found_for_animal:
            scan.empty_animal_dirs.append(animal_id)

    return scan


def report_scan_issues(scan: DatasetScan) -> None:
    """Taramada bulunan sorunları stdout'a yazar. Fatal karar burada verilmez —
    hangi eksikliğin eğitimi durdurması gerektiğine çağıran (train.py) karar verir."""
    animal_count = len({img.animal_id for img in scan.images})
    print(f"Tarama: {scan.directory} -> {len(scan.images)} görüntü, {animal_count} hayvan.")

    if scan.stray_files:
        print(
            f"Uyarı: {len(scan.stray_files)} dosya yoksayıldı (animal_id klasörü dışında "
            f"veya tanınmayan uzantı), ör: {scan.stray_files[0]}"
        )
    if scan.unreadable_files:
        print(
            f"Uyarı: {len(scan.unreadable_files)} dosya açılamadı (bozuk görüntü), "
            f"ör: {scan.unreadable_files[0]}"
        )
    if scan.empty_animal_dirs:
        print(
            f"Uyarı: {len(scan.empty_animal_dirs)} hayvan klasöründe geçerli görüntü yok: "
            f"{', '.join(scan.empty_animal_dirs)}"
        )


def load_rgb_image(path: Path) -> Image.Image:
    """Bir eğitim görüntüsünü analize hazır RGB'ye çevirir.

    `app.embedder.goruntu_ac` ile aynı üç adımı uygular (EXIF döndürme, şeffaflığı
    beyaz zemine yerleştirme, asgari boyut kontrolü) ama o modülü import etmez —
    bkz. dosya başındaki not. Modül taşınmadıkça bu iki yer elle senkron tutulmalı.
    """
    with Image.open(path) as img:
        img.load()
        img = ImageOps.exif_transpose(img)

        if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
            img = img.convert("RGBA")
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        img = img.convert("RGB")

    if min(img.size) < MIN_EDGE:
        raise ValueError(
            f"Görüntü çok küçük: {path} ({img.size[0]}x{img.size[1]}, "
            f"en az {MIN_EDGE}x{MIN_EDGE} olmalı)"
        )
    return img
