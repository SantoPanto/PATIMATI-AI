"""Görsel dosyalarını keşfetme ve güvenli biçimde yükleme yardımcıları."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})


class ImageLoadError(RuntimeError):
    """Bir görsel dosyası açılamadığında veya bozuk olduğunda yükseltilir."""


def is_supported_extension(path: Path) -> bool:
    """Dosya uzantısının desteklenen görsel formatlarından biri olup olmadığını döndürür."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def load_rgb_image(path: Path) -> Image.Image:
    """Bir görseli diskten güvenli biçimde yükleyip RGB moduna çevirir.

    Dosya bulunamazsa veya bozuksa anlaşılır bir `ImageLoadError` yükseltilir.
    """
    if not path.exists():
        raise ImageLoadError(f"Görsel bulunamadı: {path}")
    try:
        with Image.open(path) as raw:
            raw.load()
            return raw.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(f"Görsel okunamadı ({path.name}): {exc}") from exc


@dataclass(frozen=True)
class ScannedImage:
    """`root/animal_id/dosya.jpg` yapısında taranmış tek bir görsel kaydı."""

    path: Path
    animal_id: str


@dataclass
class DatasetScanResult:
    """`scan_animal_dataset` tarafından üretilen tarama sonucu."""

    images: list[ScannedImage] = field(default_factory=list)
    unreadable: list[tuple[Path, str]] = field(default_factory=list)
    unsupported: list[Path] = field(default_factory=list)

    @property
    def animal_ids(self) -> list[str]:
        return sorted({image.animal_id for image in self.images})

    def counts_per_animal(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for image in self.images:
            counts[image.animal_id] = counts.get(image.animal_id, 0) + 1
        return counts


def infer_species(animal_id: str) -> str | None:
    """Klasör adından basit bir sezgiyle tür çıkarımı yapar; emin olunamıyorsa None döner."""
    lowered = animal_id.lower()
    if lowered.startswith(("kedi", "cat")):
        return "cat"
    if lowered.startswith(("kopek", "köpek", "dog")):
        return "dog"
    return None


def report_scan_issues(scan: DatasetScanResult) -> None:
    """Tarama sırasında bulunan bozuk/desteklenmeyen dosyaları stdout'a raporlar."""
    if scan.unreadable:
        print(f"Uyarı: {len(scan.unreadable)} bozuk/okunamayan dosya atlandı:")
        for path, reason in scan.unreadable:
            print(f"  {path}: {reason}")
    if scan.unsupported:
        print(f"Uyarı: {len(scan.unsupported)} desteklenmeyen formatlı dosya atlandı:")
        for path in scan.unsupported:
            print(f"  {path}")


def scan_animal_dataset(root: Path, verify: bool = True) -> DatasetScanResult:
    """`root/animal_id/*.jpg` yapısındaki bir veri klasörünü tarar.

    Her alt klasör bir hayvanı (`animal_id`) temsil eder. Bozuk veya desteklenmeyen
    dosyalar ayrı listelerde raporlanır; tek bir hatalı dosya taramayı durdurmaz.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Veri klasörü bulunamadı: {root}")

    result = DatasetScanResult()
    for animal_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for file_path in sorted(animal_dir.iterdir()):
            if not file_path.is_file():
                continue
            if not is_supported_extension(file_path):
                result.unsupported.append(file_path)
                continue
            if verify:
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    result.unreadable.append((file_path, str(exc)))
                    continue
            result.images.append(ScannedImage(path=file_path, animal_id=animal_dir.name))
    return result
