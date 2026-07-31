#!/usr/bin/env python
"""`<animal_id>/<foto>` yapısındaki ham bir veri kümesini train/val/test'e böler.

`scripts/train.py` ve `scripts/evaluate_baseline.py`'nin beklediği klasör yapısını
üretir (bkz. `app/services/image_service.py`). Kaynak klasör de AYNI yapıda olmalı
— yani zaten hayvan başına bir klasör altında toplanmış fotoğraflar olmalı; bu
script yalnızca bölme/kopyalama yapar, ham fotoğraf indirmez.

Bölme, her `animal_id` İÇİNDE ayrı ayrı yapılır (genel oranla değil): aksi halde
küçük veri kümelerinde bazı hayvanlar tamamen tek bir bölmede kalabilir. Her
hayvan için en az 1 fotoğraf train'de kalması garanti edilir; val/test oranları
yuvarlamadan sonra train'i sıfıra düşürürse önce val'den, sonra test'ten geri
çalınır (bkz. `_split_counts`).

Kullanım:
    python scripts/prepare_dataset.py \\
        --source-dir data/raw \\
        --output-dir data \\
        --val-ratio 0.15 \\
        --test-ratio 0.15 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.image_service import (  # noqa: E402
    DatasetScan,
    report_scan_issues,
    scan_animal_dataset,
)

MIN_TRAIN_IDENTITIES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="animal_id klasör yapısındaki ham veriyi train/val/test'e böler."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="train/val/test altında zaten dosya varsa önce onları siler. "
        "Verilmezse ve dosya varsa script hata verip durur (eski ve yeni verinin "
        "sessizce karışmasını önlemek için).",
    )
    return parser.parse_args()


def _split_counts(n: int, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    """`n` fotoğrafı (train, val, test) sayılarına böler; train her zaman >= 1 olur
    (n >= 1 iken). Yuvarlama train'i sıfıra düşürürse önce val'den, sonra test'ten
    geri alınır."""
    n_test = round(n * test_ratio)
    n_val = round(n * val_ratio)
    n_train = n - n_val - n_test

    eksik = 1 - n_train
    while eksik > 0 and (n_val > 0 or n_test > 0):
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
        eksik -= 1
    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def _copy_split(images: list, animal_id: str, split_name: str, output_dir: Path) -> None:
    dest_dir = output_dir / split_name / animal_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    for scanned in images:
        shutil.copy2(scanned.path, dest_dir / scanned.path.name)


def split_dataset(
    scan: DatasetScan, output_dir: Path, val_ratio: float, test_ratio: float, seed: int
) -> dict[str, int]:
    rng = random.Random(seed)
    by_animal: dict[str, list] = {}
    for img in scan.images:
        by_animal.setdefault(img.animal_id, []).append(img)

    counts = {"train": 0, "val": 0, "test": 0}
    single_image_animals = []
    for animal_id, images in sorted(by_animal.items()):
        images = images[:]
        rng.shuffle(images)
        n_train, n_val, n_test = _split_counts(len(images), val_ratio, test_ratio)
        if len(images) == 1:
            single_image_animals.append(animal_id)

        train_images = images[:n_train]
        val_images = images[n_train : n_train + n_val]
        test_images = images[n_train + n_val : n_train + n_val + n_test]

        _copy_split(train_images, animal_id, "train", output_dir)
        _copy_split(val_images, animal_id, "val", output_dir)
        _copy_split(test_images, animal_id, "test", output_dir)

        counts["train"] += len(train_images)
        counts["val"] += len(val_images)
        counts["test"] += len(test_images)

    if single_image_animals:
        print(
            f"Uyarı: {len(single_image_animals)} hayvanın tek fotoğrafı var, "
            f"hepsi train'e kondu (val/test'e düşen olmadı): "
            f"{', '.join(single_image_animals)}"
        )

    return counts


def _ensure_clean_output(output_dir: Path, overwrite: bool) -> None:
    """train/val/test altında dosya varsa eski ve yeni verinin sessizce karışmasını
    önler: `--overwrite` verilmediyse hata verip durur, verildiyse önce siler."""
    existing = [
        split_dir
        for split_dir in (output_dir / "train", output_dir / "val", output_dir / "test")
        if split_dir.exists() and any(split_dir.iterdir())
    ]
    if not existing:
        return
    if not overwrite:
        print(
            "Hata: Şu klasörlerde zaten dosya var: "
            f"{', '.join(str(d) for d in existing)}. "
            "Eski ve yeni veri sessizce karışmasın diye durduruldu. "
            "Farklı bir --output-dir seçin ya da --overwrite verin."
        )
        raise SystemExit(1)
    for split_dir in existing:
        shutil.rmtree(split_dir)


def main() -> None:
    args = parse_args()

    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        print("Hata: --val-ratio ve --test-ratio negatif olamaz, toplamları 1'den küçük olmalı.")
        raise SystemExit(1)

    _ensure_clean_output(args.output_dir, args.overwrite)

    try:
        scan = scan_animal_dataset(args.source_dir)
    except FileNotFoundError as exc:
        print(f"Hata: {exc}")
        raise SystemExit(1) from exc
    report_scan_issues(scan)

    animal_count = len({img.animal_id for img in scan.images})
    if animal_count < MIN_TRAIN_IDENTITIES:
        print(
            f"Hata: Bölme için en az {MIN_TRAIN_IDENTITIES} farklı hayvan gerekli, "
            f"'{args.source_dir}' içinde {animal_count} bulundu."
        )
        raise SystemExit(1)

    counts = split_dataset(scan, args.output_dir, args.val_ratio, args.test_ratio, args.seed)

    print()
    print(f"Bölme tamamlandı -> {args.output_dir}")
    print(f"  train : {counts['train']} fotoğraf")
    print(f"  val   : {counts['val']} fotoğraf")
    print(f"  test  : {counts['test']} fotoğraf")
    if counts["val"] == 0:
        print("  Uyarı: val bölmesi boş kaldı, train.py doğrulama metriği hesaplayamayacak.")
    if counts["test"] == 0:
        print("  Uyarı: test bölmesi boş kaldı, evaluate_baseline.py'ye veri kalmadı.")


if __name__ == "__main__":
    main()
