#!/usr/bin/env python
"""Ham veri klasöründen train/val/test bölmeleri ve `dataset.csv` üretir.

İki çalışma modu vardır:

- closed-set: Aynı hayvanın fotoğrafları train/val/test arasında dağıtılır.
  Sistemin, önceden gördüğü bir hayvanın yeni fotoğrafını tanıma yeteneğini ölçer.
- open-set: Hayvan kimlikleri train/val/test arasında tamamen ayrılır.
  Modelin hiç görmediği hayvanlar üzerindeki genelleme gücünü ölçer.

Orijinal dosyalar asla taşınmaz veya silinmez; yalnızca kopyalanır.

Kullanım:
    python scripts/prepare_dataset.py \\
        --source data/raw \\
        --output data \\
        --mode closed-set \\
        --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from app.services.image_service import (  # noqa: E402
    ScannedImage,
    infer_species,
    report_scan_issues,
    scan_animal_dataset,
)

SPLITS = ("train", "val", "test")
CLOSED_SET_MIN_PHOTOS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Veri setini train/val/test olarak böl ve dataset.csv üret."
    )
    parser.add_argument(
        "--source", type=Path, required=True, help="animal_id alt klasörlerini içeren kök klasör."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="train/val/test/metadata klasörlerinin kökü."
    )
    parser.add_argument("--mode", choices=["closed-set", "open-set"], required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _split_counts(
    n: int, train_ratio: float, val_ratio: float, test_ratio: float
) -> tuple[int, int, int]:
    """n fotoğrafı train/val/test'e, her bölmede en az 1 olacak dağıtır (n >= 3 varsayılır)."""
    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        remaining = n - n_train
        denom = val_ratio + test_ratio if (val_ratio + test_ratio) > 0 else 1.0
        n_test = max(1, round(remaining * (test_ratio / denom)))
        n_val = remaining - n_test
        if n_val < 1:
            n_val = 1
            n_test = remaining - n_val
    return n_train, n_val, n_test


def assign_closed_set(
    images: list[ScannedImage],
    rng: random.Random,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[dict[str, str], list[str]]:
    """Her hayvanın kendi fotoğraflarını train/val/test arasında dağıtır.

    Döndürür: (image_path_str -> split), uyarı mesajları listesi.
    """
    by_animal: dict[str, list[ScannedImage]] = {}
    for image in images:
        by_animal.setdefault(image.animal_id, []).append(image)

    assignment: dict[str, str] = {}
    warnings: list[str] = []

    for animal_id, photos in by_animal.items():
        shuffled = photos[:]
        rng.shuffle(shuffled)
        n = len(shuffled)

        if n == 1:
            warnings.append(
                f"{animal_id}: yalnızca 1 fotoğraf var, sadece train'e eklendi (test edilemez)."
            )
            assignment[str(shuffled[0].path)] = "train"
        elif n == 2:
            warnings.append(f"{animal_id}: yalnızca 2 fotoğraf var, val atlandı (1 train, 1 test).")
            assignment[str(shuffled[0].path)] = "train"
            assignment[str(shuffled[1].path)] = "test"
        else:
            if n < CLOSED_SET_MIN_PHOTOS:
                warnings.append(
                    f"{animal_id}: {n} fotoğraf, önerilen minimum {CLOSED_SET_MIN_PHOTOS}."
                )
            n_train, n_val, n_test = _split_counts(n, train_ratio, val_ratio, test_ratio)
            for photo in shuffled[:n_train]:
                assignment[str(photo.path)] = "train"
            for photo in shuffled[n_train : n_train + n_val]:
                assignment[str(photo.path)] = "val"
            for photo in shuffled[n_train + n_val : n_train + n_val + n_test]:
                assignment[str(photo.path)] = "test"

    return assignment, warnings


def assign_open_set(
    images: list[ScannedImage],
    rng: random.Random,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[dict[str, str], list[str]]:
    """Hayvan kimliklerini train/val/test'e ayırır.

    Bir hayvanın tüm fotoğrafları hep aynı bölmeye gider.
    """
    animal_ids = sorted({image.animal_id for image in images})
    warnings: list[str] = []

    if len(animal_ids) < 3:
        warnings.append(
            f"Yalnızca {len(animal_ids)} hayvan var; open-set için en az 3 hayvan önerilir "
            "(train/val/test için birer tane). Tüm hayvanlar train'e eklendi."
        )
        animal_to_split = {animal_id: "train" for animal_id in animal_ids}
    else:
        shuffled_ids = animal_ids[:]
        rng.shuffle(shuffled_ids)
        n_train, n_val, n_test = _split_counts(
            len(shuffled_ids), train_ratio, val_ratio, test_ratio
        )
        animal_to_split = {}
        for animal_id in shuffled_ids[:n_train]:
            animal_to_split[animal_id] = "train"
        for animal_id in shuffled_ids[n_train : n_train + n_val]:
            animal_to_split[animal_id] = "val"
        for animal_id in shuffled_ids[n_train + n_val : n_train + n_val + n_test]:
            animal_to_split[animal_id] = "test"

    assignment = {str(image.path): animal_to_split[image.animal_id] for image in images}
    return assignment, warnings


def copy_assigned_files(assignment: dict[str, str], output_root: Path) -> list[dict[str, str]]:
    """Atanan her dosyayı ilgili split klasörüne kopyalar ve metadata satırlarını döndürür."""
    rows: list[dict[str, str]] = []
    for path_str, split in assignment.items():
        src = Path(path_str)
        animal_id = src.parent.name
        dst_dir = output_root / split / animal_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        shutil.copy2(src, dst)

        try:
            rel_path = dst.relative_to(output_root.parent)
        except ValueError:
            rel_path = dst

        rows.append(
            {
                "image_path": str(rel_path),
                "animal_id": animal_id,
                "split": split,
                "species": infer_species(animal_id) or "",
            }
        )
    return rows


def main() -> None:
    args = parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not (0.99 <= ratio_sum <= 1.01):
        print(f"Hata: train/val/test oranlarının toplamı 1.0 olmalı (şu an: {ratio_sum:.3f}).")
        raise SystemExit(1)

    try:
        scan = scan_animal_dataset(args.source)
    except FileNotFoundError as exc:
        print(f"Hata: {exc}")
        raise SystemExit(1) from exc

    if not scan.images:
        print(f"Hata: '{args.source}' altında okunabilir hiç fotoğraf bulunamadı.")
        print("Her hayvan için bir alt klasör (animal_id) açıp .jpg/.png/.webp fotoğraf koyun.")
        raise SystemExit(1)

    report_scan_issues(scan)

    rng = random.Random(args.seed)
    if args.mode == "closed-set":
        assignment, warnings = assign_closed_set(
            scan.images, rng, args.train_ratio, args.val_ratio, args.test_ratio
        )
    else:
        assignment, warnings = assign_open_set(
            scan.images, rng, args.train_ratio, args.val_ratio, args.test_ratio
        )

    for warning in warnings:
        print(f"Uyarı: {warning}")

    rows = copy_assigned_files(assignment, args.output)

    metadata_dir = args.output / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metadata_dir / "dataset.csv"
    df = pd.DataFrame(rows, columns=["image_path", "animal_id", "split", "species"])
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 60)
    print(f"Mod: {args.mode}")
    print(f"Toplam kopyalanan fotoğraf: {len(rows)}")
    for split in SPLITS:
        print(f"  {split}: {sum(1 for row in rows if row['split'] == split)}")
    print(f"Metadata: {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
