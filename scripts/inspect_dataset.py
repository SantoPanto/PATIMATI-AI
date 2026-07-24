#!/usr/bin/env python
"""Veri klasörünü inceleyip özet istatistikler raporlar.

Kullanım:
    python scripts/inspect_dataset.py --data-dir data/raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.image_service import scan_animal_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Veri klasörünü incele ve özet istatistikler üret."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Her alt klasörü bir animal_id olan kök veri klasörü.",
    )
    parser.add_argument(
        "--min-photos",
        type=int,
        default=3,
        help="Bir hayvanın 'az fotoğraflı' sayılacağı eşik (varsayılan: 3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir

    try:
        result = scan_animal_dataset(data_dir)
    except FileNotFoundError as exc:
        print(f"Hata: {exc}")
        raise SystemExit(1) from exc

    counts = result.counts_per_animal()

    print("=" * 60)
    print(f"Veri klasörü: {data_dir}")
    print(f"Hayvan sayısı: {len(counts)}")
    print(f"Toplam okunabilir fotoğraf sayısı: {len(result.images)}")
    print("=" * 60)

    if counts:
        print("\nHayvan başına fotoğraf sayısı:")
        for animal_id, count in sorted(counts.items()):
            print(f"  {animal_id}: {count}")

    low_count_animals = {aid: c for aid, c in counts.items() if c < args.min_photos}
    print(f"\nAz fotoğraflı hayvanlar (< {args.min_photos} foto): {len(low_count_animals)}")
    for animal_id, count in sorted(low_count_animals.items()):
        print(f"  {animal_id}: {count} foto")

    print(f"\nOkunamayan/bozuk dosyalar: {len(result.unreadable)}")
    for path, reason in result.unreadable:
        print(f"  {path}: {reason}")

    print(f"\nDesteklenmeyen format dosyaları: {len(result.unsupported)}")
    for path in result.unsupported:
        print(f"  {path}")

    if not counts:
        print("\nUyarı: Klasörde hiç okunabilir fotoğraf bulunamadı.")
        print(f"Lütfen '{data_dir}' altına her hayvan için bir alt klasör (animal_id) açın ve")
        print("içine desteklenen formatta (.jpg, .jpeg, .png, .webp) fotoğraflar koyun.")


if __name__ == "__main__":
    main()
