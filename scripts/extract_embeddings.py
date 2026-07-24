#!/usr/bin/env python
"""Bir veri klasöründeki tüm görsellerden toplu embedding çıkarıp `.npz` olarak kaydeder.

Kullanım:
    python scripts/extract_embeddings.py \\
        --data-dir data/test \\
        --output outputs/embeddings/test_embeddings.npz \\
        --batch-size 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from _common import load_and_embed  # noqa: E402

from app.services.embedding_service import get_embedding_service  # noqa: E402
from app.services.image_service import (  # noqa: E402
    infer_species,
    report_scan_issues,
    scan_animal_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bir veri klasöründen embedding çıkar ve .npz olarak kaydet."
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="animal_id alt klasörlerini içeren klasör."
    )
    parser.add_argument("--output", type=Path, required=True, help="Çıktı .npz dosyasının yolu.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Varsayılan MegaDescriptor yerine kullanılacak model.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="Fine-tune edilmiş bir checkpoint kullan."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        scan = scan_animal_dataset(args.data_dir)
    except FileNotFoundError as exc:
        print(f"Hata: {exc}")
        raise SystemExit(1) from exc

    if not scan.images:
        print(f"Hata: '{args.data_dir}' altında okunabilir hiç fotoğraf bulunamadı.")
        raise SystemExit(1)

    report_scan_issues(scan)

    kwargs: dict[str, Any] = {}
    if args.model_name:
        kwargs["model_name"] = args.model_name
    if args.checkpoint:
        kwargs["checkpoint_path"] = args.checkpoint
    embedding_service = get_embedding_service(**kwargs)
    print(
        f"Model: {embedding_service.model_name} | Cihaz: {embedding_service.device} | "
        f"Boyut: {embedding_service.embedding_dim}"
    )

    embeddings_matrix, valid_meta, load_failures = load_and_embed(
        scan.images, embedding_service, args.batch_size
    )

    if load_failures:
        print(f"\nUyarı: {len(load_failures)} görsel yüklenirken atlandı:")
        for path, reason in load_failures:
            print(f"  {path}: {reason}")

    if not valid_meta:
        print("Hata: Hiçbir görsel başarıyla yüklenemedi.")
        raise SystemExit(1)

    image_paths = np.array([str(m.path) for m in valid_meta], dtype=object)
    animal_ids = np.array([m.animal_id for m in valid_meta], dtype=object)
    species = np.array([infer_species(m.animal_id) or "" for m in valid_meta], dtype=object)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        embeddings=embeddings_matrix,
        image_path=image_paths,
        animal_id=animal_ids,
        species=species,
    )

    print("\n" + "=" * 60)
    print(f"Toplam embedding: {len(valid_meta)}")
    print(f"Atlanan görsel: {len(load_failures)}")
    print(f"Kaydedildi: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
