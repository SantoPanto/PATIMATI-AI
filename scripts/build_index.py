#!/usr/bin/env python
"""Bir veri klasöründeki tüm fotoğraflardan yerel arama indeksi (.npz) oluşturur.

Kullanım:
    python scripts/build_index.py \\
        --data-dir data/raw \\
        --output outputs/embeddings/pet_index.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import load_and_embed  # noqa: E402

from app.services.embedding_service import get_embedding_service  # noqa: E402
from app.services.image_service import (  # noqa: E402
    infer_species,
    report_scan_issues,
    scan_animal_dataset,
)
from app.services.index_service import IndexRecord, LocalNumpyIndex  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="animal_id alt klasörlerinden yerel bir arama indeksi oluşturur."
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="animal_id alt klasörlerini içeren klasör."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Çıktı .npz indeks dosyasının yolu."
    )
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
        print("Her hayvan için bir alt klasör (animal_id) oluşturup içine fotoğraflar koyun.")
        raise SystemExit(1)

    report_scan_issues(scan)

    kwargs: dict[str, Any] = {}
    if args.model_name:
        kwargs["model_name"] = args.model_name
    if args.checkpoint:
        kwargs["checkpoint_path"] = args.checkpoint
    embedding_service = get_embedding_service(**kwargs)
    print(f"Model: {embedding_service.model_name} | Cihaz: {embedding_service.device}")

    embeddings, metas, load_failures = load_and_embed(
        scan.images, embedding_service, args.batch_size
    )
    if load_failures:
        print(f"\nUyarı: {len(load_failures)} görsel atlandı:")
        for path, reason in load_failures:
            print(f"  {path}: {reason}")

    if not metas:
        print("Hata: Hiçbir görsel başarıyla yüklenemedi.")
        raise SystemExit(1)

    index = LocalNumpyIndex()
    records = [
        IndexRecord(
            embedding=embeddings[i],
            image_id=str(meta.path.relative_to(args.data_dir)),
            image_path=str(meta.path),
            animal_id=meta.animal_id,
            species=infer_species(meta.animal_id),
            metadata={},
        )
        for i, meta in enumerate(metas)
    ]
    index.add_batch(records)
    index.save(args.output)

    print("\n" + "=" * 60)
    print(f"İndekslenen fotoğraf sayısı: {len(index)}")
    print(f"Hayvan sayısı: {len({r.animal_id for r in records})}")
    print(f"Kaydedildi: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
