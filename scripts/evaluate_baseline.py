#!/usr/bin/env python
"""İnce ayar YAPILMAMIŞ SigLIP2 backbone'un kimlik ayırt etme performansını ölçer.

`scripts/train.py`'nin ilk güvenlik kontrolü budur: fine-tuning'e başlamadan önce
bu script çalıştırılmış ve `--output-dir`'e bir `results.json` yazmış olmalı
(bkz. `train.py`'deki `check_baseline_exists`). Amaç kör bir şekilde fine-tune
etmemek — eğitim bittiğinde `training_history.json`'daki `val_top1` bu sonuçla
karşılaştırılıp gerçekten bir kazanç olup olmadığı görülebilir.

Protokol: LEAVE-ONE-OUT. `--data-dir` altındaki her `<animal_id>/<foto>` için o
fotoğraf sorgu, AYNI KLASÖRDEKİ DİĞER TÜM fotoğraflar (kendisi hariç) galeri
kabul edilir; en yakın komşu (cosine benzerlik) doğru `animal_id`'yi verirse
top-1 isabet sayılır. Tek fotoğraflı hayvanlar değerlendirmeye katılmaz — galoride
eşleşecek başka bir fotoğrafları olmadığı için "doğru cevap" diye bir şey yok,
katılırlarsa metrik modelden bağımsız olarak yapay biçimde düşer.

`scripts/train.py`'deki `SigLIP2VisionBackbone`'u İÇE AKTARMIYORUZ: o import
`wildlife_tools`'u da beraberinde getirir (ArcFaceLoss/BasicTrainer), ama baseline
ölçümü onlara hiç ihtiyaç duymaz. Burada aynı üç satırlık model yükleme/özellik
çıkarma mantığı bağımsız tutuluyor ki wildlife-tools kurulu olmadan da çalışsın.

Kullanım:
    python scripts/evaluate_baseline.py \\
        --data-dir data/test \\
        --output-dir outputs/evaluations/baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402

from app.device import get_device  # noqa: E402
from app.services.image_service import (  # noqa: E402
    load_rgb_image,
    report_scan_issues,
    scan_animal_dataset,
)

MODEL_NAME = "google/siglip2-base-patch16-224"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="İnce ayar öncesi SigLIP2 backbone'un leave-one-out kimlik doğruluğunu ölçer."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


class _ImageOnlyDataset(Dataset):
    def __init__(self, paths: list[Path], processor) -> None:
        self.paths = paths
        self.processor = processor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = load_rgb_image(self.paths[idx])
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]


def extract_features(model, pixel_values: torch.Tensor) -> torch.Tensor:
    features = model.get_image_features(pixel_values=pixel_values)
    # transformers 4.x düz tensor, 5.x çıktı nesnesi döndürebiliyor (bkz. app/embedder.py).
    if not isinstance(features, torch.Tensor):
        pooled = getattr(features, "pooler_output", None)
        features = pooled if pooled is not None else features[0]
    return features


def embed_images(
    model, processor, device: torch.device, paths: list[Path], batch_size: int
) -> np.ndarray:
    dataset = _ImageOnlyDataset(paths, processor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            features = extract_features(model, batch)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            embeddings.append(features.cpu().numpy())
    if not embeddings:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(embeddings)


def leave_one_out_accuracy(
    embeddings: np.ndarray, animal_ids: list[str], top_k: int
) -> dict[str, float | int]:
    """Her görüntüyü sırayla sorgu kabul edip kalan görüntülerde (kendisi hariç)
    en yakın komşuyu arar. Tek fotoğraflı hayvanlar zaten `animal_ids`'e girmemiş
    olmalı (bkz. `main`'deki filtre)."""
    n = len(embeddings)
    similarities = embeddings @ embeddings.T
    np.fill_diagonal(similarities, -np.inf)

    top1_correct = 0
    topk_correct = 0
    for i in range(n):
        order = np.argsort(-similarities[i])
        ranked_ids = [animal_ids[j] for j in order]
        top1_correct += int(ranked_ids[0] == animal_ids[i])
        topk_correct += int(animal_ids[i] in ranked_ids[:top_k])

    return {
        "processed_queries": n,
        "top1_accuracy": top1_correct / n if n else None,
        "top_k": top_k,
        "topk_accuracy": topk_correct / n if n else None,
    }


def main() -> None:
    args = parse_args()

    try:
        scan = scan_animal_dataset(args.data_dir)
    except FileNotFoundError as exc:
        print(f"Hata: {exc}")
        raise SystemExit(1) from exc
    report_scan_issues(scan)

    counts_by_animal: dict[str, int] = {}
    for img in scan.images:
        counts_by_animal[img.animal_id] = counts_by_animal.get(img.animal_id, 0) + 1

    evaluable = [img for img in scan.images if counts_by_animal[img.animal_id] >= 2]
    skipped_singletons = len(scan.images) - len(evaluable)
    if skipped_singletons:
        print(
            f"Uyarı: {skipped_singletons} fotoğraf değerlendirmeye katılmadı "
            "(hayvanının galoride eşleşecek başka fotoğrafı yok)."
        )

    device = get_device()
    print(f"Cihaz: {device} | Model: {args.model_name}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.json"

    if not evaluable:
        print("Uyarı: Değerlendirilebilecek hiçbir sorgu yok (processed_queries=0).")
        results = {
            "model_name": args.model_name,
            "data_dir": str(args.data_dir),
            "metrics": {
                "processed_queries": 0,
                "top1_accuracy": None,
                "top_k": args.top_k,
                "topk_accuracy": None,
            },
        }
        results_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Sonuç yazıldı: {results_path}")
        return

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.eval()
    model.to(device)

    paths = [img.path for img in evaluable]
    animal_ids = [img.animal_id for img in evaluable]
    embeddings = embed_images(model, processor, device, paths, args.batch_size)

    metrics = leave_one_out_accuracy(embeddings, animal_ids, args.top_k)

    results = {
        "model_name": args.model_name,
        "data_dir": str(args.data_dir),
        "skipped_singleton_images": skipped_singletons,
        "metrics": metrics,
    }
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"Sorgu sayısı : {metrics['processed_queries']}")
    print(f"Top-1        : {metrics['top1_accuracy']:.4f}")
    print(f"Top-{args.top_k:<8}: {metrics['topk_accuracy']:.4f}")
    print(f"Sonuç yazıldı: {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
