#!/usr/bin/env python
"""Hazır (baseline) modeli ve fine-tune edilmiş bir checkpoint'i aynı test kümesinde karşılaştırır.

Fine-tune edilmiş modelin daha iyi olduğu varsayılmaz; fark ne olursa olsun raporlanır.

Kullanım:
    python scripts/evaluate_model.py \\
        --data-dir data/test \\
        --checkpoint outputs/training/run_001/best_model.pth \\
        --output-dir outputs/evaluations/comparison \\
        --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import evaluate_leave_one_out, load_and_embed, summarize_results  # noqa: E402

from app.services.embedding_service import EmbeddingService, get_embedding_service  # noqa: E402
from app.services.image_service import (  # noqa: E402
    ScannedImage,
    report_scan_issues,
    scan_animal_dataset,
)

METRIC_LABELS: dict[str, str] = {
    "top1_accuracy": "Top-1",
    "top_5_accuracy": "Top-5",
    "mean_reciprocal_rank": "MRR",
    "avg_query_latency_ms": "Ortalama gecikme (ms)",
    "false_positive_rate": "Yanlış pozitif oranı",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline ve fine-tune edilmiş modeli aynı test kümesinde karşılaştırır."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="animal_id alt klasörlerini içeren test klasörü.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Fine-tune edilmiş checkpoint dosyası."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-name", type=str, default="hf-hub:BVRA/MegaDescriptor-S-224")
    return parser.parse_args()


def run_evaluation(
    embedding_service: EmbeddingService, images: list[ScannedImage], batch_size: int, top_k: int
) -> dict[str, Any]:
    embeddings, metas, load_failures = load_and_embed(images, embedding_service, batch_size)
    if len(metas) < 2:
        return {"metrics": None, "skipped_on_load": len(load_failures)}
    results = evaluate_leave_one_out(embeddings, metas, top_k)
    metrics = summarize_results(results, top_k)
    if metrics.get("top1_accuracy") is not None:
        metrics["false_positive_rate"] = 1.0 - metrics["top1_accuracy"]
    else:
        metrics["false_positive_rate"] = None
    return {"metrics": metrics, "skipped_on_load": len(load_failures)}


def print_comparison_table(baseline: dict[str, Any], finetuned: dict[str, Any], top_k: int) -> None:
    keys = [
        "top1_accuracy",
        f"top_{top_k}_accuracy",
        "mean_reciprocal_rank",
        "avg_query_latency_ms",
        "false_positive_rate",
    ]
    labels = {**METRIC_LABELS, f"top_{top_k}_accuracy": f"Top-{top_k}"}

    print(f"\n{'Metric':<24}{'Baseline':>14}{'Fine-tuned':>14}{'Difference':>14}")
    print("-" * 66)
    for key in keys:
        base_val = baseline.get(key)
        fine_val = finetuned.get(key)
        label = labels.get(key, key)
        if base_val is None or fine_val is None:
            print(f"{label:<24}{'—':>14}{'—':>14}{'—':>14}")
            continue
        diff = fine_val - base_val
        print(f"{label:<24}{base_val:>14.4f}{fine_val:>14.4f}{diff:>+14.4f}")

    top1_base = baseline.get("top1_accuracy")
    top1_fine = finetuned.get("top1_accuracy")
    if top1_base is not None and top1_fine is not None:
        if top1_fine < top1_base:
            print("\nUyarı: Fine-tune edilmiş model, Top-1 doğrulukta baseline'dan daha kötü.")
        elif top1_fine > top1_base:
            print("\nFine-tune edilmiş model, Top-1 doğrulukta baseline'ı geçti.")
        else:
            print("\nFine-tune edilmiş model ile baseline Top-1 doğrulukta eşit.")


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        print(f"Hata: Checkpoint bulunamadı: {args.checkpoint}")
        raise SystemExit(1)

    try:
        scan = scan_animal_dataset(args.data_dir)
    except FileNotFoundError as exc:
        print(f"Hata: {exc}")
        raise SystemExit(1) from exc

    if not scan.images:
        print(f"Hata: '{args.data_dir}' altında okunabilir hiç fotoğraf bulunamadı.")
        raise SystemExit(1)

    report_scan_issues(scan)

    print("\n--- Baseline (hazır model) değerlendiriliyor ---")
    baseline_service = get_embedding_service(model_name=args.model_name)
    baseline_result = run_evaluation(baseline_service, scan.images, args.batch_size, args.top_k)

    print("\n--- Fine-tune edilmiş model değerlendiriliyor ---")
    finetuned_service = get_embedding_service(
        model_name=args.model_name, checkpoint_path=args.checkpoint
    )
    finetuned_result = run_evaluation(finetuned_service, scan.images, args.batch_size, args.top_k)

    if baseline_result["metrics"] is None or finetuned_result["metrics"] is None:
        print("\nHata: Değerlendirme için yetersiz veri (en az 2 okunabilir fotoğraf gerekir).")
        raise SystemExit(1)

    print_comparison_table(baseline_result["metrics"], finetuned_result["metrics"], args.top_k)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "data_dir": str(args.data_dir),
        "checkpoint": str(args.checkpoint),
        "model_name": args.model_name,
        "top_k": args.top_k,
        "generated_at": datetime.now(UTC).isoformat(),
        "baseline": baseline_result,
        "finetuned": finetuned_result,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nKarşılaştırma raporu kaydedildi: {args.output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
