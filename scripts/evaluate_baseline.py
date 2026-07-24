#!/usr/bin/env python
"""Bir veri klasöründeki her fotoğrafı sırayla sorgu olarak kullanarak Top-1/Top-5/MRR ölçer.

Her sorgu için, aynı klasördeki diğer tüm fotoğraflar aday havuzunu oluşturur
(sorgunun kendisi hariç). Bir hayvanın havuzda başka fotoğrafı yoksa, o sorgu
doğruluk metriklerinden hariç tutulur (doğru cevap yapısal olarak imkansızdır)
ve "atlanan sorgu" olarak raporlanır.

Kullanım:
    python scripts/evaluate_baseline.py \\
        --data-dir data/test \\
        --output-dir outputs/evaluations/baseline \\
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _common import (  # noqa: E402
    QueryResult,
    evaluate_leave_one_out,
    load_and_embed,
    summarize_results,
)

from app.services.embedding_service import get_embedding_service  # noqa: E402
from app.services.image_service import report_scan_issues, scan_animal_dataset  # noqa: E402

COLOR_CORRECT = "#2a78d6"
COLOR_WRONG = "#eb6834"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MegaDescriptor baseline'ını Top-1/Top-5/MRR ile değerlendirir."
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="animal_id alt klasörlerini içeren klasör."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Sonuçların yazılacağı klasör."
    )
    parser.add_argument("--top-k", type=int, default=5)
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


def save_top_k_csv(results: list[QueryResult], path: Path) -> None:
    rows = []
    for result in results:
        if not result.top_candidates:
            continue
        for rank, (cand_path, cand_animal_id, similarity, is_correct) in enumerate(
            result.top_candidates, start=1
        ):
            rows.append(
                {
                    "query_path": result.query_path,
                    "query_animal_id": result.query_animal_id,
                    "rank": rank,
                    "candidate_path": cand_path,
                    "candidate_animal_id": cand_animal_id,
                    "similarity": round(similarity, 4),
                    "is_correct": is_correct,
                }
            )
    columns = [
        "query_path",
        "query_animal_id",
        "rank",
        "candidate_path",
        "candidate_animal_id",
        "similarity",
        "is_correct",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def save_false_positives_csv(results: list[QueryResult], path: Path) -> None:
    rows = []
    for result in results:
        if not result.has_positive or not result.top_candidates:
            continue
        top1_path, top1_animal_id, top1_similarity, top1_correct = result.top_candidates[0]
        if not top1_correct:
            rows.append(
                {
                    "query_path": result.query_path,
                    "true_animal_id": result.query_animal_id,
                    "predicted_animal_id": top1_animal_id,
                    "predicted_path": top1_path,
                    "similarity": round(top1_similarity, 4),
                }
            )
    columns = [
        "query_path",
        "true_animal_id",
        "predicted_animal_id",
        "predicted_path",
        "similarity",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def plot_similarity_distribution(results: list[QueryResult], path: Path) -> None:
    correct_sims = [r.correct_similarity for r in results if r.correct_similarity is not None]
    wrong_sims = [r.wrong_similarity for r in results if r.wrong_similarity is not None]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-1.0, 1.0, 40).tolist()
    if correct_sims:
        ax.hist(
            correct_sims,
            bins=bins,
            alpha=0.75,
            color=COLOR_CORRECT,
            label="Doğru eşleşme",
            edgecolor="white",
        )
    if wrong_sims:
        ax.hist(
            wrong_sims,
            bins=bins,
            alpha=0.75,
            color=COLOR_WRONG,
            label="Yanlış eşleşme",
            edgecolor="white",
        )

    ax.set_xlabel("Cosine benzerliği")
    ax.set_ylabel("Sorgu sayısı")
    ax.set_title("Doğru vs. yanlış eşleşme benzerlik dağılımı")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    if correct_sims or wrong_sims:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_best_worst(results: list[QueryResult], n: int = 3) -> None:
    processed = [r for r in results if r.has_positive and r.correct_similarity is not None]
    if not processed:
        print("\nDeğerlendirilebilecek sorgu yok (en iyi/en kötü örnek raporlanamıyor).")
        return

    def margin(r: QueryResult) -> float:
        wrong = r.wrong_similarity if r.wrong_similarity is not None else -1.0
        correct = r.correct_similarity if r.correct_similarity is not None else -1.0
        return correct - wrong

    ranked = sorted(processed, key=margin, reverse=True)
    print(f"\nEn başarılı {n} sorgu (doğru/yanlış benzerlik farkı en yüksek):")
    for r in ranked[:n]:
        print(f"  {r.query_path} | doğru={r.correct_similarity:.3f} yanlış={r.wrong_similarity}")

    print(f"\nEn başarısız {n} sorgu (doğru/yanlış benzerlik farkı en düşük):")
    for r in ranked[-n:][::-1]:
        print(f"  {r.query_path} | doğru={r.correct_similarity:.3f} yanlış={r.wrong_similarity}")


def main() -> None:
    args = parse_args()

    try:
        scan = scan_animal_dataset(args.data_dir)
    except FileNotFoundError as exc:
        print(f"Hata: {exc}")
        raise SystemExit(1) from exc

    if not scan.images:
        print(f"Hata: '{args.data_dir}' altında okunabilir hiç fotoğraf bulunamadı.")
        print("Değerlendirme için önce scripts/prepare_dataset.py ile bir test bölmesi oluşturun.")
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

    if len(metas) < 2:
        print("Hata: Değerlendirme için en az 2 okunabilir fotoğraf gerekli.")
        raise SystemExit(1)

    results = evaluate_leave_one_out(embeddings, metas, args.top_k)
    metrics = summarize_results(results, args.top_k)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if metrics["processed_queries"] == 0:
        print("\nUyarı: Havuzda ikinci fotoğrafı olan hayvan yok; Top-1/Top-5/MRR hesaplanamaz.")
        print(f"'{args.data_dir}' altındaki her animal_id klasörüne en az 2 fotoğraf ekleyin.")
        (args.output_dir / "results.json").write_text(
            json.dumps(
                {
                    "data_dir": str(args.data_dir),
                    "top_k": args.top_k,
                    "generated_at": datetime.now(UTC).isoformat(),
                    "metrics": metrics,
                    "note": "Hiçbir sorgu değerlendirilemedi: eşleşen ikinci foto yok.",
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        raise SystemExit(1)

    print("\n" + "=" * 60)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("=" * 60)

    print_best_worst(results)

    save_top_k_csv(results, args.output_dir / "top_k_results.csv")
    save_false_positives_csv(results, args.output_dir / "false_positives.csv")
    plot_similarity_distribution(results, args.output_dir / "similarity_distribution.png")

    report = {
        "data_dir": str(args.data_dir),
        "model_name": embedding_service.model_name,
        "device": str(embedding_service.device),
        "top_k": args.top_k,
        "generated_at": datetime.now(UTC).isoformat(),
        "total_images": len(metas),
        "skipped_images_on_load": len(load_failures),
        "metrics": metrics,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nSonuçlar kaydedildi: {args.output_dir}")


if __name__ == "__main__":
    main()
