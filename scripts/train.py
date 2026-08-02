#!/usr/bin/env python
"""SigLIP2 görüntü backbone'unu ArcFace kaybıyla kendi verimizde fine-tune eder.

Wildlife Tools kütüphanesinin `BasicTrainer` ve `ArcFaceLoss` bileşenlerini kullanır
(bkz. `wildlife_tools.train.trainer` / `wildlife_tools.train.objective`). Backbone'un
katman isimlerine göre güvenilir biçimde bölünemeyeceği varsayımıyla, iki aşamalı
fine-tuning parametre sırasına göre kaba ama mimariden bağımsız bir dondurma/açma
stratejisiyle uygulanır: ilk aşamada son parametrelerin küçük bir kısmı eğitilir,
ikinci aşamada tüm görüntü encoder'ı daha düşük bir öğrenme oranıyla açılır.
Yalnızca görüntü (vision) encoder'ı kullanılır; metin encoder'ı eğitim dışında
kalıcı olarak dondurulur.

Eğitime başlamadan önce iki güvenlik kontrolü yapılır:
  1. Bir baseline değerlendirmesi (`scripts/evaluate_baseline.py`) daha önce çalıştırılmış olmalı.
  2. Eğitim klasöründe en az 2 farklı hayvan ve yeterli sayıda fotoğraf bulunmalı.

Kullanım:
    python scripts/train.py \\
        --train-dir data/train \\
        --val-dir data/val \\
        --output-dir outputs/training/run_001 \\
        --epochs 10 \\
        --batch-size 8 \\
        --learning-rate 0.0001 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import chain
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402
from wildlife_tools.train.objective import ArcFaceLoss  # noqa: E402
from wildlife_tools.train.trainer import BasicTrainer, set_seed  # noqa: E402

from dataclasses import dataclass
from PIL import Image

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class ScannedImage:
    path: Path
    animal_id: str

@dataclass
class DatasetScan:
    images: list[ScannedImage]
    issues: list[str]

def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")

def scan_animal_dataset(data_dir: Path) -> DatasetScan:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Klasör bulunamadı: {data_dir}")
    images = []
    issues = []
    for animal_dir in data_dir.iterdir():
        if not animal_dir.is_dir():
            continue
        animal_id = animal_dir.name
        animal_imgs = [
            ScannedImage(path=f, animal_id=animal_id) 
            for f in animal_dir.iterdir() 
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
        if not animal_imgs:
            issues.append(f"Hayvan '{animal_id}' klasörü boş.")
        images.extend(animal_imgs)
    return DatasetScan(images=images, issues=issues)

def report_scan_issues(scan: DatasetScan) -> None:
    for issue in scan.issues:
        print(f"Uyarı: {issue}")

MODEL_NAME = "google/siglip2-base-patch16-224"
MIN_TRAIN_IDENTITIES = 2
MIN_TRAIN_IMAGES = 4
COLOR_CORRECT = "#2a78d6"
COLOR_WRONG = "#eb6834"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SigLIP2 görüntü backbone'unu ArcFace kaybıyla fine-tune eder."
    )
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping sabrı (epoch).")
    parser.add_argument(
        "--warmup-epochs", type=int, default=2, help="1. aşamada eğitilecek epoch sayısı."
    )
    parser.add_argument(
        "--unfreeze-fraction",
        type=float,
        default=0.2,
        help="1. aşamada eğitime açık bırakılacak son parametre tensörlerinin oranı.",
    )
    parser.add_argument(
        "--stage2-lr-factor",
        type=float,
        default=0.1,
        help="2. aşamada learning-rate'in çarpılacağı katsayı.",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument(
        "--resume", type=Path, default=None, help="Devam edilecek checkpoint dosyası."
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=Path("outputs/evaluations/baseline/results.json"),
        help="Eğitimden önce var olması gereken baseline sonuç dosyası.",
    )
    return parser.parse_args()


def check_baseline_exists(baseline_results_path: Path) -> None:
    """Eğitime başlamadan önce anlamlı bir baseline değerlendirmesinin var olduğunu doğrular."""
    if not baseline_results_path.exists():
        print(f"Hata: Baseline değerlendirme sonucu bulunamadı: {baseline_results_path}")
        print(
            "Önce şunu çalıştırın: python scripts/evaluate_baseline.py "
            "--data-dir data/test --output-dir outputs/evaluations/baseline"
        )
        raise SystemExit(1)
    try:
        data = json.loads(baseline_results_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Hata: Baseline sonuç dosyası okunamadı: {exc}")
        raise SystemExit(1) from exc

    processed = data.get("metrics", {}).get("processed_queries", 0) or 0
    if processed == 0:
        print("Hata: Baseline değerlendirmesi hiçbir sorguyu işleyememiş (processed_queries=0).")
        print("Eğitime geçmeden önce anlamlı bir baseline sonucu elde edin.")
        raise SystemExit(1)

    print(
        f"Baseline bulundu: {baseline_results_path} (top1={data['metrics'].get('top1_accuracy')})"
    )


class AnimalDataset(Dataset):
    """`animal_id` klasörlerinden okunan görselleri model girdisine çeviren PyTorch veri kümesi.

    `label_to_index`'te olmayan bir `animal_id` -1 etiketiyle işaretlenir (örn. val kümesinde
    train'de hiç görülmemiş bir hayvan varsa); bu örnekler değerlendirmede ayrıca elenir.
    """

    def __init__(
        self, images: list[ScannedImage], label_to_index: dict[str, int], transform: object
    ) -> None:
        self.images = images
        self.label_to_index = label_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        scanned = self.images[idx]
        image = load_rgb_image(scanned.path)
        # `self.transform` artık bir HF `AutoProcessor`; `return_tensors="pt"` her zaman
        # bir batch ekseni ekler ([1, C, H, W]) — DataLoader'ın varsayılan collate'i kendi
        # batch eksenini oluşturacağı için burada [0] ile tek örneğe indirgiyoruz.
        tensor = self.transform(images=image, return_tensors="pt")["pixel_values"][0]
        label = self.label_to_index.get(scanned.animal_id, -1)
        return tensor, label


class SigLIP2VisionBackbone(torch.nn.Module):
    """SigLIP2'nin yalnızca görüntü encoder'ını MegaDescriptor'ın yerine geçecek şekilde sarmalar.

    `AutoModel.from_pretrained` metin ve görüntü kulelerini birlikte yükler (ayrıca
    `logit_scale`/`logit_bias` gibi kontrastif eğitime özgü üst düzey parametreler de
    modelde bulunur). Bu eğitim akışında yalnızca görüntü özelliği kullanıldığından
    tüm model önce dondurulur, ardından yalnızca görüntü encoder'ı (`vision_model`)
    varsayılan eğitilebilir durumuna geri açılır. `set_stage` da yalnızca
    `vision_named_parameters()` üzerinde çalıştığı için metin encoder'ı ve diğer
    üst düzey parametreler hiçbir aşamada açılmaz.
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        for _, param in self.vision_named_parameters():
            param.requires_grad = True
        with torch.inference_mode():
            dummy_pixel_values = torch.zeros(1, 3, 224, 224)
            dummy_features = self._extract_features(dummy_pixel_values)
        self.num_features = dummy_features.shape[-1]

    def _extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.model.get_image_features(pixel_values=pixel_values)
        # transformers 4.x düz tensor, 5.x çıktı nesnesi döndürebiliyor (bkz. app/embedder.py).
        if not isinstance(features, torch.Tensor):
            pooled = getattr(features, "pooler_output", None)
            features = pooled if pooled is not None else features[0]
        return features

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self._extract_features(pixel_values)

    def vision_named_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        return list(self.model.vision_model.named_parameters())


class ClippingTrainer(BasicTrainer):
    """`BasicTrainer`ın epoch döngüsünü gradient clipping ekleyerek genişletir (kütüphanede yok)."""

    def __init__(self, *args: object, grad_clip_norm: float = 5.0, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.grad_clip_norm = grad_clip_norm

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        model = self.model.train()
        losses = []
        clip_params = list(self.model.parameters()) + list(self.objective.parameters())
        last_i = -1
        for i, (x, y) in enumerate(loader):
            last_i = i
            x, y = x.to(self.device), y.to(self.device)
            out = model(x)
            loss = self.objective(out, y)
            loss_scaled = loss / self.accumulation_steps
            loss_scaled.backward()
            if (i + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(clip_params, self.grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
            losses.append(loss.detach().cpu())

        if last_i >= 0 and (last_i + 1) % self.accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(clip_params, self.grad_clip_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.scheduler:
            self.scheduler.step()

        return {"train_loss_epoch_avg": float(np.mean(losses)) if losses else float("nan")}


def set_stage(model: SigLIP2VisionBackbone, unfreeze_last_fraction: float) -> None:
    """Son `unfreeze_last_fraction` oranındaki vision encoder parametrelerini eğitime açar.

    Mimariden bağımsız, kaba ama güvenilir bir yaklaşım: yalnızca `model.vision_named_parameters()`
    (SigLIP2 görüntü encoder'ı) üzerinde çalışır; bu sıra girişten çıkışa doğru olduğundan son
    parametreler encoder'ın son bloklarına karşılık gelir. Metin encoder'ı bu fonksiyona hiç
    dahil edilmez, dolayısıyla asla açılmaz.
    """
    named_params = model.vision_named_parameters()
    n_total = len(named_params)
    n_unfreeze = max(1, int(round(n_total * unfreeze_last_fraction)))
    freeze_until = n_total - n_unfreeze
    for i, (_, param) in enumerate(named_params):
        param.requires_grad = i >= freeze_until


def embed_dataset(
    model: torch.nn.Module, device: torch.device, dataset: Dataset, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Modeli `eval()` moduna alıp veri kümesi için L2-normalize embedding ve etiket çıkarır."""
    was_training = model.training
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            out = torch.nn.functional.normalize(out, p=2, dim=1)
            embeddings.append(out.cpu().numpy())
            labels.append(y.numpy())
    if was_training:
        model.train()
    if not embeddings:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.concatenate(embeddings), np.concatenate(labels)


def validate(
    model: torch.nn.Module,
    device: torch.device,
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    top_k: int = 5,
) -> dict[str, float | int | None]:
    """Train kümesini galeri, val kümesini sorgu kabul ederek Top-1/Top-K val doğruluğu hesaplar."""
    gallery_emb, gallery_labels = embed_dataset(model, device, train_dataset, batch_size)
    val_emb, val_labels = embed_dataset(model, device, val_dataset, batch_size)

    if len(val_emb) == 0 or len(gallery_emb) == 0:
        return {
            "val_top1": None,
            "val_top5": None,
            "val_queries": 0,
            "val_skipped": len(val_labels),
        }

    valid_mask = val_labels != -1
    n_skipped = int((~valid_mask).sum())
    val_emb = val_emb[valid_mask]
    val_labels = val_labels[valid_mask]

    if len(val_emb) == 0:
        return {"val_top1": None, "val_top5": None, "val_queries": 0, "val_skipped": n_skipped}

    similarities = val_emb @ gallery_emb.T
    top1_correct = 0
    topk_correct = 0
    for i in range(len(val_emb)):
        order = np.argsort(-similarities[i])
        ranked_labels = gallery_labels[order]
        top1_correct += int(ranked_labels[0] == val_labels[i])
        topk_correct += int(bool(np.any(ranked_labels[:top_k] == val_labels[i])))

    return {
        "val_top1": top1_correct / len(val_emb),
        "val_top5": topk_correct / len(val_emb),
        "val_queries": len(val_emb),
        "val_skipped": n_skipped,
    }


def plot_training_curves(history: list[dict], path: Path) -> None:
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_top1 = [h.get("val_top1") for h in history]
    val_top5 = [h.get("val_top5") for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs, train_losses, color=COLOR_CORRECT, marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Eğitim kaybı (ArcFace)")
    axes[0].set_title("Eğitim kaybı")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].grid(alpha=0.2)

    if any(v is not None for v in val_top1):
        axes[1].plot(epochs, val_top1, color=COLOR_CORRECT, marker="o", label="Val Top-1")
        axes[1].plot(epochs, val_top5, color=COLOR_WRONG, marker="o", label="Val Top-5")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(frameon=False)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Doğruluk")
    axes[1].set_title("Doğrulama doğruluğu")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    check_baseline_exists(args.baseline_results)

    try:
        train_scan = scan_animal_dataset(args.train_dir)
    except FileNotFoundError as exc:
        print(f"Hata: {exc}")
        raise SystemExit(1) from exc
    report_scan_issues(train_scan)

    train_animal_ids = sorted({img.animal_id for img in train_scan.images})
    if len(train_animal_ids) < MIN_TRAIN_IDENTITIES:
        print(
            f"Hata: Fine-tuning için en az {MIN_TRAIN_IDENTITIES} farklı hayvan gerekli, "
            f"'{args.train_dir}' içinde {len(train_animal_ids)} bulundu."
        )
        print("Önce scripts/prepare_dataset.py ile yeterli hayvanlı bir train bölmesi oluşturun.")
        raise SystemExit(1)
    if len(train_scan.images) < MIN_TRAIN_IMAGES:
        print(
            f"Hata: Fine-tuning için en az {MIN_TRAIN_IMAGES} eğitim fotoğrafı gerekli, "
            f"{len(train_scan.images)} bulundu."
        )
        raise SystemExit(1)

    val_scan = (
        scan_animal_dataset(args.val_dir) if args.val_dir.is_dir() else None
    )
    if val_scan is not None:
        report_scan_issues(val_scan)
    else:
        print(
            f"Uyarı: val klasörü bulunamadı ({args.val_dir}); doğrulama metrikleri hesaplanmayacak."
        )

    device = get_device()
    set_seed(args.seed)
    print(f"Cihaz: {device} | Seed: {args.seed}")

    label_to_index = {animal_id: i for i, animal_id in enumerate(train_animal_ids)}

    model = SigLIP2VisionBackbone(args.model_name)
    transform = model.processor
    embedding_dim = model.num_features

    train_dataset = AnimalDataset(train_scan.images, label_to_index, transform)
    val_dataset = AnimalDataset(val_scan.images if val_scan else [], label_to_index, transform)

    objective = ArcFaceLoss(num_classes=len(label_to_index), embedding_size=embedding_dim)

    set_stage(model, args.unfreeze_fraction)
    trainable_params = [
        p for p in chain(model.parameters(), objective.parameters()) if p.requires_grad
    ]
    optimizer = AdamW(trainable_params, lr=args.learning_rate)

    trainer = ClippingTrainer(
        dataset=train_dataset,
        model=model,
        objective=objective,
        optimizer=optimizer,
        epochs=args.epochs,
        device=str(device),
        batch_size=args.batch_size,
        num_workers=0,
        grad_clip_norm=args.grad_clip_norm,
    )

    start_epoch = 0
    if args.resume:
        if not args.resume.exists():
            print(f"Hata: Devam edilecek checkpoint bulunamadı: {args.resume}")
            raise SystemExit(1)
        trainer.load(str(args.resume))
        start_epoch = trainer.epoch
        print(f"Checkpoint'ten devam ediliyor: {args.resume} (epoch {start_epoch})")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    stage2_started = start_epoch >= args.warmup_epochs
    if stage2_started:
        set_stage(model, 1.0)

    best_val_top1: float | None = None
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(start_epoch, args.epochs):
        if not stage2_started and epoch >= args.warmup_epochs:
            print("2. aşama başlıyor: tüm görüntü encoder parametreleri açılıyor, learning-rate düşürülüyor.")
            set_stage(model, 1.0)
            new_lr = args.learning_rate * args.stage2_lr_factor
            trainable_params = [
                p for p in chain(model.parameters(), objective.parameters()) if p.requires_grad
            ]
            trainer.optimizer = AdamW(trainable_params, lr=new_lr)
            stage2_started = True

        loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        epoch_data = trainer.train_epoch(loader)
        trainer.epoch += 1

        val_metrics = validate(model, device, train_dataset, val_dataset, args.batch_size, top_k=5)

        train_loss = epoch_data["train_loss_epoch_avg"]
        print(
            f"Epoch {epoch + 1}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_top1={val_metrics['val_top1']} | val_top5={val_metrics['val_top5']} "
            f"(sorgu={val_metrics['val_queries']}, atlanan={val_metrics['val_skipped']})"
        )

        history.append({"epoch": epoch + 1, "train_loss": train_loss, **val_metrics})

        trainer.save(str(args.output_dir), file_name="last_model.pth")

        current_score = val_metrics["val_top1"]
        if current_score is None:
            trainer.save(str(args.output_dir), file_name="best_model.pth")
            best_epoch = epoch + 1
        elif best_val_top1 is None or current_score > best_val_top1:
            best_val_top1 = current_score
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            trainer.save(str(args.output_dir), file_name="best_model.pth")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping: {args.patience} epoch boyunca val_top1 iyileşmedi.")
                break

    history_path = args.output_dir / "training_history.json"
    history_path.write_text(
        json.dumps(
            {"history": history, "best_epoch": best_epoch, "best_val_top1": best_val_top1}, indent=2
        ),
        encoding="utf-8",
    )
    plot_training_curves(history, args.output_dir / "training_curves.png")

    print("\n" + "=" * 60)
    print("Eğitim tamamlandı.")
    best_path = args.output_dir / "best_model.pth"
    print(f"En iyi model: {best_path} (epoch {best_epoch}, val_top1={best_val_top1})")
    print(f"Son model: {args.output_dir / 'last_model.pth'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
