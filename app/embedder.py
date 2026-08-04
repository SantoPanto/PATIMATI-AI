# app/embedder.py
import io
import logging
from pathlib import Path

import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor

from .hatalar import GecersizGoruntu
from .surum import _SECIM

logger = logging.getLogger(__name__)

# Bundan küçük görüntüler anlamlı bir vektör üretmez (1x1 bile sessizce
# 512'lik bir vektör döndürüyordu — çöp veriyi veritabanına yazmayalım).
ASGARI_KENAR = 32

# PIL'in varsayılan "decompression bomb" sınırı ~89 megapiksel; bu, RGB olarak
# ~268 MB bellek demek ve küçük bir konteyneri öldürür. 40 MP fazlasıyla yeterli.
Image.MAX_IMAGE_PIXELS = 40_000_000


def goruntu_ac(image_bytes: bytes) -> Image.Image:
    """Ham baytları analize hazır RGB görüntüye çevirir.

    Görüntü açma işi TEK BURADA yapılır — hem embedding hem renk analizi bunu
    kullanır. Ayrı ayrı açılsaydı düzeltmelerden birini diğerine eklemeyi
    unutmak çok kolay olurdu (ör. döndürmeyi ekleyip renkte unutmak).

    Yapılanlar:
      1. EXIF döndürmesini uygular — telefon fotoğrafları çoğu zaman "yan"
         kaydedilir; düzeltilmezse vektör bozulur ve eşleştirme kötüleşir.
      2. Şeffaflığı BEYAZ zemine yerleştirir — düz RGB'ye çevirmek şeffaf
         alanları siyaha çevirip renk analizini bozuyordu.
      3. Çok küçük görüntüleri reddeder.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as e:
        raise GecersizGoruntu(f"Görüntü açılamadı: {e}") from e

    img = ImageOps.exif_transpose(img)

    if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
        img = img.convert("RGBA")
        zemin = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(zemin, img)

    img = img.convert("RGB")

    if min(img.size) < ASGARI_KENAR:
        raise GecersizGoruntu(
            f"Görüntü çok küçük: {img.size[0]}x{img.size[1]} "
            f"(en az {ASGARI_KENAR}x{ASGARI_KENAR} olmalı)")
    return img


class PetEmbedder:
    """
    CLIP ViT-B/32 modeli ile görsel özellik çıkarıcı.
    İlk çağrıda model HuggingFace'den indirilir (~400 MB).
    """
    MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(self):
        logger.info("CLIP modeli yükleniyor...")
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_ID)
        self.model = CLIPModel.from_pretrained(self.MODEL_ID)
        self.model.eval()  # Inference moduna al
        logger.info("CLIP modeli hazır.")

    def embed(self, image_path: str) -> list[float]:
        """
        Bir görüntüyü 512 boyutlu L2-normalize embedding'e dönüştürür.
        Dönüş: Python float listesi (JSON serileştirilebilir)
        """
        return self.embed_bytes(Path(image_path).read_bytes())

    def embed_bytes(self, image_bytes: bytes) -> list[float]:
        """Byte dizisinden embedding çıkar. Geçersiz görüntüde GecersizGoruntu fırlatır."""
        return self._embed_pil(goruntu_ac(image_bytes))

    def embed_text(self, texts: list[str]) -> torch.Tensor:
        """Metin listesini L2-normalize CLIP embedding matrisine dönüştürür (zero-shot için)."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        if not isinstance(feats, torch.Tensor):
            feats = feats.pooler_output
        return feats / feats.norm(dim=-1, keepdim=True)

    def _embed_pil(self, img: Image.Image) -> list[float]:
        inputs = self.processor(images=img, return_tensors="pt")
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        # transformers 4.x düz tensor, 5.x çıktı nesnesi döndürür
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        # L2 normalizasyon — cosine similarity için ZORUNLU
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().tolist()


class KimlikGomucu:
    """Eşleştirmede kullanılan vektörü üreten model.

    ETİKETÇİDEN AYRI: `embedder` (CLIP) tür/desen/cins etiketlerini üretir,
    bu sınıf ise yalnızca kimlik vektörünü. Ölçümle ayrıldılar — kimlik için
    ince ayarlanmış modeller etiket işini yapamıyor (bkz. app/surum.py başlığı).

    Seçim `KIMLIK_MODEL` ortam değişkeniyle yapılır. Varsayılan "clip" ise
    AYRI BİR MODEL YÜKLENMEZ; etiketçi CLIP kimlik için de kullanılır, ek bellek
    maliyeti sıfır olur ve davranış eskisiyle birebir aynı kalır.
    """

    def __init__(self, secim: dict, etiketci: PetEmbedder):
        self.boyut = secim["boyut"]
        # "clip" seçiliyse ikinci bir model yüklemenin anlamı yok
        self._clip_mi = secim["kimlik"] == etiketci.MODEL_ID
        if self._clip_mi:
            self._etiketci = etiketci
            logger.info("Kimlik vektörü etiketçi CLIP'ten alınacak (ek model yok).")
            return

        from transformers import AutoModel, AutoProcessor
        logger.info("Kimlik modeli yükleniyor: %s", secim["kimlik"])
        self._islemci = AutoProcessor.from_pretrained(secim["islemci"] or secim["kimlik"])
        self._model, bilgi = AutoModel.from_pretrained(secim["kimlik"],
                                                       output_loading_info=True)
        # Ağırlıklar eksik yüklenirse transformers yalnızca UYARI basar ve model
        # rastgele değerlerle çalışmaya devam eder. Bu, sessizce çöp vektör
        # üretmek demektir — 2026-07-27'de tam bunu yaşadık (bkz. olcum-raporu §7).
        eksik = bilgi.get("missing_keys") or []
        uyusmayan = bilgi.get("mismatched_keys") or []
        if eksik or uyusmayan:
            onarilmis = self._onekli_agirliklari_onar(secim["kimlik"])
            if onarilmis is None:
                raise RuntimeError(
                    f"{secim['kimlik']}: ağırlıklar tam yüklenmedi "
                    f"({len(eksik)} eksik, {len(uyusmayan)} uyuşmayan). "
                    f"Rastgele ağırlıkla çalışmaktansa başlatmıyoruz.")
            self._model = onarilmis
        self._model.eval()
        logger.info("Kimlik modeli hazır (%d boyut).", self.boyut)

    @staticmethod
    def _onekli_agirliklari_onar(kimlik):
        """Tüm anahtarları ortak önek taşıyan checkpoint'i yüklenebilir hâle getirir.

        AvitoTech'in SigLIP2 deposunda 408 anahtarın hepsi `clip.` önekli ve
        `text_config.vocab_size` eksik. transformers yalnızca hedef sınıfın kendi
        önekini soyduğu için hiçbiri tutmuyor ve model RASTGELE ağırlıkla yükleniyor
        — hata değil, yalnızca uyarı basarak.

        Onarım burada olmalı: elle düzeltilmiş yerel bir kopyaya bağlı kalırsak
        modeli ekipten kimse kullanamaz (2026-07-28'de tam bu oldu).
        Tanımadığı bir durumda None döner, çağıran taraf hata fırlatır.
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel

        yol = Path(kimlik)
        dosya = (yol / "model.safetensors") if yol.exists() else Path(
            hf_hub_download(kimlik, "model.safetensors"))
        sd = load_file(str(dosya))

        onek = "clip."
        if not sd or not all(k.startswith(onek) for k in sd):
            return None
        sd = {k[len(onek):]: v for k, v in sd.items()}

        cfg = AutoConfig.from_pretrained(kimlik)
        gomme = sd.get("text_model.embeddings.token_embedding.weight")
        if gomme is not None and hasattr(cfg, "text_config"):
            cfg.text_config.vocab_size = gomme.shape[0]

        model = AutoModel.from_config(cfg)
        eksik, _ = model.load_state_dict(sd, strict=False)
        if eksik:
            return None
        logger.info("Kimlik modeli onarıldı: '%s' öneki soyuldu", onek)
        return model

    def embed_bytes(self, image_bytes: bytes) -> list[float]:
        if self._clip_mi:
            return self._etiketci.embed_bytes(image_bytes)
        return self._embed_pil(goruntu_ac(image_bytes))

    def _embed_pil(self, img: Image.Image) -> list[float]:
        inputs = self._islemci(images=img, return_tensors="pt")
        with torch.no_grad():
            features = self._model.get_image_features(**inputs)
        if not isinstance(features, torch.Tensor):
            havuz = getattr(features, "pooler_output", None)
            features = havuz if havuz is not None else features[0]
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().tolist()


# Singleton — uygulama başlangıcında bir kez yüklenir
embedder = PetEmbedder()
kimlik_gomucu = KimlikGomucu(_SECIM, embedder)
