# app/kirpma.py
"""Fotoğraftaki hayvanı (kedi/köpek) tespit edip yalnızca o bölgeye kırpma.

BAĞLAM (docs/olcum-raporu.md §4/§7): ekip, mevcut kimlik eşleştirmesinin
ayırt ediciliğinin büyük ölçüde HAYVANDAN değil ARKA PLANDAN geldiğini
ölçmüştü (tam fotoğraf AUC 0.793, sadece arka plan AUC 0.800 -- arka plan
tek başına tam fotoğraftan bile iyi ayırt ediyor). Bu, açı/poz farkı olan
gerçek fotoğraf çiftlerinde (kayıp evde çekilir, bulundu sokakta -- arka
planlar hiç örtüşmez) ve poster/afiş gibi "hayvanın kareyi doldurmadığı"
görsellerde eşleşmeyi doğrudan bozuyor.

Hayvana kırpma bir kez denenmiş, yanlış bir ölçütle ("ayrım gücü", ölçeğe
duyarlı) ve adil olmayan bir veri kümesiyle (aynı kedi = sokak↔sokak, farklı
kedi = sokak↔stüdyo) "zararlı" bulunup reddedilmişti. Bu karar sonradan
GEÇERSİZ ilan edildi (§4 DÜZELTME, 2026-07-27) ama "adil bir kümeyle yeniden
ölçülüp karara bağlanacak" denilip KAPANMADI (§4 sonu, §7 "Hâlâ
ölçemediğimiz").

BU MODÜL YALNIZCA YETENEĞİ SAĞLAR, CANLI DAVRANIŞI DEĞİŞTİRMEZ:
`CROP_TO_ANIMAL` varsayılan olarak KAPALIDIR (bkz. app/embedder.py:goruntu_ac).
Açılması ayrı, bilinçli bir karar gerektirir çünkü embedding'in üretilme
biçimini değiştirir -- app/surum.py'nin kendi kuralına göre bu, MODEL_SURUMU
artışı gerektirir (aksi hâlde eski/yeni vektörler sessizce yanlış kıyaslanır).
Açma kararı, bu modülün sağladığı yetenekle yapılacak ADİL bir yeniden ölçüme
(gerçek çok-bireyli veri, aynı ortam, tek koşutta karar verilmez -- bkz. §7
"Karar kuralı") bağlı, burada verilmiyor.
"""
from __future__ import annotations

import logging
import os

import torch
from PIL import Image

from .gorsel_sabitleri import ASGARI_KENAR

logger = logging.getLogger(__name__)

# torchvision'ın hazır, COCO üzerinde eğitilmiş dedektörü kullanılıyor --
# ekibin §4'te "hazır bir COCO detektörüyle test edildi" dediği yöntemin
# üründe kullanılabilir hâli. Ayrı bir indirme/lisans riski YOK: torch/
# torchvision zaten sabit bir bağımlılık (requirements.txt), bu ağırlık
# torchvision'ın kendi resmi deposunda (~74 MB, ilk kullanımda iner).
_HEDEF_SINIF_ADLARI = {"cat", "dog"}


def _bool(ad: str, varsayilan: bool) -> bool:
    deger = os.getenv(ad)
    if deger is None:
        return varsayilan
    return deger.strip().lower() in {"1", "true", "yes", "on"}


# ⚠ VARSAYILAN KAPALI. Açan kişi app/surum.py:MODEL_SURUMU'nu da artırmalı --
# bkz. bu modülün docstring'i. Sadece bu bayrağı açmak MODEL_SURUMU'nu
# OTOMATİK artırmaz; bu bilinçli bir insan kararı olarak bırakıldı (bu
# kod tabanındaki KIMLIK_MODEL/TEXT_AI_PROVIDER seçimleriyle aynı desen:
# yeni yetenek env değişkeniyle açılır, varsayılan davranışı korur).
CROP_TO_ANIMAL = _bool("CROP_TO_ANIMAL", False)

# Tespit edilen kutunun etrafına bırakılan pay -- dedektör kutuları genelde
# hayvanın kendisinden biraz dar çıkar (kulak/kuyruk ucu taşabilir).
KENAR_PAYI_ORANI = float(os.getenv("KIRPMA_KENAR_PAYI", "0.10"))

# Bu güvenin altındaki tespitler yok sayılır -- düşük güvenli bir kutuya
# göre kırpmak, hiç kırpmamaktan daha kötü olabilir (yanlış bölgeye kırpar).
# ÖLÇÜLMEDİ: gerçek poster/afiş ya da çok-açılı veri kümesiyle henüz
# doğrulanmadı, torchvision'ın genel COCO değerlendirmesindeki tipik
# çalışma noktasından alınmış temkinli bir başlangıç değeri.
GUVEN_ESIGI = float(os.getenv("KIRPMA_GUVEN_ESIGI", "0.50"))


class HayvanDedektoru:
    """Tek sorumluluk: bir görüntüde en belirgin kedi/köpek kutusunu bulmak.

    Yalnızca gerçekten kullanıldığında (bkz. `_dedektoru_al`'ın tembel
    singleton'ı) yüklenir -- CROP_TO_ANIMAL kapalıyken hiçbir ek model
    belleğe/ağa yük getirmez, PetEmbedder/KimlikGomucu'nun singleton deseniyle
    aynı gerekçe (bkz. app/embedder.py)."""

    def __init__(self):
        # Yerel import: torchvision.models.detection modülü yalnızca burada
        # gerekiyor, dosyanın en üstüne alınırsa CROP_TO_ANIMAL kapalıyken
        # bile içe aktarma anında ekstra modül yükü eklenmiş olurdu.
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )
        from torchvision.transforms.functional import to_tensor

        self._to_tensor = to_tensor
        agirliklar = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
        kategoriler = agirliklar.meta["categories"]
        self._hedef_idx = {i for i, ad in enumerate(kategoriler) if ad in _HEDEF_SINIF_ADLARI}
        logger.info("Hayvan dedektörü yükleniyor (torchvision, COCO)...")
        self._model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=agirliklar)
        self._model.eval()
        logger.info("Hayvan dedektörü hazır.")

    def en_iyi_kutu(self, img: Image.Image) -> tuple[float, float, float, float] | None:
        """En belirgin kedi/köpek kutusunu (x1, y1, x2, y2) piksel
        koordinatlarında döner; GUVEN_ESIGI'ni geçen hiçbir tespit yoksa None.

        Birden fazla hayvan tespit edilirse EN BÜYÜK ALANLI kutu seçilir --
        gerekçe: bir posterde/kolajda ana konu olan hayvan genelde kareyi en
        çok kaplayandır (bir logo/ikon hayvanı ya da arka plandaki başka bir
        hayvan değil). ÖLÇÜLMEDİ -- gerçek poster verisiyle doğrulanmadı."""
        with torch.no_grad():
            tahmin = self._model([self._to_tensor(img)])[0]

        adaylar = [
            (float(k[0]), float(k[1]), float(k[2]), float(k[3]))
            for k, etiket, skor in zip(tahmin["boxes"], tahmin["labels"], tahmin["scores"])
            if int(etiket) in self._hedef_idx and float(skor) >= GUVEN_ESIGI
        ]
        if not adaylar:
            return None
        return max(adaylar, key=lambda k: (k[2] - k[0]) * (k[3] - k[1]))


_dedektor: HayvanDedektoru | None = None


def _dedektoru_al() -> HayvanDedektoru:
    global _dedektor
    if _dedektor is None:
        _dedektor = HayvanDedektoru()
    return _dedektor


def hayvana_kirp(img: Image.Image, *, dedektor: HayvanDedektoru | None = None) -> Image.Image:
    """Görseli, tespit edilen kedi/köpek kutusuna (kenar payıyla) kırpar.

    Hayvan bulunamazsa -- ekibin ölçümünde 7 fotoğrafın 2'sinde (%29)
    olduğu gibi, bkz. docs/olcum-raporu.md §4 -- görsel OLDUĞU GİBİ döner:
    sessizce boş/anlamsız bir kırpma üretip fotoğrafı çöpe çevirmez, tam
    kareye geri düşer. Aynı fallback, kırpma sonrası kalan bölge çok küçük
    çıkarsa da uygulanır (ör. dedektörün minik, muhtemelen yanlış bir kutu
    bulduğu durum) -- çok küçük bir kırpma, hiç kırpmamaktan daha kötü bir
    sonuçtur.

    `dedektor` testlerin gerçek modeli indirmeden sahte bir dedektör
    enjekte edebilmesi için var; verilmezse tembel yüklenen tekil
    (singleton) gerçek dedektör kullanılır."""
    dedektor = dedektor or _dedektoru_al()
    kutu = dedektor.en_iyi_kutu(img)
    if kutu is None:
        return img

    x1, y1, x2, y2 = kutu
    w, h = img.size
    pay_x = (x2 - x1) * KENAR_PAYI_ORANI
    pay_y = (y2 - y1) * KENAR_PAYI_ORANI
    x1 = max(0.0, x1 - pay_x)
    y1 = max(0.0, y1 - pay_y)
    x2 = min(float(w), x2 + pay_x)
    y2 = min(float(h), y2 + pay_y)

    kirpilmis = img.crop((int(x1), int(y1), int(x2), int(y2)))
    if min(kirpilmis.size) < ASGARI_KENAR:
        logger.warning(
            "kırpılmış bölge çok küçük (%dx%d) -- tam görüntüye geri dönülüyor",
            *kirpilmis.size,
        )
        return img
    return kirpilmis
