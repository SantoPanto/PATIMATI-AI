FROM python:3.11-slim

WORKDIR /app

# Kimlik modeli seçimi (bkz. app/surum.py). Yalnızca runtime'da KIMLIK_MODEL
# ortam değişkenini değiştirmek YETMEZ: "clip" dışındaki seçenekler ağırlığı
# HuggingFace'ten indirir ve bu, ilk istekte ~1,4 GB'a kadar ağ bekleyişi +
# dışa bağımlılık demektir — imajın "çalışma zamanında ağ bağımlılığı kalmasın"
# amacını bozar. Farklı bir model kullanacaksanız aynı değeri hem build-arg hem
# runtime env olarak verin:
#   docker build --build-arg KIMLIK_MODEL=siglip2-animal -t patimati-ai .
#   docker run -e KIMLIK_MODEL=siglip2-animal patimati-ai
ARG KIMLIK_MODEL=clip

# curl: container healthcheck için
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Önce bağımlılıklar: kod değişince bu katman önbellekten gelir, tekrar kurulmaz
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# CLIP her koşulda gerekir (etiketçi SABİT, bkz. app/surum.py) — build sırasında
# indir ki çalışma zamanında ağ bağımlılığı kalmasın.
RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"

# Kimlik modeli CLIP'ten farklıysa (KIMLIK_MODEL_YOLU ile yerel bir yol
# verilmediği sürece) o ağırlıklar da burada indirilir. Model kimlikleri
# bilerek app/surum.py'den TEKRAR YAZILDI (import etmedik): app/ kodu henüz
# COPY edilmedi ki bu katman yalnızca KIMLIK_MODEL değişince yeniden çalışsın,
# ilgisiz bir kod değişikliğinde 1,4 GB'lık indirme önbellekten düşmesin.
# app/surum.py'deki KIMLIK_MODELLERI sözlüğü değişirse burası da güncellenmeli.
RUN case "$KIMLIK_MODEL" in \
      clip) echo "Kimlik modeli CLIP ile aynı, ek indirme yok." ;; \
      siglip2-animal) python -c "from transformers import AutoModel, AutoProcessor; \
AutoModel.from_pretrained('AvitoTech/SigLIP2-Base-for-animal-identification'); \
AutoProcessor.from_pretrained('google/siglip-base-patch16-224')" ;; \
      google-siglip2) python -c "from transformers import AutoModel, AutoProcessor; \
AutoModel.from_pretrained('google/siglip2-base-patch16-224'); \
AutoProcessor.from_pretrained('google/siglip2-base-patch16-224')" ;; \
      *) echo "UYARI: bilinmeyen KIMLIK_MODEL='$KIMLIK_MODEL', build sırasında indirilemedi (app/surum.py başlarken hata verecek)." ;; \
    esac
ENV KIMLIK_MODEL=${KIMLIK_MODEL}

COPY app/ ./app/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Railway PORT ortam değişkeni basar; lokalde 8000'e düşer
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
