FROM python:3.11-slim

WORKDIR /app

# Kimlik modeli seçimi (bkz. app/surum.py). Varsayılan "siglip2-animal": ağırlığı
# ~1,4 GB ve depoda değil, o yüzden AŞAĞIDA BUILD SIRASINDA indiriliyor — imajın
# "çalışma zamanında ağ bağımlılığı kalmasın" amacı ancak böyle korunur. Bedeli
# build'in yavaşlaması ve imajın büyümesidir; bilinçli takas.
#
# Modeli değiştirecekseniz aynı değeri HEM build-arg HEM runtime env olarak verin;
# yalnız runtime'da vermek yetmez, o ağırlık imajda bulunmaz ve ilk istekte
# indirilmeye çalışılır:
#   docker build --build-arg KIMLIK_MODEL=google-siglip2 -t patimati-ai .
#   docker run -e KIMLIK_MODEL=google-siglip2 patimati-ai
#
# CI bilerek "clip" ile build ediyor (.github/workflows/ci.yml) — 1,4 GB'lık
# indirme her koşumu yavaşlatır ve kırılganlaştırır.
ARG KIMLIK_MODEL=siglip2-animal

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

# Python stdout'u boruya yazarken blok tamponlaması yapmasın: aksi hâlde
# günlükler öbekler hâlinde ve geç görünür, konteyner SIGKILL yerse hiç
# görünmez. Tüketicinin "Kuyruk dinleniyor" satırı da buna takılıyordu.
ENV PYTHONUNBUFFERED=1

# Varsayılan rol imajın İÇİNDE dursun: hem `docker inspect` ile görünür olur
# hem de aşağıdaki HEALTHCHECK okuyabilir. hepsi = HTTP + kuyruk bir arada.
ENV SERVIS_ROLU=hepsi

# rol=hepsi'de İKİ torch süreci yan yana çalışıyor ve her biri varsayılan
# olarak çekirdek sayısı kadar OpenMP iş parçacığı açıp diğerinin CPU'sunu
# yiyor. Tek mesajlık iş yükünde (AMQP_PREFETCH=1) çok iş parçacığının
# getirisi zaten küçük; çakışmanın maliyeti büyük.
ENV OMP_NUM_THREADS=2

# `COPY app/`ın ÜSTÜNDE: entrypoint.sh app/ kodundan çok daha seyrek değişir.
# (Dürüst olalım: ikisi de KB'lık katmanlar, pahalı katmanlar — pip kurulumu ve
# model indirme — zaten yukarıda. Bu hız değil düzen kararı.)
COPY entrypoint.sh /entrypoint.sh
# chmod: bu depoda core.filemode=false ve 100755 kipli TEK BİR dosya yok, yani
# git çalıştırma bitini kaydetmiyor. Burada açıkça veriyoruz ki hata ayıklarken
# `docker exec ai /entrypoint.sh` elle koşulabilsin. Konteynerin AÇILIŞI buna
# BAĞLI DEĞİL: aşağıdaki CMD betiği açıkça `bash` ile çağırıyor.
RUN chmod +x /entrypoint.sh

COPY app/ ./app/

EXPOSE 8000

# Üç düzeltme:
#  1) Port SABİT 8000 yazıyordu, CMD ise ${PORT}'u onurlandırıyordu. Railway
#     PORT=1234 bastığında sağlık kontrolü ÖLÜ bir porta bakıp konteyneri
#     sonsuza kadar "unhealthy" gösteriyordu. (Dockerfile ayrıştırıcısı
#     CMD/HEALTHCHECK argümanlarında değişken genişletmez; ${PORT:-8000}'i
#     çalışma zamanında `sh` genişletir — bu yüzden burada işe yarar.)
#     127.0.0.1, "localhost" DEĞİL: localhost ::1'e de çözülüyor, uvicorn ise
#     yalnızca 0.0.0.0'ı (IPv4) dinliyor.
#  2) --start-period: modeller içe aktarma anında yükleniyor (app/embedder.py
#     sonundaki singleton'lar; app/topoloji.py notu süreç başına ~25 sn diyor)
#     ve rol=hepsi'de İKİ süreç aynı anda yüklüyor. Bu süre boyunca /health
#     cevap veremez; start-period olmadan konteyner her açılışta "unhealthy"
#     görünür ve alan güvenilirliğini yitirirdi.
#  3) kuyruk rolünde HTTP sunucusu HİÇ YOK, curl her zaman başarısız olurdu.
#     O rolde entrypoint `exec` kullanıyor, yani konteynerin ayakta olması
#     TÜKETİCİNİN ayakta olmasıyla birebir aynı şey — ek kontrol gereksiz.
#
# ⚠ SINIRI: bu kontrol yalnızca HTTP'ye bakar. Tüketici ölürse "healthy" demeye
# devam ederdi — tüketicinin canlılığı HEALTHCHECK ile değil, entrypoint.sh'ın
# fail-fast davranışıyla (biri ölünce konteyner ölür) sağlanıyor.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD [ "${SERVIS_ROLU:-hepsi}" = "kuyruk" ] || curl -fsS "http://127.0.0.1:${PORT:-8000}/health" || exit 1

# ENTRYPOINT DEĞİL, CMD — bilerek:
#   - Railway'in "Custom Start Command" alanı konteynerin Cmd'sini DEĞİŞTİRİR.
#     ENTRYPOINT olsaydı oraya yazılan komut entrypoint'e ARGÜMAN olurdu; bizim
#     betik argümanları yok sayıyor, yani YANLIŞ SÜREÇ sessizce çalışırdı.
#   - `docker run patimati-ai bash` hata ayıklamak için çalışmaya devam etsin.
#     ENTRYPOINT olsaydı `--entrypoint bash` yazmak gerekirdi.
# `bash /entrypoint.sh`, `/entrypoint.sh` DEĞİL: çalıştırma biti eksik olsa bile
# açılış çalışsın (yukarıdaki chmod notuna bak).
CMD ["bash", "/entrypoint.sh"]
