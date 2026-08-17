#!/bin/bash
# entrypoint.sh — tek imajdan iki süreç
#
# NEDEN VAR: Dockerfile yalnızca uvicorn'u başlatıyordu. Üretimde ilanlar
# oluşuyor ama `python -m app.kuyruk` hiç çalışmadığı için AI analizi HİÇ
# yapılmıyor, ilanlar sonsuza kadar ai_status=PENDING kalıyordu. Hata veren
# değil SESSİZCE ÇALIŞMAYAN bir sistemdi: bütün birim testler yeşildi ve hiçbiri
# bunu göremezdi, çünkü kusur kodda değil PAKETLEMEDEYDİ.
#
# NEDEN AYRI SÜREÇ, TEK SÜREÇTE THREAD DEĞİL: app/kuyruk.py:17-22'deki not.
# Model çıkarımı CPU'yu saniyelerce meşgul eden SENKRON bir iş; aynı süreçte
# olsaydı FastAPI'nin olay döngüsünü bloklar, /health bile cevapsız kalırdı.
# Bu karar burada TARTIŞILMIYOR, uygulanıyor.
#
# NEDEN ROL: Railway'de bir servis = bir süreç. Orada aynı imajdan iki servis
# açılır (SERVIS_ROLU=http ve SERVIS_ROLU=kuyruk). Lokalde ve tek makinede ise
# tek `docker run` ile ikisi birden kalksın diye varsayılan "hepsi".
#
# NEDEN bash, /bin/sh (dash) DEĞİL: aşağıdaki `wait -n` — "çocuklardan hangisi
# önce biterse" demenin dolambaçsız tek yolu — dash'te YOK. Debian tabanlı
# python:3.11-slim'de bash zaten kurulu (Debian'da "required" öncelikli),
# ek paket gerekmiyor.

set -u
# `set -e` BİLEREK YOK. Aşağıdaki `wait -n`in sıfırdan farklı dönmesi HATA
# değil, beklediğimiz OLAYIN KENDİSİ. `-e` açık olsaydı kabuk tam o satırda
# sessizce ölür; hangi çocuğun neden bittiğini günlüğe yazamaz ve hayatta kalan
# çocuğa TERM gönderemezdik.

SERVIS_ROLU="${SERVIS_ROLU:-hepsi}"
PORT="${PORT:-8000}"

# `python -m app.kuyruk` için çalışma dizini /app olmalı: `-m`, sys.path[0]'a
# ÇALIŞMA DİZİNİNİ koyar, betiğin bulunduğu yeri değil. `docker run -w` ile
# dizin ezilirse burada düzeltiyoruz.
cd /app

# %q ile basıyoruz, %s ile DEĞİL: değişkende görünmez bir karakter varsa
# (Windows'ta CRLF olarak çalışma ağacına açılmış bir dosyadan gelen \r gibi)
# burada $'hepsi\r' diye GÖRÜNÜR olur. %s olsaydı ekrana "hepsi" yazardı ve
# aşağıdaki case'in neden eşleşmediği hiç anlaşılmazdı. (.gitattributes bu
# durumu önlüyor; burası ikinci ağ.)
printf '[entrypoint] rol=%q port=%s\n' "$SERVIS_ROLU" "$PORT" >&2

case "$SERVIS_ROLU" in
  http)
    # exec: uvicorn PID 1 olur ve SIGTERM'i DOĞRUDAN alır. Araya kabuk
    # koymuyoruz — sinyali iletmeyi unutan bir kabuk, kapanışı SIGKILL'e
    # düşürür. uvicorn kendi SIGTERM işleyicisini kuruyor.
    exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
    ;;
  kuyruk)
    # exec: tüketici PID 1 olur.
    # DİKKAT: Linux, PID 1'e gelen ve İŞLEYİCİSİ OLMAYAN sinyalin varsayılan
    # eylemini UYGULAMAZ — sinyali sessizce yutar. Yani app/kuyruk.py'deki
    # SIGTERM işleyicisi olmadan bu rol `docker stop` ile HİÇ durmaz; 10 sn
    # beklenir ve SIGKILL gelir. İki değişiklik birbirine bağlı.
    exec python -m app.kuyruk
    ;;
  hepsi)
    : # aşağıda
    ;;
  *)
    printf '[entrypoint] HATA: SERVIS_ROLU=%q tanınmıyor. Seçenekler: hepsi|http|kuyruk\n' \
        "$SERVIS_ROLU" >&2
    exit 64   # EX_USAGE
    ;;
esac

# ---------------------------------------------------------------------------
# rol=hepsi : iki süreç bir arada
#
# BELLEK UYARISI: her süreç modeli KENDİ belleğine yükler — app/embedder.py
# sonundaki `embedder = PetEmbedder()` ve `kimlik_gomucu = KimlikGomucu(...)`
# modül seviyesinde, yani içe aktarma anında, SÜREÇ BAŞINA bir kez çalışıyor.
# Süreçler arası paylaşım YOK. KIMLIK_MODEL=clip ile süreç başına bir CLIP,
# "siglip2-animal" ile süreç başına İKİ model. Yani bu rol belleği kabaca
# İKİYE KATLAR. Küçük örneklerde bu rolü kullanmayın; orada iki ayrı servis
# açın (http + kuyruk).
# ---------------------------------------------------------------------------

uvicorn app.main:app --host 0.0.0.0 --port "$PORT" &
HTTP_PID=$!

python -m app.kuyruk &
KUYRUK_PID=$!

# Bu satır SÜSLEME DEĞİL: .github/workflows/ci.yml'deki "tüketici ölünce
# konteyner de ölmeli" adımı öldüreceği süreci buradan okuyor (imajda pgrep
# yok, procps kurulu değil). Biçimi değişirse CI adımı da değişmeli.
printf '[entrypoint] http pid=%s  kuyruk pid=%s\n' "$HTTP_PID" "$KUYRUK_PID" >&2

KAPANIYOR=0

_kapat() {
    KAPANIYOR=1
    # Bundan sonraki TERM/INT bu kabuğu bozmasın: aşağıdaki `wait` çağrılarının
    # sinyalle yarıda kesilmesini istemiyoruz. Çocuklar bu değişiklikten
    # ETKİLENMEZ; sinyal düzenleri fork anında kopyalandı.
    trap '' TERM INT
    printf '[entrypoint] kapatma sinyali alındı, çocuklara TERM gönderiliyor\n' >&2
    kill -TERM "$HTTP_PID" "$KUYRUK_PID" 2>/dev/null || true
}
# Tuzak, İKİ PID de belliyken kuruluyor: daha erken kurulsaydı iki `&` arasına
# düşen bir sinyalde işleyici tanımsız değişken okur ve `set -u` yüzünden
# patlardı.
trap _kapat TERM INT

# `wait -n`: çocuklardan HERHANGİ BİRİ bitince döner.
#   - çocuk kendiliğinden bittiyse -> $? o çocuğun çıkış kodu
#   - araya TERM/INT girdiyse       -> `wait` 128'den büyük bir kodla döner ve
#     HEMEN ARDINDAN tuzak çalışır (bash: wait sinyalle kesilince önce döner,
#     sonra tuzak yürütülür). KAPANIYOR bayrağı iki durumu ayırır.
#   - `sleep 1` ile yoklamıyoruz: hem kapanışı bir saniyeye kadar geciktirir
#     hem de boşuna bir süreç yakar.
# -p: biten sürecin PID'ini değişkene yazar. Değişken önce UNSET edilir,
#     o yüzden aşağıda ${BITEN_PID:-} diye okunuyor (set -u açık).
BITEN_PID=""
CIKIS=0
wait -n -p BITEN_PID || CIKIS=$?

if [ "$KAPANIYOR" -eq 1 ]; then
    # `docker stop` / Ctrl+C yolu. TERM zaten gitti; kapanmalarını bekle.
    wait "$HTTP_PID"   2>/dev/null || true
    wait "$KUYRUK_PID" 2>/dev/null || true
    printf '[entrypoint] düzgün kapandı\n' >&2
    exit 0
fi

# Buraya düştüysek çocuklardan biri KENDİLİĞİNDEN bitti.
#
# NEDEN HEMEN ÇIKIYORUZ, ölen çocuğu yerinde yeniden başlatmıyoruz:
#   - HEALTHCHECK yalnızca HTTP'ye bakıyor. Tüketici ölse bile "healthy" der;
#     ilanlar yine sessizce PENDING'de kalır — düzeltmeye çalıştığımız hatanın
#     TA KENDİSİ.
#   - app/kuyruk.py:266 broker kopmasını ZATEN kendi içinde, artan beklemeyle
#     karşılıyor. Yani buraya düşen bir tüketici ölümü geçici bir ağ sorunu
#     DEĞİL, gerçek bir arızadır (içe aktarma hatası, yakalanmamış istisna).
#     Yerinde yeniden başlatmak, yeşil bir /health arkasında sonsuza kadar
#     çöken bir tüketici saklamak olurdu. app/kuyruk.py'nin kendi kuralı:
#     "sessizce asılı kalan bir tüketici, çökmüş olandan daha kötüdür".
#   - Sıfırdan farklı çıkınca konteyner yöneticisi (Railway; lokalde
#     `docker run --restart unless-stopped`) HER İKİ süreci de temiz bir
#     durumdan başlatır. Kurtarma mantığı burada değil, orada olmalı.
#
# ⚠ BAĞIMLILIK: bu tasarım app/kuyruk.py'nin AMQP hatalarını SONSUZA KADAR
# yeniden denemesine dayanıyor. Oraya bir deneme üst sınırı eklenirse, bu
# satırlar broker kesintisini sessizce bir HTTP kesintisine çevirir.
#
# DİKKAT: `docker run`ın VARSAYILAN yeniden başlatma politikası "no"dur.
# --restart verilmezse konteyner ölür ve ÖYLE KALIR. README bunu söylüyor.
if [ -z "${BITEN_PID:-}" ]; then
    # `wait -n -p` bash 5.1+ ister (bookworm'da 5.2 var). Yine de sessiz
    # kalmayalım: biten sürecin PID'i artık yoktur, `kill -0` onu ele verir.
    if ! kill -0 "$HTTP_PID" 2>/dev/null; then
        BITEN_PID="$HTTP_PID"
    else
        BITEN_PID="$KUYRUK_PID"
    fi
fi

if [ "$BITEN_PID" = "$HTTP_PID" ]; then
    BITEN="http (uvicorn)"
else
    BITEN="kuyruk (python -m app.kuyruk)"
fi
printf '[entrypoint] %s süreci %s koduyla bitti — konteyner kapatılıyor\n' \
    "$BITEN" "$CIKIS" >&2

trap '' TERM INT
kill -TERM "$HTTP_PID" "$KUYRUK_PID" 2>/dev/null || true
wait "$HTTP_PID"   2>/dev/null || true
wait "$KUYRUK_PID" 2>/dev/null || true

# Ölen çocuk 0 ile bitmiş olsa bile SIFIRDAN FARKLI çıkıyoruz: "hepsi" rolünde
# süreçlerden birinin tek başına bitmesi tanım gereği arızadır ve
# `--restart on-failure` yalnızca sıfırdan farklı çıkışta devreye girer.
# (app/kuyruk.py:255 KeyboardInterrupt yolunda 0 ile döner — o yol yukarıdaki
# KAPANIYOR dalında ele alınıyor, buraya düşmemeli.)
if [ "$CIKIS" -eq 0 ]; then
    CIKIS=1
fi
exit "$CIKIS"
