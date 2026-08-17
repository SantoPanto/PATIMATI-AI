# tests/test_docker_baslatma.py
"""Bekçi: imaj İKİ süreci de başlatıyor mu?

NEDEN VAR: Dockerfile'ın CMD'i yalnızca uvicorn'u başlatıyordu; kuyruk
tüketicisi (`python -m app.kuyruk`) imajda hiçbir yerde çalıştırılmıyordu.
Sonuç, üretimde ilanların sonsuza kadar ai_status=PENDING kalmasıydı. Bütün
birim testler yeşildi ve hiçbiri bunu göremezdi — çünkü kusur KODDA değil
PAKETLEMEDEYDİ.

Asıl kanıt .github/workflows/ci.yml'deki `docker` işidir: imajı gerçekten kurar,
çalıştırır ve kuyruğa mesaj bırakıp sonucu bekler. Ama o iş ~10 dakika sürüyor.
Bu dosya aynı gerilemeyi SANİYELER içinde yakalar; `docker` işi ise gerilemenin
gerçekten olmadığını kanıtlar. İkisi birbirinin yerine geçmez.

TASARIM: `app`'ten HİÇBİR ŞEY içe aktarılmıyor. `app` içe aktarmak CLIP'i de
yükler (~25 sn, yüzlerce MB — bkz. app/topoloji.py'nin varlık sebebi). Bu dosya
saf `pathlib` + `re` ile çalışır ki tek başına koşturmak bedava olsun.

TÜRKÇE TUZAĞI — BU DOSYANIN EN ÖNEMLİ KURALI:
Bu depoda bir bekçi tam olarak bu yüzden sessizce bozuk kaldı: desen `"eşik"`
diye aranıyordu ama README'de kelime `"eşiği"` diye geçiyordu ve Türkçedeki
k→ğ yumuşaması yüzünden `"eşik"` alt dizgisi o kelimenin içinde HİÇ YOK.
Bekçi hiç tetiklenemezdi. Bu yüzden buradaki her denetim TÜRKÇE DÜZYAZIYA
DEĞİL, ASCII MAKİNE BELİRTECİNE demirlenmiştir: `uvicorn`, `app.kuyruk`,
`entrypoint.sh`, `SERVIS_ROLU`, `eol=lf`. Bunların hiçbiri Türkçe ek almaz.

Ayrıca `re.IGNORECASE` KULLANILMAZ: Python'un küçük harfe çevirmesi Türkçe
farkında değil — `"HEPSİ".lower()` sonucu `"hepsi"` DEĞİL, `i` + U+0307
(birleşen üstteki nokta); `"I".lower()` de `"ı"` değil `"i"`. Bu yüzden rol
adları saf ASCII kalmalı (aşağıdaki ROLLER) ve karşılaştırmalar büyük/küçük
harfe duyarlıdır.
"""
import re
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
DOCKERFILE = KOK / "Dockerfile"
GIRIS = KOK / "entrypoint.sh"
GITATTRIBUTES = KOK / ".gitattributes"
DOCKERIGNORE = KOK / ".dockerignore"
README = KOK / "README.md"
ORNEK_ENV = KOK / ".env.example"
CI = KOK / ".github" / "workflows" / "ci.yml"

# Roller ASCII kalmalı — yukarıdaki IGNORECASE notu.
ROLLER = ("hepsi", "http", "kuyruk")

# Sütun 0'daki CMD: HEALTHCHECK'in devam satırı ("    CMD [ ... ] || curl ...")
# girintili olduğu için bu desen ona TAKILMAZ. Girinti bozulursa test gürültülü
# biçimde kırılır — sessizce yanlış satıra bakmasından iyidir.
CMD_SATIRI = re.compile(r"^CMD\s+(.*)$", re.MULTILINE)

# (?![\w.]) ŞART: onsuz `app.kuyrukk` gibi bir yazım hatası bu bekçiden GEÇERDİ,
# çünkü desen `app.kuyruk`u onun ÖNEKİ olarak bulurdu. Mutasyon M3 bunu kanıtlar.
# Aynı sınıf hata (bir dizginin başka bir kelimenin içinde olması/olmaması) bu
# depoda daha önce bir bekçiyi işlevsiz bıraktı.
UVICORN = re.compile(r"uvicorn\s+app\.main:app(?![\w.:])")
KUYRUK = re.compile(r"python\s+-m\s+app\.kuyruk(?![\w.])")

# ⚠ AŞAĞIDAKİ DÖRT DESENİN NEDEN AYRI OLDUĞU — MUTASYONLA ÖĞRENİLDİ:
# Bu dosyanın ilk hâli yalnızca "betikte `python -m app.kuyruk` GEÇİYOR MU"
# diye soruyordu. Mutasyon sınavında M2 (hepsi dalındaki başlatma satırını sil)
# ve M3 (`app.kuyrukk` yazım hatası) bekçiden KAÇTI: betikte o dizgi bir kez
# daha geçiyor — `kuyruk)` rolündeki `exec python -m app.kuyruk` satırında.
# Yani dizgi vardı ama VARSAYILAN ROL tüketiciyi başlatmıyordu; düzeltmeye
# çalıştığımız gerilemenin ta kendisi bekçinin altından geçiyordu.
#
# Ders, deponun eşik bekçisindekiyle aynı sınıf: "desen tutuyor" ile "doğru
# yerde tutuyor" ayrı şeyler. Bu yüzden her rol AYRI AYRI denetleniyor:
#   arka plan (`... &`) -> "hepsi" rolü, tek `docker run` ile iki süreç
#   `exec ...`          -> "http" / "kuyruk" rolleri, Railway'de servis başına tek süreç
UVICORN_ARKAPLAN = re.compile(
    r"^\s*uvicorn\s+app\.main:app(?![\w.:])[^\n]*&\s*$", re.MULTILINE)
UVICORN_EXEC = re.compile(r"^\s*exec\s+uvicorn\s+app\.main:app(?![\w.:])", re.MULTILINE)
KUYRUK_ARKAPLAN = re.compile(
    r"^\s*python\s+-m\s+app\.kuyruk(?![\w.])[^\n]*&\s*$", re.MULTILINE)
KUYRUK_EXEC = re.compile(r"^\s*exec\s+python\s+-m\s+app\.kuyruk(?![\w.])", re.MULTILINE)


def _metin(yol: Path) -> str:
    # encoding açıkça verilir: Windows'ta varsayılan cp1254 olur ve Türkçe
    # karakterli satırlar okunurken patlar. newline çevirisi de burada olur,
    # yani desenlerde \r? taşımaya gerek kalmaz.
    return yol.read_text(encoding="utf-8")


def _son_cmd() -> str:
    eslesmeler = CMD_SATIRI.findall(_metin(DOCKERFILE))
    assert eslesmeler, "Dockerfile'da sütun 0'da CMD satırı yok"
    return eslesmeler[-1]


# --------------------------------------------------------------------------
# 1-2: asıl gerileme ve sinsi çeşidi
# --------------------------------------------------------------------------

def test_cmd_giris_betigine_isaret_ediyor():
    """CMD doğrudan uvicorn çalıştırmamalı; giriş betiğini çağırmalı."""
    cmd = _son_cmd()
    assert "entrypoint.sh" in cmd, (
        f"CMD giriş betiğine işaret etmiyor: {cmd!r}. Doğrudan uvicorn "
        f"çalıştırmak kuyruk tüketicisini imajdan düşürür.")
    assert not UVICORN.search(cmd), (
        f"CMD hâlâ doğrudan uvicorn başlatıyor: {cmd!r}. Bu, düzeltilen "
        f"gerilemenin ta kendisi — tüketici hiç çalışmaz.")


def test_varsayilan_rol_iki_sureci_de_baslatiyor():
    """CMD betiğe işaret edip betik tüketiciyi unutursa hata aynen geri gelir.

    Varsayılan rol ("hepsi") ikisini de ARKA PLANDA başlatmalı. Yalnızca
    "betikte şu dizgi geçiyor mu" diye sormak YETMEZ — bkz. yukarıdaki
    mutasyon notu.
    """
    betik = _metin(GIRIS)
    assert UVICORN_ARKAPLAN.search(betik), (
        "entrypoint.sh varsayılan rolde uvicorn'u arka planda başlatmıyor")
    assert KUYRUK_ARKAPLAN.search(betik), (
        "entrypoint.sh varsayılan rolde `python -m app.kuyruk` başlatmıyor — "
        "üretimde ilanlar sonsuza kadar ai_status=PENDING kalır. Bu, bu "
        "paketin düzelttiği gerilemenin ta kendisi.")


def test_tekil_roller_dogru_sureci_exec_ediyor():
    """http ve kuyruk rolleri Railway için şart; sessizce bozulmasınlar.

    `exec` olmazsa araya bir kabuk girer ve SIGTERM'i iletmeyi unutabilir;
    o zaman `docker stop` SIGKILL'e düşer.
    """
    betik = _metin(GIRIS)
    assert UVICORN_EXEC.search(betik), (
        "`http` rolü `exec uvicorn app.main:app` ile başlatmıyor")
    assert KUYRUK_EXEC.search(betik), (
        "`kuyruk` rolü `exec python -m app.kuyruk` ile başlatmıyor")


# --------------------------------------------------------------------------
# 3-5: betiğin imaja SAĞLAM girmesi
# --------------------------------------------------------------------------

def test_giris_betiginde_cr_yok():
    """CRLF'li bir kabuk betiği imajda 'bad interpreter' verir.

    CI Linux'ta klonluyor ve dosyayı LF alıyor ⇒ CI YEŞİL KALIR. Kırılan
    yalnızca Windows'taki geliştiricinin lokal build'i. Bu test o asimetriyi
    kapatır: geliştiricinin makinesinde patlar. Bayt olarak okunur —
    read_text() aranan \\r'ı zaten yer.
    """
    ham = GIRIS.read_bytes()
    assert b"\r" not in ham, (
        "entrypoint.sh CRLF içeriyor. .gitattributes zaten var olan bir "
        "dosyayı düzeltmez: `rm entrypoint.sh && git checkout -- entrypoint.sh` "
        "ile yeniden çıkarın, sonra `git ls-files --eol entrypoint.sh` ile w/lf olduğunu doğrulayın.")


def test_dockerfile_giris_betigini_kopyaliyor():
    """CMD, imajda olmayan bir dosyaya işaret ederse konteyner hiç açılmaz."""
    metin = _metin(DOCKERFILE)
    assert re.search(r"^COPY\s+entrypoint\.sh\s", metin, re.MULTILINE), (
        "Dockerfile'da `COPY entrypoint.sh ...` satırı yok")
    # .dockerignore betiği dışarıda bırakmamalı. Dizin desenleri (tests/,
    # scripts/) zararsız; tehlike olan doğrudan .sh ya da entrypoint kalıbı.
    yoksay = _metin(DOCKERIGNORE)
    for satir in yoksay.splitlines():
        satir = satir.strip()
        if not satir or satir.startswith("#"):
            continue
        assert satir not in ("entrypoint.sh", "*.sh", "/entrypoint.sh"), (
            f".dockerignore giriş betiğini dışlıyor: {satir!r}")


def test_gitattributes_kabuk_betiklerini_lf_e_sabitliyor():
    """Bir üstteki testin ön koşulu: her klonda LF gelsin."""
    metin = _metin(GITATTRIBUTES)
    assert re.search(r"^\*\.sh\s+.*eol=lf", metin, re.MULTILINE), (
        ".gitattributes `*.sh ... eol=lf` içermiyor; Windows'ta klonlayan "
        "herkesin lokal build'i 'bad interpreter' ile kırılır")


# --------------------------------------------------------------------------
# 6-7: roller ve HEALTHCHECK
# --------------------------------------------------------------------------

def test_roller_betikte_ve_belgelerde_tutuyor():
    """Kod ve belge birbirinden kaymasın (deponun mevcut bekçi kalıbı)."""
    betik = _metin(GIRIS)
    for rol in ROLLER:
        # `case` dalı: satır başında (boşluklu) `rol)` biçiminde.
        assert re.search(rf"^\s*{rol}\)", betik, re.MULTILINE), (
            f"entrypoint.sh'ta `{rol})` case dalı yok. (Türkçe büyük/küçük "
            f"harf tuzağı yüzünden karşılaştırma harfe duyarlı ve rol adları "
            f"ASCII olmak zorunda.)")
    for belge in (README, ORNEK_ENV):
        assert "SERVIS_ROLU" in _metin(belge), (
            f"{belge.name} SERVIS_ROLU'nden hiç söz etmiyor. Değişkeni bilmeyen "
            f"biri üretimde tek konteynerde iki süreci de çalıştırır ve belleği "
            f"gereksiz yere ikiye katlar.")


def test_healthcheck_portu_gomulu_degil():
    """CMD ${PORT}'u onurlandırıyordu, HEALTHCHECK 8000'i gömüyordu.

    Railway PORT basınca sağlık kontrolü ölü bir porta bakıp konteyneri
    sonsuza kadar 'unhealthy' gösteriyordu.
    """
    metin = _metin(DOCKERFILE)
    assert ":8000/health" not in metin, (
        "HEALTHCHECK portu gömüyor. ${PORT:-8000} kullanılmalı — aksi hâlde "
        "PORT basan ortamlarda konteyner kalıcı olarak unhealthy görünür.")


# --------------------------------------------------------------------------
# 8: asıl kanıtın kendisi silinmesin
# --------------------------------------------------------------------------

def test_ci_imaji_gercekten_calistiriyor():
    """Bu dosya dizgi denetler; imajın ÇALIŞTIĞINI yalnızca CI kanıtlar.

    O iş silinirse hiçbir test kırılmazdı — bu test o boşluğu kapatır.
    """
    metin = _metin(CI)
    assert "patimati-ai:ci" in metin, (
        "ci.yml imajı kurup çalıştıran `docker` işini içermiyor. Bu dosyadaki "
        "denetimler yalnızca metne bakar; imajın gerçekten kalktığını ve "
        "kuyruğun işlediğini yalnızca o iş ölçer.")
    assert "sahte_java.py" in metin, (
        "ci.yml uçtan uca kanıtı (scripts/sahte_java.py) koşturmuyor")
