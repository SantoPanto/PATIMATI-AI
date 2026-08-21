"""Kimlik modeli varsayılanı tek yerden okunsun — üç dosya birbirinden kaymasın.

NEDEN VAR: 19.08.2026'da varsayılan `clip`ten `siglip2-animal`a çevrildi (E3).
Varsayılan ÜÇ ayrı dosyada tekrarlanıyor ve üçü de kuruluma etki ediyor:

    app/surum.py    os.getenv("KIMLIK_MODEL", ...)  — çalışma zamanı varsayılanı
    .env.example    KIMLIK_MODEL=...                — kurulum yapanın kopyaladığı
    Dockerfile      ARG KIMLIK_MODEL=...            — imaja HANGİ ağırlığın gireceği

Üçünün ayrışması sessiz bozulma üretir, gürültülü hata değil:
  - Dockerfile ARG geride kalırsa imaj `clip` ağırlığıyla build edilir, ama
    çalışma zamanı `siglip2-animal` ister ⇒ ilk istekte 1,4 GB indirme başlar.
    İmajın "çalışma zamanında ağ bağımlılığı kalmasın" amacı sessizce ölür.
  - `.env.example` geride kalırsa kurulum yapan kişi yanlış modeli kopyalar ve
    ürettiği vektörler herkesinkiyle KIYASLANAMAZ olur (model_version farkı).

Değeri BİLEREK değiştiriyorsan üçünü birden değiştir; bu bekçi onu zorluyor.

⚠ Bu bekçi ortam değişkenine BAKMAZ, kaynaktaki VARSAYILANA bakar. Bakmak
zorunda: `tests/conftest.py` test oturumunu bilerek `clip`e sabitliyor, yani
`app.surum.SECILEN_KIMLIK` burada her zaman "clip" döner. İçe aktarılan değeri
ölçseydik bekçi kendi ölçmek istediği şeyi kaçırırdı — aynı tuzağa
`test_esik_tutarliligi.py` de aynı gerekçeyle düşmüyor.

KAPSAM DIŞI ve sebebi:
  - `docs/olcum-sonuclari/*.json` ve `docs/olcum-raporu.md`: TARİHSEL ölçüm
    kayıtları. "O ölçüm clip ile yapıldı" cümlesi varsayılan değişince yanlış
    olmaz, doğru kalması gerekir.
  - `.github/workflows/ci.yml`: CI bilerek `clip` koşuyor (ağırlık 1,4 GB).
    Eşitlik değil, AÇIKÇA SABİTLENMİŞ olması denetleniyor — aşağıdaki son test.
"""
import re
from pathlib import Path

from app.surum import KIMLIK_MODELLERI

KOK = Path(__file__).resolve().parent.parent

# `os.getenv("KIMLIK_MODEL", "siglip2-animal")` içindeki VARSAYILANI yakalar.
# `["\']KIMLIK_MODEL["\']` kapanış tırnağı sayesinde aynı dosyadaki
# `os.getenv("KIMLIK_MODEL_YOLU", ...)` çağrısına YANLIŞLIKLA tutmaz.
KOD_VARSAYILANI = re.compile(
    r'os\.getenv\(\s*["\']KIMLIK_MODEL["\']\s*,\s*["\']([\w.\-]+)["\']'
)

# `.env.example` içindeki yorumsuz `KIMLIK_MODEL=...` satırı.
# `(?![\w])` olmadan `KIMLIK_MODEL_YOLU` satırına da tutardı.
ENV_SATIRI = re.compile(r'^[ \t]*KIMLIK_MODEL(?![\w])[ \t]*=[ \t]*([\w.\-]+)[ \t]*$',
                        re.MULTILINE)

DOCKER_ARG = re.compile(r'^[ \t]*ARG[ \t]+KIMLIK_MODEL(?![\w])[ \t]*=[ \t]*([\w.\-]+)[ \t]*$',
                        re.MULTILINE)


def _oku(*parcalar) -> str:
    return (KOK.joinpath(*parcalar)).read_text(encoding="utf-8")


def _tek_eslesme(desen: re.Pattern, metin: str, nerede: str) -> str:
    """Deseni uygular; bulamazsa SESSİZ GEÇMEK yerine burada patlar."""
    eslesmeler = desen.findall(metin)
    assert eslesmeler, (
        f"{nerede} içinde KIMLIK_MODEL varsayılanı bulunamadı. Satırın biçimi "
        f"değiştiyse bu bekçinin deseni de güncellenmeli — 0 eşleşme 'sorun yok' "
        f"demek DEĞİLDİR, bekçinin körleşmesi demek olabilir."
    )
    assert len(eslesmeler) == 1, (
        f"{nerede} içinde KIMLIK_MODEL varsayılanı {len(eslesmeler)} kez "
        f"geçiyor: {eslesmeler}. Varsayılan tek yerde yazılmalı."
    )
    return eslesmeler[0]


def kod_varsayilani() -> str:
    return _tek_eslesme(KOD_VARSAYILANI, _oku("app", "surum.py"), "app/surum.py")


def test_env_example_kodla_ayni_varsayilani_soyluyor():
    """.env.example'daki değer, app/surum.py'deki varsayılanla aynı olmalı."""
    env_degeri = _tek_eslesme(ENV_SATIRI, _oku(".env.example"), ".env.example")
    kod = kod_varsayilani()
    assert env_degeri == kod, (
        f".env.example KIMLIK_MODEL={env_degeri} diyor ama app/surum.py "
        f"varsayılanı '{kod}'. .env.example kurulum yapan kişinin kopyaladığı "
        f"dosya; koddan farklıysa kurulum sessizce BAŞKA bir modelle çalışır ve "
        f"ürettiği vektörler diğerleriyle kıyaslanamaz. İkisini birden değiştir."
    )


def test_dockerfile_arg_kodla_ayni_varsayilani_soyluyor():
    """Dockerfile ARG'ı, app/surum.py varsayılanıyla aynı olmalı.

    Ayrışırsa imaj bir modelin ağırlığıyla build edilir, çalışma zamanı
    başkasını ister ve o ağırlık İLK İSTEKTE indirilmeye başlar.
    """
    arg_degeri = _tek_eslesme(DOCKER_ARG, _oku("Dockerfile"), "Dockerfile")
    kod = kod_varsayilani()
    assert arg_degeri == kod, (
        f"Dockerfile 'ARG KIMLIK_MODEL={arg_degeri}' diyor ama app/surum.py "
        f"varsayılanı '{kod}'. İmaja giren ağırlıkla çalışma zamanının istediği "
        f"model farklı olur: konteyner ilk istekte HuggingFace'e çıkar ve imajın "
        f"'çalışma zamanında ağ bağımlılığı kalmasın' amacı sessizce bozulur."
    )


def test_varsayilan_taninan_bir_secenek():
    """Varsayılan, KIMLIK_MODELLERI sözlüğünde gerçekten var olmalı.

    Yoksa servis içe aktarma anında ValueError ile ölür — ama bu, ancak biri
    servisi çalıştırmayı denediğinde görülür. Burada erken patlasın.
    """
    kod = kod_varsayilani()
    assert kod in KIMLIK_MODELLERI, (
        f"Varsayılan '{kod}' KIMLIK_MODELLERI sözlüğünde yok. "
        f"Tanınan anahtarlar: {sorted(KIMLIK_MODELLERI)}. "
        f"Servis bu hâlde açılışta ValueError ile ölür."
    )


def test_dockerfile_her_kimlik_modelini_taniyor():
    """Dockerfile'daki `case` her seçeneği ağırlık indirmesiyle karşılamalı.

    Dockerfile'ın kendi yorumu bunu zaten şart koşuyor ("KIMLIK_MODELLERI
    sözlüğü değişirse burası da güncellenmeli") ama hiçbir şey denetlemiyordu.
    Eksik kalan seçenek `*)` dalına düşer: build UYARI basıp GEÇER, ağırlık
    imaja girmez ve sorun ancak çalışma zamanında ortaya çıkar.
    """
    dockerfile = _oku("Dockerfile")
    blok = re.search(r'case\s+"\$KIMLIK_MODEL"\s+in(.*?)esac', dockerfile, re.DOTALL)
    assert blok, (
        "Dockerfile'da `case \"$KIMLIK_MODEL\" in ... esac` bloğu bulunamadı. "
        "Yapı değiştiyse bu bekçi de güncellenmeli."
    )

    # Dal etiketleri: satır başında `clip)` / `siglip2-animal)` biçiminde.
    dallar = set(re.findall(r'^[ \t]*([\w.\-]+)\)', blok.group(1), re.MULTILINE))
    eksik = sorted(set(KIMLIK_MODELLERI) - dallar)
    assert not eksik, (
        f"Dockerfile'daki `case` şu seçenekleri tanımıyor: {eksik}. "
        f"Bulunan dallar: {sorted(dallar)}. Tanınmayan seçenek `*)` dalına düşer, "
        f"build sessizce geçer ve ağırlık imaja GİRMEZ."
    )


# ci.yml'de KIMLIK_MODEL ÜÇ ayrı biçimde sabitleniyor ve üçü ÜÇ AYRI şeyi
# koruyor. Bu yüzden üçü ayrı ayrı denetleniyor.
#
# 🔬 Neden böyle: bu bekçinin ilk iki hâli mutasyon denemesinde KÖR ÇIKTI.
#   1. hâl "toplamda en az bir pin var mı" diye baktı -> iş seviyesindeki pini
#      sildim, diğer üçü yetti, bekçi YEŞİL kaldı.
#   2. hâl ":" ve "=" biçimlerini ayırdı -> build-args pinini sildim, geriye
#      kalan iki `-e` satırı "=" biçimini doyurdu, bekçi YİNE YEŞİL kaldı.
# Ders: "desen tutuyor" ile "doğru yerde tutuyor" aynı şey değil.
CI_ENV_PINI = re.compile(r'^[ \t]*KIMLIK_MODEL(?![\w])[ \t]*:[ \t]*([\w.\-]+)')
CI_BUILDARG_PINI = re.compile(r'^[ \t]*KIMLIK_MODEL(?![\w])[ \t]*=[ \t]*([\w.\-]+)')
CI_RUN_PINI = re.compile(r'-e[ \t]+KIMLIK_MODEL(?![\w])[ \t]*=[ \t]*([\w.\-]+)')


def _ci_pinleri() -> dict:
    """Biçim -> bulunan değerler. Yorum satırları sayılmaz."""
    bulunan = {"is_env": [], "build_arg": [], "docker_run": []}
    for satir in _oku(".github", "workflows", "ci.yml").splitlines():
        if satir.lstrip().startswith("#"):
            continue          # yorumdaki örnek pin, pin değildir
        bulunan["is_env"] += CI_ENV_PINI.findall(satir)
        bulunan["docker_run"] += CI_RUN_PINI.findall(satir)
        # build-arg satırı ÇIPLAK olmalı: `-e ...` satırları buraya sayılmasın.
        if "-e " not in satir:
            bulunan["build_arg"] += CI_BUILDARG_PINI.findall(satir)
    return bulunan


def test_ci_pytest_icin_kimlik_modelini_sabitliyor():
    """İş seviyesindeki `KIMLIK_MODEL: <değer>` pini durmalı.

    Bu pin pytest'in hangi modelle koştuğunu sabitliyor. `tests/conftest.py` de
    `clip` sabitliyor ama ci.yml'nin kendi yorumunun dediği gibi iki katman
    BİLEREK var: conftest'teki varsayılan ileride değişirse CI sessizce
    1,4 GB'lık ağırlığı indirmeye başlamasın.
    """
    assert _ci_pinleri()["is_env"], (
        "ci.yml'de iş seviyesinde `KIMLIK_MODEL: <değer>` pini yok. Silinirse "
        "koruma tek katmana (tests/conftest.py) iner ve o değişince CI 1,4 GB indirir."
    )


def test_ci_imaji_hangi_agirlikla_build_ettigini_sabitliyor():
    """`build-args:` altındaki çıplak `KIMLIK_MODEL=<değer>` pini durmalı.

    İmaja HANGİ ağırlığın gireceğini yalnız bu belirliyor. Silinirse Dockerfile'ın
    ARG varsayılanı devralır — yani varsayılan model — ve her CI koşumu 1,4 GB'lık
    ağırlığı indirmeye başlar. `-e` pinleri bunun yerini TUTMAZ: onlar çalışma
    zamanına etki eder, imajın içine ne girdiğine değil.
    """
    assert _ci_pinleri()["build_arg"], (
        "ci.yml'de `build-args:` altında `KIMLIK_MODEL=<değer>` pini yok. İmaj "
        "Dockerfile'daki ARG varsayılanıyla build edilir ⇒ her koşumda 1,4 GB'lık "
        "ağırlık indirilir. `-e KIMLIK_MODEL=...` satırları bunu KARŞILAMAZ."
    )


def test_ci_konteyneri_hangi_modelle_kostugunu_sabitliyor():
    """`docker run -e KIMLIK_MODEL=<değer>` pini durmalı.

    İmajdaki ENV zaten build-arg'dan geliyor, yani bu pin ikinci katman; ama
    silinirse konteynerin hangi modelle koştuğu artık ci.yml'den okunamaz ve
    imaj varsayılanı değişince kimse fark etmez.
    """
    assert _ci_pinleri()["docker_run"], (
        "ci.yml'de `docker run` adımlarında `-e KIMLIK_MODEL=<değer>` pini yok. "
        "Konteynerin hangi modelle koştuğu iş akışından okunamaz hâle gelir."
    )


def test_ci_pinleri_birbiriyle_ayni():
    """Üç biçimdeki bütün pinler TEK bir değeri göstermeli.

    Ayrışırlarsa aynı koşumun farklı adımları farklı modelle çalışır: imaj bir
    modelle build edilip başka bir modelle koşturulabilir ve yeşilin ne anlama
    geldiği belirsizleşir.
    """
    pinler = _ci_pinleri()
    hepsi = set(pinler["is_env"]) | set(pinler["build_arg"]) | set(pinler["docker_run"])
    assert len(hepsi) == 1, (
        f"ci.yml KIMLIK_MODEL'i farklı değerlere sabitliyor: {sorted(hepsi)} "
        f"(biçim kırılımı: {pinler}). Aynı koşumun adımları farklı modelle çalışır."
    )
