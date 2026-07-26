# tests/test_indirici.py
"""Adres doğrulama ve çoklu fotoğraf analizi testleri.

Buradaki güvenlik testleri ağa çıkmaz: hepsi adres denetimini, yani isteği
GÖNDERMEDEN ÖNCEKİ kontrolü sınar. Zaten olay da bu — engellenen adrese hiç
istek atılmamalı.
"""
import io

import pytest
from PIL import Image

from app import indirici
from app.analiz import baytlari_analiz_et
from app.hatalar import FotografIndirilemedi, FotografYok, GecersizGoruntu
from app.surum import VEKTOR_BOYUTU

SEED = __import__("pathlib").Path("tests/seed_data")
seed_yok = pytest.mark.skipif(
    not SEED.exists(), reason="seed_data yok — önce scripts/prepare_seed.py")


def jpg(img):
    b = io.BytesIO()
    img.save(b, format="JPEG")
    return b.getvalue()


# --------------------------------------------------------------------------
# SSRF koruması — iç ağa istek attırılamamalı
# --------------------------------------------------------------------------

@pytest.mark.parametrize("url, aciklama", [
    ("https://127.0.0.1/a.jpg",            "loopback"),
    ("https://localhost/a.jpg",            "loopback (alan adı)"),
    ("https://10.0.0.5/a.jpg",             "özel ağ"),
    ("https://192.168.1.1/a.jpg",          "özel ağ"),
    ("https://172.16.0.1/a.jpg",           "özel ağ"),
    ("https://[::1]/a.jpg",                "IPv6 loopback"),
    ("https://169.254.169.254/latest/meta-data/", "bulut kimlik bilgisi ucu"),
    ("https://0.0.0.0/a.jpg",              "belirsiz adres"),
])
def test_yerel_adresler_reddedilir(url, aciklama):
    """Bu adreslerin hiçbirine istek atılmamalı — SSRF'in tam hedefi bunlar."""
    with pytest.raises(FotografIndirilemedi):
        indirici.dogrula(url)


def test_bulut_kimlik_ucu_beyaz_listede_olsa_bile_reddedilir(monkeypatch):
    """169.254.169.254 bulut sunucularında makinenin kimlik bilgilerini döndürür.
    Bunun bizim için hiçbir meşru kullanımı yok; yanlışlıkla beyaz listeye
    yazılsa bile geçmemeli. Tek istisnasız kural bu."""
    monkeypatch.setattr(indirici, "IZINLI_HOSTLAR", ["169.254.169.254"])
    with pytest.raises(FotografIndirilemedi, match="Bağlantı-yerel"):
        indirici.dogrula("https://169.254.169.254/latest/meta-data/iam/")


def test_yerel_adres_beyaz_listedeyse_gecer(monkeypatch):
    """Lokal geliştirme (MinIO, test sunucusu) için izin, ayrı bir bayrakla
    değil beyaz listeye YAZILARAK veriliyor. Böylece izin dar kapsamlı kalıyor."""
    monkeypatch.setattr(indirici, "IZINLI_HOSTLAR", ["127.0.0.1"])
    indirici.dogrula("https://127.0.0.1/a.jpg")          # hata beklenmiyor
    with pytest.raises(FotografIndirilemedi):
        indirici.dogrula("https://10.0.0.5/a.jpg")       # listede olmayan yerel adres


# --------------------------------------------------------------------------
# Şema ve biçim
# --------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "ftp://example.com/a.jpg",
    "gopher://example.com/a.jpg",
    "a.jpg",
])
def test_desteklenmeyen_sema_reddedilir(url):
    with pytest.raises(FotografIndirilemedi):
        indirici.dogrula(url)


def test_http_beyaz_liste_disinda_reddedilir(monkeypatch):
    """Şifresiz http yalnızca açıkça izin verilmiş adresler için (lokal MinIO gibi)."""
    monkeypatch.setattr(indirici, "IZINLI_HOSTLAR", [])
    with pytest.raises(FotografIndirilemedi, match="http yalnızca"):
        indirici.dogrula("http://ornek.com/a.jpg")


def test_alan_adi_olmayan_adres_reddedilir():
    with pytest.raises(FotografIndirilemedi):
        indirici.dogrula("https:///a.jpg")


# --------------------------------------------------------------------------
# Beyaz liste — tek güven mekanizması
# --------------------------------------------------------------------------

def test_beyaz_liste_disindaki_alan_adi_reddedilir(monkeypatch):
    monkeypatch.setattr(indirici, "IZINLI_HOSTLAR", ["patimati-media.s3.amazonaws.com"])
    assert indirici._acikca_izinli("patimati-media.s3.amazonaws.com")
    assert not indirici._acikca_izinli("kotu-site.com")


def test_beyaz_listede_nokta_alt_alan_adlarini_kapsar(monkeypatch):
    monkeypatch.setattr(indirici, "IZINLI_HOSTLAR", [".amazonaws.com"])
    assert indirici._acikca_izinli("bucket.s3.eu-central-1.amazonaws.com")
    assert not indirici._acikca_izinli("amazonaws.com.kotu-site.com")


def test_bos_liste_hicbir_seyi_acikca_izinli_yapmaz(monkeypatch):
    """Liste boşken dış adresler indirilebilir ama hiçbiri 'açıkça izinli'
    sayılmaz — yani yerel adresler ve http yine kapalı kalır."""
    monkeypatch.setattr(indirici, "IZINLI_HOSTLAR", [])
    assert not indirici._acikca_izinli("herhangi-bir-site.com")
    with pytest.raises(FotografIndirilemedi):
        indirici.dogrula("https://127.0.0.1/a.jpg")


# --------------------------------------------------------------------------
# Çoklu fotoğraf
# --------------------------------------------------------------------------

def test_bos_adres_listesi_hata_verir():
    with pytest.raises(FotografYok):
        indirici.hepsini_indir([])


def test_hepsi_basarisizsa_hata_verir():
    with pytest.raises(FotografIndirilemedi):
        indirici.hepsini_indir(["https://127.0.0.1/a.jpg", "file:///x.jpg"])


@seed_yok
def test_coklu_fotograf_analizi():
    fotograflar = [p.read_bytes() for p in sorted((SEED / "class_00").glob("*.jpg"))]
    d = baytlari_analiz_et(fotograflar)
    assert len(d["embeddings"]) == len(fotograflar)
    assert all(len(e) == VEKTOR_BOYUTU for e in d["embeddings"])
    assert d["photo_count"] == len(fotograflar)
    assert d["failed_photos"] == []
    assert d["model_version"]


@seed_yok
def test_bozuk_fotograf_digerlerini_dusurmez():
    """İlanda 3 fotoğraf varsa biri bozuk diye analiz tamamen düşmemeli."""
    iyi = [p.read_bytes() for p in sorted((SEED / "class_00").glob("*.jpg"))][:2]
    bozuk = b"bu bir goruntu degil"
    d = baytlari_analiz_et(iyi + [bozuk])
    assert d["photo_count"] == 2
    assert len(d["failed_photos"]) == 1
    assert d["failed_photos"][0]["index"] == 2


def test_hepsi_bozuksa_hata_verir():
    with pytest.raises(GecersizGoruntu):
        baytlari_analiz_et([b"bozuk", b"yine bozuk"])


@seed_yok
def test_etiketler_en_net_fotograftan_alinir():
    """Bir hayvan bir de manzara fotoğrafı varsa etiketler hayvandan gelmeli."""
    kedi = next((SEED / "class_00").glob("*.jpg")).read_bytes()
    manzara = jpg(Image.new("RGB", (256, 256), (128, 128, 128)))
    d = baytlari_analiz_et([manzara, kedi])
    assert d["species"] == "cat", "etiketler düz gri kareden alınmış"
    assert d["is_pet"] is True, "bir fotoğrafta hayvan varsa is_pet doğru olmalı"
