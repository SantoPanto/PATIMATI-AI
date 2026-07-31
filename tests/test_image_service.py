# tests/test_image_service.py
"""`app/services/image_service.py` için testler.

`scripts/train.py`, `scripts/prepare_dataset.py` ve `scripts/evaluate_baseline.py`
hepsi buradaki tarama/yükleme mantığına dayanıyor; klasör yapısı yanlış
yorumlanırsa (ör. bozuk dosya sessizce galeri'ye girerse) sonuçlar sessizce
yanlış çıkar, hata vermez. O yüzden burada asıl önemli olan "sessiz yanlış"
senaryolarını yakalamak.
"""
import pytest
from PIL import Image

from app.services.image_service import (
    MIN_EDGE,
    load_rgb_image,
    scan_animal_dataset,
)


def _jpg(path, size=(64, 64), color=(10, 20, 30)):
    Image.new("RGB", size, color).save(path)


# --------------------------------------------------------------------------
# scan_animal_dataset
# --------------------------------------------------------------------------

def test_klasor_yoksa_hata_verir(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_animal_dataset(tmp_path / "yok")


def test_gorseller_animal_id_ile_eslesir(tmp_path):
    """Her alt klasör bir animal_id — görüntüler doğru id'ye atanmalı."""
    (tmp_path / "kedi1").mkdir()
    (tmp_path / "kedi2").mkdir()
    _jpg(tmp_path / "kedi1" / "a.jpg")
    _jpg(tmp_path / "kedi1" / "b.png")
    _jpg(tmp_path / "kedi2" / "a.jpg")

    scan = scan_animal_dataset(tmp_path)

    by_animal = {}
    for img in scan.images:
        by_animal.setdefault(img.animal_id, []).append(img.path.name)
    assert sorted(by_animal["kedi1"]) == ["a.jpg", "b.png"]
    assert by_animal["kedi2"] == ["a.jpg"]


def test_kok_klasordeki_basibos_dosya_yoksayilir(tmp_path):
    """`animal_id` klasörüne ait olmayan bir dosya (kök klasörde) bir hayvana
    atanamaz — sessizce yanlış bir animal_id'ye eklenmek yerine `stray_files`'a
    düşmeli."""
    (tmp_path / "kedi1").mkdir()
    _jpg(tmp_path / "kedi1" / "a.jpg")
    _jpg(tmp_path / "stray.jpg")

    scan = scan_animal_dataset(tmp_path)

    assert len(scan.images) == 1
    assert [p.name for p in scan.stray_files] == ["stray.jpg"]


def test_taninmayan_uzanti_yoksayilir(tmp_path):
    (tmp_path / "kedi1").mkdir()
    _jpg(tmp_path / "kedi1" / "a.jpg")
    (tmp_path / "kedi1" / "notlar.txt").write_text("bu bir goruntu degil")

    scan = scan_animal_dataset(tmp_path)

    assert len(scan.images) == 1
    assert [p.name for p in scan.stray_files] == ["notlar.txt"]


def test_bozuk_dosya_unreadable_olarak_raporlanir(tmp_path):
    """Bozuk bir dosya sessizce ScannedImage listesine girip sonradan eğitimi
    çökertmemeli — tarama sırasında ayıklanıp raporlanmalı."""
    (tmp_path / "kedi1").mkdir()
    _jpg(tmp_path / "kedi1" / "iyi.jpg")
    (tmp_path / "kedi1" / "bozuk.jpg").write_bytes(b"bu bir jpg degil")

    scan = scan_animal_dataset(tmp_path)

    assert [i.path.name for i in scan.images] == ["iyi.jpg"]
    assert [p.name for p in scan.unreadable_files] == ["bozuk.jpg"]


def test_gecerli_goruntusu_olmayan_klasor_bos_olarak_raporlanir(tmp_path):
    (tmp_path / "kedi1").mkdir()
    _jpg(tmp_path / "kedi1" / "a.jpg")
    (tmp_path / "kedi_bos").mkdir()

    scan = scan_animal_dataset(tmp_path)

    assert scan.empty_animal_dirs == ["kedi_bos"]


def test_gizli_dosya_ve_klasorler_atlanir(tmp_path):
    (tmp_path / "kedi1").mkdir()
    _jpg(tmp_path / "kedi1" / "a.jpg")
    (tmp_path / ".DS_Store").write_bytes(b"x")
    hidden_dir = tmp_path / ".hidden_animal"
    hidden_dir.mkdir()
    _jpg(hidden_dir / "a.jpg")

    scan = scan_animal_dataset(tmp_path)

    assert {i.animal_id for i in scan.images} == {"kedi1"}
    assert scan.stray_files == []


# --------------------------------------------------------------------------
# load_rgb_image
# --------------------------------------------------------------------------

def test_cok_kucuk_goruntu_reddedilir(tmp_path):
    """1x1 gibi bir görüntü anlamlı bir embedding üretmez — app/embedder.py'deki
    ASGARI_KENAR kontrolüyle aynı eşik burada da uygulanmalı."""
    path = tmp_path / "kucuk.jpg"
    _jpg(path, size=(8, 8))
    with pytest.raises(ValueError):
        load_rgb_image(path)


def test_asgari_boyut_kabul_edilir(tmp_path):
    path = tmp_path / "tam.jpg"
    _jpg(path, size=(MIN_EDGE, MIN_EDGE))
    img = load_rgb_image(path)
    assert img.size == (MIN_EDGE, MIN_EDGE)
    assert img.mode == "RGB"


def test_seffaf_png_beyaza_yerlesir(tmp_path):
    """Şeffaflığı düz RGB'ye çevirmek alanları siyaha çevirir; beyaz zemine
    yerleştirmek gerekir (bkz. app/embedder.py'deki aynı kontrol)."""
    path = tmp_path / "seffaf.png"
    Image.new("RGBA", (64, 64), (255, 0, 0, 0)).save(path)  # tamamen şeffaf kırmızı
    img = load_rgb_image(path)
    assert img.mode == "RGB"
    assert img.getpixel((32, 32)) == (255, 255, 255)


def test_exif_dondurmesi_uygulanir(tmp_path):
    path = tmp_path / "exif.jpg"
    img = Image.new("RGB", (128, 64), (10, 20, 30))
    exif = img.getexif()
    exif[274] = 6  # Orientation: 90 derece döndür
    img.save(path, format="JPEG", exif=exif.tobytes())

    acilan = load_rgb_image(path)
    assert acilan.size == (64, 128), "EXIF döndürmesi uygulanmadı"
