# app/models.py
# Alan adları docs/entegrasyon-sozlesmesi.md ile hizalıdır (ad_id, model_version...).
from pydantic import BaseModel, Field, model_validator


class AnalyzeResponse(BaseModel):
    embedding: list[float]          # kimlik vektörü (CLIP 512 / SigLIP2 768)
    labels: list[str]               # Eşleştirme skorunda kullanılan etiketler
    species: str                    # "cat" | "dog" | "unknown"
    species_confidence: float = 0.0
    is_pet: bool = True             # False → fotoğrafta kedi/köpek görünmüyor
    breed: str | None = None        # Güven eşiğinin altındaysa None
    breed_confidence: float = 0.0
    pattern: str | None = None      # tabby | spotted | solid | bicolor | unknown
    colors: list[dict]              # Dominant renkler
    model_version: str              # Hangi model/ön işlemeyle üretildi
    # BİLGİ AMAÇLI -- ⚠ ÖLÇÜLMEDİ (bkz. app/attributes.py:TASARIM_PROMPTS).
    # Ne eşleştirmeyi ne de bu yanıttaki diğer alanları etkiler; ekranda
    # gösterme/eşleştirmede kullanma kararı ayrıca, ölçümden sonra alınacak.
    is_designed_graphic: bool = False
    graphic_confidence: float = 0.0


class AnalyzeUrlRequest(BaseModel):
    """Kuyruk mesajının taşıdığı biçim: dosya değil adres.

    Üretim akışında fotoğraflar S3'te durur ve mesajda yalnızca adresleri
    taşınır — büyük dosyaları kuyruktan geçirmemek için.
    """
    photo_urls: list[str] = Field(min_length=1, max_length=5)


class MatchCandidate(BaseModel):
    # Faz 2 (Instagram entegrasyonu) öncesi `ad_id` zorunluydu. Artık iki
    # kaynaktan biri kimliklendirebilir: ya `ad_id` (native PatiMati ilanı)
    # ya da `external_record_id` (Instagram kökenli external_pet_records
    # kaydı) — TAM OLARAK biri dolu olmalı. Bu, ne matcher.py'nin ne de bu
    # şemanın Instagram'a özel bilgi taşıması gerektiği anlamına gelmez:
    # skorlama iki kimlik türünü de aynı şekilde ele alır, yalnızca hangi
    # tarafa ait olduğunu ayırt eder (bkz. adaylari_eslestir).
    ad_id: int | None = None
    external_record_id: int | None = None
    # İlanın HER fotoğrafı için bir vektör. Tek fotoğrafla eşleşme oranı gerçek
    # veride %24'te kaldığı için çoklu fotoğraf zorunlu hale geldi — görsel skor
    # en iyi fotoğraf çiftinden alınır (bkz. docs/olcum-raporu.md).
    embeddings: list[list[float]] = Field(min_length=1)
    labels: list[str] = Field(default_factory=list)
    species: str = "unknown"
    # None = mesafe BİLİNMİYOR. "0 km" DEĞİL — eskiden varsayılan 0.0 idi ve
    # location_score(0.0) maksimum puan (1.0) döndürdüğü için alanı hiç
    # göndermeyen bir çağıran skoruna sessizce +0.15'e kadar bonus alıyordu.
    # Artık eksik mesafe, geçersiz mesafeyle (NaN/inf) aynı davranır: 0.0.
    distance_km: float | None = None
    # Adayın vektörü hangi model sürümüyle üretildi. Farklıysa aday ATLANIR:
    # farklı sürümlerin vektörleri kıyaslanamaz ve sessizce yanlış sonuç üretir.
    # None → sürüm bilinmiyor, eski kayıt sayılır ve yine atlanır.
    model_version: str | None = None

    @model_validator(mode="after")
    def _tam_olarak_bir_kimlik(self):
        """Bir aday ya native bir ilan ya da external bir kayıttır — ikisi
        birden değil, hiçbiri de değil. İkisi de boş kalırsa adaylari_eslestir
        içindeki kimlik bazlı tekrar/kendisi eleme mantığı ayırt edemeyeceği
        adayları sessizce aynı kayıt sanabilir (bkz. matcher.py:_aday_kimligi)."""
        if (self.ad_id is None) == (self.external_record_id is None):
            raise ValueError(
                "tam olarak biri dolu olmalı: ad_id veya external_record_id")
        return self


class MatchRequest(BaseModel):
    embeddings: list[list[float]] = Field(min_length=1)
    labels: list[str] = Field(default_factory=list)
    species: str = "unknown"
    candidates: list[MatchCandidate] = Field(default_factory=list)
    # Verilirse aday listesindeki aynı kimlikli kayıt elenir (ilan kendisiyle
    # eşleşmesin). Java yanlışlıkla gönderirse %100 skorla eşleşme çıkardı.
    ad_id: int | None = None
    # Faz 2: sorgunun kendisi de bir external_pet_records kaydı olabilir
    # (Instagram Aşama 2 eşleştirmesi — bkz. matcher.py). `ad_id` ile birlikte
    # dolu olmamalı.
    external_record_id: int | None = None
    # Kaynağa özel eşik. Boş bırakılırsa matcher.py kendi ortam değişkeni
    # varsayılanını (MATCH_THRESHOLD) kullanır — native davranış hiç değişmez.
    # Instagram tarafı için Java bunu ai.matching.instagram-threshold ile
    # doldurabilir; 0.80 hiçbir yerde Instagram için doğrulanmış kesin bir
    # değer olarak sabitlenmemiştir.
    match_threshold: float | None = None

    @model_validator(mode="after")
    def _en_fazla_bir_kimlik(self):
        """MatchCandidate/KuyrukIstegi'ndeki _tam_olarak_bir_kimlik'in AYNISI
        DEĞİL, kasıtlı olarak daha gevşek: burada İKİSİ DE boş kalabilir --
        self-exclusion o zaman devre dışı kalır (bkz. matcher.py:
        sorgu_kimligi == (None, None) ise "kendisi" eleme hiç çalışmaz).
        Bu KASITLI ve mevcut testlerle zaten doğrulanmış bir davranış
        (test_match_bos_aday_listesiyle_bos_sonuc_doner,
        test_match_bozuk_sorgu_embeddingi_400_doner -- ikisi de ne ad_id ne
        external_record_id göndermiyor). O yüzden "tam olarak biri" değil,
        yalnızca "ikisi BİRDEN değil" zorunlu kılınır -- asıl rapor edilen
        bug zaten bu: ikisi birden dolu geldiğinde matcher.py._aday_kimligi
        ad_id'yi sessizce önceliklendiriyordu (bkz. matcher.py:281-287,
        332-334), external_record_id sorgusu yanlış kimlikle
        eşleştiriliyordu."""
        if self.ad_id is not None and self.external_record_id is not None:
            raise ValueError(
                "ad_id ve external_record_id aynı anda dolu olamaz")
        return self


class KuyrukIstegi(BaseModel):
    """`ai.analysis.request` kuyruğundan gelen mesaj (sözleşme §3).

    Doğrulamada bilinçli olarak GEVŞEK davranıyoruz — "gönderene sıkı, alana
    hoşgörülü" kuralı (bkz. matcher.py'deki tür karşılaştırması notu):

    - `photo_urls` boş gelebilir: sözleşme bunun karşılığını Pydantic hatası
      değil, `NO_PHOTOS` hata kodu olarak tanımlıyor. Doğrulamada zorlarsak
      Java'ya anlamsız bir "INVALID_REQUEST" döner.
    - `photo_urls` 5'ten uzun gelebilir: indirici fazlasını kırpıp günlüğe
      yazıyor (`app/indirici.py: AZAMI_FOTOGRAF`). Mesajı tümden düşürmek,
      işlenebilecek 5 fotoğrafı da çöpe atmak olurdu.
    - `ad_type` sözleşmede zorunlu ama AI tarafında KULLANILMIYOR (adayları
      Java süzüyor, §5). Kullanmadığımız bir alan yüzünden analizi düşürmek
      kötü takas; varsayılanı var.

    Faz 2 (Instagram) ekleri — hepsi opsiyonel, eski mesajlar hiçbir değişiklik
    olmadan aynı şekilde doğrulanmaya devam eder:

    - `ad_id` artık zorunlu değil: bir external_pet_records kaydı için Java
      `external_record_id` gönderir, `ad_id` boş kalır.
    - `source`: "PATIMATI" (varsayılan) | "INSTAGRAM" — yalnızca izlenebilirlik
      içindir, AI tarafı davranışını DEĞİŞTİRMEZ (Java hangi tabloya
      yazacağına bunun yerine `ad_id`/`external_record_id`'nin hangisinin
      dolu olduğuna bakarak karar verir).
    - `caption` / `triggering_comment`: yalnızca Instagram kökenli istekler
      doldurur. İkisi de boşsa (native ilan akışı gibi) metin analizi hiç
      çalışmaz ve `nlp_attributes`/`extracted_features` eskisi gibi boş
      döner (bkz. kuyruk.py).
    - `match_threshold`: bkz. MatchRequest.
    """
    schema_version: int
    request_id: str
    ad_id: int | None = None
    external_record_id: int | None = None
    source: str = "PATIMATI"
    ad_type: str = "LOST"
    declared_species: str | None = None
    photo_urls: list[str] = Field(default_factory=list)
    candidates: list[MatchCandidate] = Field(default_factory=list)
    caption: str | None = None
    triggering_comment: str | None = None
    match_threshold: float | None = None

    @model_validator(mode="after")
    def _tam_olarak_bir_kimlik(self):
        if (self.ad_id is None) == (self.external_record_id is None):
            raise ValueError(
                "tam olarak biri dolu olmalı: ad_id veya external_record_id")
        return self


class MatchResult(BaseModel):
    ad_id: int | None = None
    external_record_id: int | None = None
    score: float
    visual: float
    label: float
    location: float
    match: bool
    photo_a: int | None = None   # eşleşen fotoğraf çiftinin indeksleri
    photo_b: int | None = None


class PetDegerGuven(BaseModel):
    """{deger, guven} ikilisini taşıyan alanlar için ortak şekil
    (renk_tarifi, goz_rengi) -- pet_raporu.py'nin ürettiği rapor."""
    deger: str
    guven: float = 0.0


class PetYasTahmini(BaseModel):
    aralik: str                 # "4-6 ay" | "2-4 yaş" (bkz. prompt: YAŞ FORMATI)
    yasam_evresi: str           # "Yavru" | "Genç" | "Yetişkin" | "Yaşlı"
    guven: float = 0.0


class PetCinsiyetTahmini(BaseModel):
    tahmin: str                 # "Erkek" | "Dişi" | "Belirsiz"
    guven: float = 0.0


class PetBoyutTahmini(BaseModel):
    deger: str
    guven: float = 0.0


class PetBakimIpuclari(BaseModel):
    beslenme: str | None = None
    tuy_bakimi: str | None = None
    aktivite: str | None = None


class PetReportResult(BaseModel):
    """`app/pet_raporu.py::PetReportAnalyzer.analyze()` çıktısı.

    Alan adları/anlamları `app/pet_raporu_prompt.py`'deki ÇIKTI FORMATI ile
    birebir eşleşir. `gecerli=False` ise diğer tüm analiz alanları None/boş
    kalır (prompt'un kendi kuralı) -- burada da bilerek hepsi opsiyonel.
    """
    gecerli: bool
    # Yalnızca prompt'un ALAN KURALLARI'ndaki 6 sabit değerden biri OLMALI;
    # ayrıca pet_raporu.py'nin KENDİ ürettiği, LLM'e hiç gitmeyen dahili bir
    # sebep de olabilir ("SERVIS_KULLANILAMIYOR" -- sağlayıcı yapılandırılmamış
    # ya da çağrı başarısız oldu). Bu yüzden bilerek Literal değil, düz str.
    hata_nedeni: str | None = None
    irka_ozel_icerik: bool | None = None
    renk_tarifi: PetDegerGuven | None = None
    goz_rengi: PetDegerGuven | None = None
    tahmini_yas: PetYasTahmini | None = None
    cinsiyet: PetCinsiyetTahmini | None = None
    tahmini_boyut: PetBoyutTahmini | None = None
    ayirt_edici_isaretler: list[str] = Field(default_factory=list)
    genel_durum_gozlemi: str | None = None
    karakter_profili: str | None = None
    sasirtici_bilgiler: list[str] = Field(default_factory=list)
    dikkat_edilmesi_gerekenler: list[str] = Field(default_factory=list)
    bakim_ipuclari: PetBakimIpuclari | None = None
    ek_hayvanlar: str | None = None
    goruntu_kalite_notu: str | None = None


class TextAnalysisResult(BaseModel):
    """`app/metin_analiz.py::TextAnalyzer.analyze()` çıktısı.

    Kaynak metinde olmayan bilgi UYDURULMAZ — çıkarılamayan her alan None
    kalır (bkz. metin_analiz.py). `category`/`category_confidence` dışında
    hiçbir alan zorunlu değildir.
    """
    category: str = "UNCERTAIN"   # LOST | FOUND | ADOPTION | IRRELEVANT | UNCERTAIN
    category_confidence: float = 0.0
    species: str | None = None
    breed: str | None = None
    colors: list[str] = Field(default_factory=list)
    gender: str | None = None
    age_text: str | None = None
    pet_name: str | None = None
    location_text: str | None = None
    location_confidence: float | None = None
    event_date: str | None = None          # ISO tarih metni, kesin değilse None
    event_date_estimated: bool = False
    distinguishing_features: str | None = None
    pet_count: int | None = None
    needs_review: bool = False
