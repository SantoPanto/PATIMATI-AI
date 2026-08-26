import json
import os
from extractor import PatiMatiTextExtractor
TEST_DATASET = [
    {"text": "Bahçede bulduk, gri tekir kedi, boynunda mavi tasması var.", "expected": {"tur": "kedi", "detay_irk": "tekir", "tasma": "mavi", "kulak": "Bilinmiyor"}},
    {"text": "Sokakta geziyordu, siyah kedicik, tasması yok.", "expected": {"tur": "kedi", "detay_irk": "Bilinmiyor", "tasma": "yok", "kulak": "Bilinmiyor"}},
    {"text": "Sol kulağı çentikli sarı pisi, tasması yok.", "expected": {"tur": "kedi", "detay_irk": "Bilinmiyor", "tasma": "yok", "kulak": "çentikli/küpeli"}},
    {"text": "Kırmızı tasmalı sevimli bir köpek.", "expected": {"tur": "köpek", "detay_irk": "Bilinmiyor", "tasma": "kırmızı", "kulak": "Bilinmiyor"}},
    {"text": "Kayıp köpeğimiz, üzerinde tasma bulunmuyor.", "expected": {"tur": "köpek", "detay_irk": "Bilinmiyor", "tasma": "yok", "kulak": "Bilinmiyor"}},
    {"text": "Evden kaçtı, tekir kedi, yeşil tasmalı.", "expected": {"tur": "kedi", "detay_irk": "tekir", "tasma": "yeşil", "kulak": "Bilinmiyor"}},
    {"text": "Kulakları küpeli sokak köpeği.", "expected": {"tur": "köpek", "detay_irk": "Bilinmiyor", "tasma": "Bilinmiyor", "kulak": "çentikli/küpeli"}},
    {"text": "Tekir cinsi kedimiz kayboldu, tasmasından eser yok.", "expected": {"tur": "kedi", "detay_irk": "tekir", "tasma": "yok", "kulak": "Bilinmiyor"}},
    {"text": "Küçük pisi, siyah renk, boynunda pembe tasma var.", "expected": {"tur": "kedi", "detay_irk": "Bilinmiyor", "tasma": "pembe", "kulak": "Bilinmiyor"}},
    {"text": "Yaralı bir kedi bulduk, tasmasi yok.", "expected": {"tur": "kedi", "detay_irk": "Bilinmiyor", "tasma": "yok", "kulak": "Bilinmiyor"}},
    {"text": "Köpek havlıyordu mahallede, siyah tasmalı.", "expected": {"tur": "köpek", "detay_irk": "Bilinmiyor", "tasma": "siyah", "kulak": "Bilinmiyor"}},
    {"text": "Çentikli kulaklı kedi.", "expected": {"tur": "kedi", "detay_irk": "Bilinmiyor", "tasma": "Bilinmiyor", "kulak": "çentikli/küpeli"}},
    {"text": "Pisi kayboldu, herhangi bir tasması yok.", "expected": {"tur": "kedi", "detay_irk": "Bilinmiyor", "tasma": "yok", "kulak": "Bilinmiyor"}},
    {"text": "Sevimli köpüş, mavi tasmayla geziyor.", "expected": {"tur": "köpek", "detay_irk": "Bilinmiyor", "tasma": "mavi", "kulak": "Bilinmiyor"}},
    {"text": "Tekir kedi, kulak çentiği var.", "expected": {"tur": "kedi", "detay_irk": "tekir", "tasma": "Bilinmiyor", "kulak": "çentikli/küpeli"}},
    {"text": "Bahçedeki kedi kedicik.", "expected": {"tur": "kedi", "detay_irk": "Bilinmiyor", "tasma": "Bilinmiyor", "kulak": "Bilinmiyor"}},
    {"text": "Sahipsiz köpek, üzerinde tasma bulunmuyor.", "expected": {"tur": "köpek", "detay_irk": "Bilinmiyor", "tasma": "yok", "kulak": "Bilinmiyor"}},
    {"text": "Gri tekir, tasmasız.", "expected": {"tur": "kedi", "detay_irk": "tekir", "tasma": "yok", "kulak": "Bilinmiyor"}},
    {"text": "Kırmızı tasmalı tatlı kedi.", "expected": {"tur": "kedi", "detay_irk": "Bilinmiyor", "tasma": "kırmızı", "kulak": "Bilinmiyor"}},
    {"text": "Büyük köpek, kulakları küpeli.", "expected": {"tur": "köpek", "detay_irk": "Bilinmiyor", "tasma": "Bilinmiyor", "kulak": "çentikli/küpeli"}},
    {"text": "Pisi pisi, yeşil tasması var.", "expected": {"tur": "kedi", "detay_irk": "Bilinmiyor", "tasma": "yeşil", "kulak": "Bilinmiyor"}},
    {"text": "Tekir kedi bulduk, tasması yok.", "expected": {"tur": "kedi", "detay_irk": "tekir", "tasma": "yok", "kulak": "Bilinmiyor"}},
    {"text": "Sokak köpeği, tasması mevcut değil.", "expected": {"tur": "köpek", "detay_irk": "Bilinmiyor", "tasma": "yok", "kulak": "Bilinmiyor"}},
    {"text": "Siyah tekir kedi.", "expected": {"tur": "kedi", "detay_irk": "tekir", "tasma": "Bilinmiyor", "kulak": "Bilinmiyor"}},
    {"text": "Mavi tasmalı köpek.", "expected": {"tur": "köpek", "detay_irk": "Bilinmiyor", "tasma": "mavi", "kulak": "Bilinmiyor"}},
    {"text": "Kulak çentiği bulunan tekir.", "expected": {"tur": "kedi", "detay_irk": "tekir", "tasma": "Bilinmiyor", "kulak": "çentikli/küpeli"}},
    # Yeni eklenen gerçek ilan dili örnekleri (27 - 42):
    {"text": "Kahverengi tasmalı sevimli kedi kayıp.", "expected": {"tur": "kedi", "detay_irk": "Bilinmiyor", "tasma": "kahverengi", "kulak": "Bilinmiyor"}},
    {"text": "Küçük köpeğimizin kulağında çentik var, boynunda tasma yok.", "expected": {"tur": "köpek", "detay_irk": "Bilinmiyor", "tasma": "yok", "kulak": "çentikli/küpeli"}},
    {"text": "Tekir kedi, kulak çentiği yok.", "expected": {"tur": "kedi", "detay_irk": "tekir", "tasma": "Bilinmiyor", "kulak": "normal"}},
    {"text": "Sokakta bulduk sarı pisi, tasması da var.", "expected": {"tur": "kedi", "detay_irk": "Bilinmiyor", "tasma": "var", "kulak": "Bilinmiyor"}},
    {"text": "Beyaz tasmalı büyük bir köpek.", "expected": {"tur": "köpek", "detay_irk": "Bilinmiyor", "tasma": "beyaz", "kulak": "Bilinmiyor"}},
    {"text": "Tasmasından eser olmayan tekir pisi.", "expected": {"tur": "kedi", "detay_irk": "tekir", "tasma": "yok", "kulak": "Bilinmiyor"}},
    {"text": "Küpeli ve sarı tasmalı köpek.", "expected": {"tur": "köpek", "detay_irk": "Bilinmiyor", "tasma": "sarı", "kulak": "çentikli/küpeli"}},
    {"text": "Gri kedi, kulaklarında çentik mevcut.", "expected": {"tur": "kedi", "detay_irk": "Bilinmiyor", "tasma": "Bilinmiyor", "kulak": "çentikli/küpeli"}},
    {"text": "Kayıp tatlı köpüş, tasmasi bulunmuyor.", "expected": {"tur": "köpek", "detay_irk": "Bilinmiyor", "tasma": "yok", "kulak": "Bilinmiyor"}},
    {"text": "Pembe tasmalı tekir kedi.", "expected": {"tur": "kedi", "detay_irk": "tekir", "tasma": "pembe", "kulak": "Bilinmiyor"}},
    {"text": "Kulak çentiği ve kırmızı tasması var.", "expected": {"tur": "Bilinmiyor", "detay_irk": "Bilinmiyor", "tasma": "kırmızı", "kulak": "çentikli/küpeli"}},
    {"text": "Tasmasız ve kulakları normal sokak kedisi.", "expected": {"tur": "kedi", "detay_irk": "Bilinmiyor", "tasma": "yok", "kulak": "normal"}},
    {"text": "Acil araniyor: Siyah tasmalı tekir kedi.", "expected": {"tur": "kedi", "detay_irk": "tekir", "tasma": "siyah", "kulak": "Bilinmiyor"}},
    {"text": "Kulak çentiği olan köpek, tasmasız.", "expected": {"tur": "köpek", "detay_irk": "Bilinmiyor", "tasma": "yok", "kulak": "çentikli/küpeli"}},
    {"text": "Mavi tasmalı tekir cinsi kedi.", "expected": {"tur": "kedi", "detay_irk": "tekir", "tasma": "mavi", "kulak": "Bilinmiyor"}},
    {"text": "Küpeli pisi, tasması yok.", "expected": {"tur": "kedi", "detay_irk": "Bilinmiyor", "tasma": "yok", "kulak": "çentikli/küpeli"}}
]

def run_evaluation():
    extractor = PatiMatiTextExtractor()
    correct = 0
    total = len(TEST_DATASET)
    results_log = []

    for i, item in enumerate(TEST_DATASET):
        pred = extractor.extract_features(item["text"])
        is_match = pred == item["expected"]
        if is_match:
            correct += 1
        results_log.append({
            "id": i + 1,
            "text": item["text"],
            "expected": item["expected"],
            "predicted": pred,
            "success": is_match
        })

    accuracy = (correct / total) * 100

    # Proje kök dizinindeki docs/ klasörüne raporu kaydetme
    os.makedirs("../../docs", exist_ok=True)
    report_path = "../../docs/text_matching_evaluation_report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Text Matching Evaluation Report\n\n")
        f.write(f"- **Toplam Test Edilen İlan:** {total}\n")
        f.write(f"- **Doğru Eşleşme Oranı:** %{accuracy:.2f}\n\n")
        f.write("## Detaylı Sonuçlar\n\n")
        f.write("| ID | İlan Metni | Durum |\n|---|---|---|\n")
        for res in results_log:
            status = "✅ Başarılı" if res["success"] else "❌ Hatalı"
            f.write(f"| {res['id']} | {res['text']} | {status} |\n")

    print(f"Değerlendirme tamamlandı. Başarı oranı: %{accuracy:.2f}. Rapor docs/ altına kaydedildi.")

if __name__ == "__main__":
    run_evaluation()