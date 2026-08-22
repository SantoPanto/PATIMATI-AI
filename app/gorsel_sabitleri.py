# app/gorsel_sabitleri.py
"""Görüntü ön işlemede birden fazla modülün paylaştığı sabitler.

Ayrı bir dosyada tutulma sebebi DÖNGÜSEL İÇE AKTARMAYI ÖNLEMEK: app/embedder.py
ve app/kirpma.py birbirine bağımlı (embedder kırpma yeteneğini çağırır,
kırpma da embedder'ın "çok küçük" eşiğini paylaşır) -- ikisi de bu sabiti
ÜÇÜNCÜ, ikisine de bağımlı olmayan bir modülden alırsa çevrim oluşmaz.
"""

# Bundan küçük görüntüler anlamlı bir vektör üretmez (1x1 bile sessizce
# 512'lik bir vektör döndürüyordu — çöp veriyi veritabanına yazmayalım).
ASGARI_KENAR = 32
