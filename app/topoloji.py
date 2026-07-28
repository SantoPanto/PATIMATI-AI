# app/topoloji.py
"""Kuyruk topolojisi ve ayarları — sözleşme §2'nin koddaki karşılığı.

Neden `kuyruk.py`den AYRI bir dosya: `app.kuyruk` içe aktarıldığında model de
yükleniyor (~25 saniye, yüzlerce MB). Oysa kuyruğa mesaj YAZAN taraflar
(uçtan uca kanıt betiği, ileride bir bakım aracı) modele hiç ihtiyaç duymaz.
Sabitleri buraya alınca o araçlar anında açılıyor, ama tek kaynak korunuyor:
kuyruk adları hâlâ tek bir yerde tanımlı, kopyalanmıyor.

Kuyruk adlarının kopyalanması bu tür entegrasyonlarda klasik bir hata kaynağı:
bir tarafta `ai.analysis.request`, diğerinde `ai.analysis.requests` yazılır ve
sistem hata vermeden hiç çalışmaz.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# --- Sözleşme §2. Bu adlar Java tarafıyla BİREBİR aynı olmalı. ---
EXCHANGE = "patimati.ai"
DLX = "patimati.ai.dlx"
ISTEK_KUYRUGU = "ai.analysis.request"
SONUC_KUYRUGU = "ai.analysis.result"
DLQ = "ai.analysis.request.dlq"
ISTEK_ANAHTARI = "analysis.request"
SONUC_ANAHTARI = "analysis.result"

SEMA_SURUMU = 1

# Sondaki %2F, varsayılan "/" sanal sunucusunun URL kodlaması. Çıplak "/"
# yazılırsa pika bunu BOŞ vhost sanar ve broker bağlantıyı reddeder.
AMQP_URL = os.getenv("AMQP_URL", "amqp://guest:guest@localhost:5672/%2F")

# Aynı anda kaç mesaj işlensin. 1 olmasının sebebi: iş CPU'ya bağlı ve tek
# süreçte tek tek yapılıyor. Daha büyük bir değer mesajları broker'dan alıp
# bizim belleğimizde kuyruğa dizmekten başka bir şey yapmaz — üstelik bu süreç
# çökerse hepsi yeniden teslim edilir. Ölçeklemek gerekirse yol prefetch'i
# büyütmek değil, ikinci bir tüketici süreci açmaktır.
PREFETCH = int(os.getenv("AMQP_PREFETCH", "1"))

# Kalp atışı aralığı (saniye). VARSAYILAN 60 BU SERVİS İÇİN TEHLİKELİ: pika'nın
# senkron istemcisi, biz mesajı işlerken kalp atışı GÖNDEREMEZ (aynı iş
# parçacığında bekler). Analiz 60 saniyeyi aşarsa broker bağlantıyı ölü sayıp
# düşürür, mesaj yeniden teslim edilir ve servis sonsuz döngüye girer — bulunması
# çok zor bir hatadır. En kötü durumumuz 5 fotoğraf x ~3 sn, yani ~20 saniye;
# 600 saniye rahat pay bırakıyor. İşleme süresi bunu da aşacaksa doğru çözüm
# burayı büyütmek değil, işi ayrı bir iş parçacığına alıp
# `add_callback_threadsafe` ile onaylamaktır.
KALP_ATISI = int(os.getenv("AMQP_HEARTBEAT", "600"))


def topolojiyi_kur(kanal) -> None:
    """Exchange, kuyruk ve bağlarını ilan eder (sözleşme §2).

    İlan etmek idempotenttir: nesne varsa bir şey olmaz. AMA argümanları FARKLI
    ilan edilirse broker PRECONDITION_FAILED verir ve kanalı kapatır. Java
    tarafı (Spring AMQP) da aynı kuyrukları ilan edeceği için argümanlar iki
    tarafta birebir aynı olmalı — özellikle `x-dead-letter-exchange`.
    Entegrasyonda en sık düşülen tuzaklardan biridir.
    """
    kanal.exchange_declare(EXCHANGE, exchange_type="direct", durable=True)
    kanal.exchange_declare(DLX, exchange_type="direct", durable=True)

    # Ayrıştırılamayan mesajların düştüğü kuyruk.
    # DİKKAT: RabbitMQ bir mesajı ölü mektuba düşürürken ORİJİNAL yönlendirme
    # anahtarını korur (x-dead-letter-routing-key verilmedikçe). Yani DLQ,
    # DLX'e kuyruk adıyla değil `analysis.request` anahtarıyla bağlanmalı.
    # Yanlış bağlanırsa mesajlar hata vermeden yok olur.
    kanal.queue_declare(DLQ, durable=True)
    kanal.queue_bind(DLQ, DLX, routing_key=ISTEK_ANAHTARI)

    kanal.queue_declare(ISTEK_KUYRUGU, durable=True,
                        arguments={"x-dead-letter-exchange": DLX})
    kanal.queue_bind(ISTEK_KUYRUGU, EXCHANGE, routing_key=ISTEK_ANAHTARI)

    kanal.queue_declare(SONUC_KUYRUGU, durable=True)
    kanal.queue_bind(SONUC_KUYRUGU, EXCHANGE, routing_key=SONUC_ANAHTARI)
