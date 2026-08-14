"""scripts/train.py için torch cihaz seçimi.

Servis tarafı (embedder.py, app/matcher.py vb.) her zaman CPU'da çalışır — model
çıkarımı zaten CPU'ya bağlı senkron bir iş olacak şekilde tasarlandı (bkz.
requirements.txt'teki pika notu). Eğitim ise saatler sürebildiği için burada
CUDA/MPS varsa onu tercih ediyoruz; servis kodunun bundan haberi yok ve olmamalı.
"""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Kullanılabilir en iyi cihazı döndürür: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
