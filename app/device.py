"""Cihaz seçimi: Apple Silicon üzerinde MPS, aksi halde CPU. CUDA'ya bağımlılık yoktur."""

import torch


def get_device() -> torch.device:
    """Kullanılabilir en iyi cihazı döndürür (MPS > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
