from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def signature(
    image: Image.Image,
    size: tuple[int, int] = (160, 90),
    blur: float = 2.0,
) -> np.ndarray:
    """Cheap perceptual fingerprint.

    A slight blur removes a moving mouse cursor and video compression noise;
    downscaling to ~160x90 keeps text changes visible while dropping detail.
    """
    gray = image.convert("L")
    if blur:
        gray = gray.filter(ImageFilter.GaussianBlur(blur))
    gray = gray.resize(size, Image.BILINEAR)
    return np.asarray(gray, dtype=np.float32)


def difference(a: np.ndarray | None, b: np.ndarray | None, delta: float = 25.0) -> float:
    """Fraction of fingerprint pixels that changed by more than `delta`.

    Range 0..1. Far more discriminative than a mean: a new slide or new words
    on a board move a noticeable fraction of pixels, while noise barely moves any.
    """
    if a is None or b is None:
        return 1.0
    if a.shape != b.shape:
        return 1.0
    return float(np.mean(np.abs(a - b) > delta))
