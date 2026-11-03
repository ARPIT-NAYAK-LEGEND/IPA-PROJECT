import numpy as np
import pytest
from PIL import Image

from utils.image_utils import (
    to_rgb,
    to_gray,
    to_bgr,
    to_pil,
    to_bytes,
    validate_image,
    ensure_same_dimensions,
    calculate_aspect_ratio,
    encode_image_to_base64,
)


def test_validate_image_valid():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    validate_image(img)


def test_validate_image_invalid():
    with pytest.raises(ValueError):
        validate_image(np.array([]))
    with pytest.raises(ValueError):
        validate_image(np.zeros((1, 1, 3), dtype=np.uint8))


def test_format_conversions():
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    img[:, :, 0] = 255

    rgb = to_rgb(img)
    assert rgb.shape == (50, 50, 3)
    assert rgb.dtype == np.uint8

    gray = to_gray(img)
    assert gray.shape == (50, 50)
    assert gray.dtype == np.uint8

    bgr = to_bgr(img)
    assert bgr.shape == (50, 50, 3)
    assert bgr[0, 0, 2] == 255

    pil = to_pil(img)
    assert isinstance(pil, Image.Image)
    assert pil.size == (50, 50)

    raw_bytes = to_bytes(img, format="PNG")
    assert isinstance(raw_bytes, bytes)
    assert len(raw_bytes) > 0

    b64 = encode_image_to_base64(img)
    assert b64.startswith("data:image/png;base64,")


def test_ensure_same_dimensions_resize():
    img1 = np.zeros((100, 200, 3), dtype=np.uint8)
    img2 = np.zeros((300, 400, 3), dtype=np.uint8)

    out1, out2, meta = ensure_same_dimensions(img1, img2, mode="resize", target_reference="first")
    assert out1.shape == (100, 200, 3)
    assert out2.shape == (100, 200, 3)
    assert meta["adjusted"] is True


def test_ensure_same_dimensions_pad():
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.zeros((50, 50, 3), dtype=np.uint8)

    out1, out2, meta = ensure_same_dimensions(img1, img2, mode="pad", target_reference="first")
    assert out1.shape == (100, 100, 3)
    assert out2.shape == (100, 100, 3)


def test_calculate_aspect_ratio():
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    assert calculate_aspect_ratio(img) == 2.0
