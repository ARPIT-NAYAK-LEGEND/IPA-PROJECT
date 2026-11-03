import io
import numpy as np
import pytest
from PIL import Image

from core.preprocessing import ImagePreprocessor, PreprocessingConfig, preprocess_images
from utils.image_utils import (
    load_image,
    to_rgb,
    to_gray,
    resize_image,
    normalize_image,
    denormalize_image,
    ensure_same_dimensions,
)


def test_load_image_diverse_formats():
    arr = np.full((50, 60, 3), 120, dtype=np.uint8)
    loaded_arr = load_image(arr, target_mode="RGB")
    assert loaded_arr.shape == (50, 60, 3)

    pil_rgb = Image.new("RGB", (80, 70), (200, 100, 50))
    loaded_pil = load_image(pil_rgb, target_mode="RGB")
    assert loaded_pil.shape == (70, 80, 3)

    pil_rgba = Image.new("RGBA", (40, 40), (255, 0, 0, 128))
    loaded_rgba = load_image(pil_rgba, target_mode="RGB")
    assert loaded_rgba.shape == (40, 40, 3)
    assert loaded_rgba.dtype == np.uint8

    buf = io.BytesIO()
    pil_rgb.save(buf, format="PNG")
    loaded_bytes = load_image(buf.getvalue(), target_mode="RGB")
    assert loaded_bytes.shape == (70, 80, 3)

    loaded_gray = load_image(arr, target_mode="GRAY")
    assert loaded_gray.shape == (50, 60)
    assert len(loaded_gray.shape) == 2


def test_rgb_and_grayscale_conversions():
    gray = np.full((30, 30), 100, dtype=np.uint8)
    rgb = to_rgb(gray)
    assert rgb.shape == (30, 30, 3)
    assert np.all(rgb[:, :, 0] == 100)
    assert np.all(rgb[:, :, 1] == 100)
    assert np.all(rgb[:, :, 2] == 100)

    back_to_gray = to_gray(rgb)
    assert back_to_gray.shape == (30, 30)
    assert back_to_gray[0, 0] == 100


def test_rgba_conversion():
    rgba = np.zeros((20, 20, 4), dtype=np.uint8)
    rgba[:, :, 0] = 255
    rgba[:, :, 3] = 128

    rgb = to_rgb(rgba)
    assert rgb.shape == (20, 20, 3)
    assert rgb[0, 0, 0] == 255
    assert 120 <= rgb[0, 0, 1] <= 135


def test_resize_with_and_without_aspect_ratio():
    img = np.zeros((100, 200, 3), dtype=np.uint8)

    resized_pad = resize_image(img, target_size=(100, 100), preserve_aspect_ratio=True)
    assert resized_pad.shape == (100, 100, 3)

    resized_direct = resize_image(img, target_size=(100, 100), preserve_aspect_ratio=False)
    assert resized_direct.shape == (100, 100, 3)


def test_handle_images_with_different_dimensions():
    img_a = np.zeros((150, 250, 3), dtype=np.uint8)
    img_b = np.zeros((300, 400, 3), dtype=np.uint8)

    preprocessor = ImagePreprocessor()
    rgb_a, rgb_b, gray_a, gray_b, meta = preprocessor.preprocess_pair(img_a, img_b)

    assert rgb_a.shape == rgb_b.shape
    assert gray_a.shape == gray_b.shape
    assert meta["geometric_adjustment"]["adjusted"] is True


def test_normalize_and_denormalize():
    img = np.array([[0, 128], [255, 64]], dtype=np.uint8)

    norm = normalize_image(img, min_val=0.0, max_val=1.0)
    assert norm.dtype == np.float32
    assert norm[0, 0] == 0.0
    assert np.isclose(norm[1, 0], 1.0)

    denorm = denormalize_image(norm, current_range=(0.0, 1.0))
    assert denorm.dtype == np.uint8
    assert denorm[0, 0] == 0
    assert denorm[1, 0] == 255
