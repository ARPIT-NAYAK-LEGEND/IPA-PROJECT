import numpy as np
import pytest

from core.comparison import ImageComparator, ComparisonConfig


def test_pixel_difference_exact_math():
    img_a = np.full((10, 10), 100, dtype=np.uint8)
    img_b = np.full((10, 10), 150, dtype=np.uint8)

    comparator = ImageComparator()
    diff_u8, mae, mse, rmse, pix_pct, psnr = comparator.calculate_pixel_difference(
        img_a, img_b, threshold=25
    )

    assert np.all(diff_u8 == 50)
    assert mae == 50.0
    assert mse == 2500.0
    assert rmse == 50.0
    assert pix_pct == 100.0


def test_ssim_normalized_bounds():
    comparator = ImageComparator()

    img_same = np.full((50, 50, 3), 180, dtype=np.uint8)
    raw_ssim, norm_ssim, ssim_diff = comparator.calculate_ssim(img_same, img_same)

    assert 0.0 <= norm_ssim <= 1.0
    assert norm_ssim >= 0.99
    assert np.all(ssim_diff == 0)

    img_white = np.full((50, 50, 3), 255, dtype=np.uint8)
    img_black = np.zeros((50, 50, 3), dtype=np.uint8)
    raw_diff, norm_diff, ssim_diff_map = comparator.calculate_ssim(img_white, img_black)

    assert 0.0 <= norm_diff <= 1.0
    assert norm_diff < 0.1


def test_distinguish_metrics():
    comparator = ImageComparator()

    img_a = np.zeros((100, 100, 3), dtype=np.uint8)
    img_b = img_a.copy()
    img_b[:10, :] = 200

    cfg = ComparisonConfig(pixel_diff_threshold=25)
    metrics, diff_u8, ssim_diff = comparator.compare(img_a, img_b, cfg)

    assert np.isclose(metrics.pixel_difference_percentage, 10.0, atol=0.1)
    assert 0.90 <= metrics.similarity_score <= 0.95
    assert 0.0 <= metrics.ssim_score <= 1.0
    assert metrics.pixel_difference_percentage != metrics.similarity_score
    assert metrics.similarity_score != metrics.ssim_score
