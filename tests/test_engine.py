import numpy as np
import pytest

from core.engine import ImageDiffEngine, ImageDiffResult, compare


def test_imagediff_engine_result_structure():
    h, w = 150, 180
    img_a = np.full((h, w, 3), 120, dtype=np.uint8)
    img_b = img_a.copy()
    img_b[40:80, 50:100] = (255, 255, 255)

    engine = ImageDiffEngine()
    result: ImageDiffResult = engine.process(
        img_a,
        img_b,
        align=False,
        threshold_value=30,
        min_contour_area=50,
    )

    assert isinstance(result.similarity_score, float)
    assert 0.0 <= result.similarity_score <= 1.0

    assert isinstance(result.difference_score, float)
    assert 0.0 <= result.difference_score <= 1.0

    assert isinstance(result.mae, float)
    assert result.mae > 0.0

    assert isinstance(result.mse, float)
    assert result.mse > 0.0

    assert isinstance(result.ssim_score, float)
    assert 0.0 <= result.ssim_score <= 1.0

    assert isinstance(result.changed_pixel_percentage, float)
    assert 0.0 <= result.changed_pixel_percentage <= 100.0

    assert isinstance(result.detected_regions, list)
    assert len(result.detected_regions) >= 1
    region = result.detected_regions[0]
    assert "id" in region
    assert "bbox" in region
    assert "area" in region
    assert "centroid" in region

    assert isinstance(result.difference_map, np.ndarray)
    assert result.difference_map.shape == (h, w)
    assert result.difference_map.dtype == np.uint8

    assert isinstance(result.heatmap, np.ndarray)
    assert result.heatmap.shape == (h, w, 3)
    assert result.heatmap.dtype == np.uint8

    assert isinstance(result.change_mask, np.ndarray)
    assert result.change_mask.shape == (h, w)
    assert result.change_mask.dtype == np.uint8

    assert isinstance(result.overlay_image, np.ndarray)
    assert result.overlay_image.shape == (h, w, 3)
    assert result.overlay_image.dtype == np.uint8

    assert isinstance(result.alignment_status, dict)
    assert "aligned" in result.alignment_status
    assert "status" in result.alignment_status
    assert "message" in result.alignment_status

    assert result["similarity_score"] == result.similarity_score
    assert result["mae"] == result.mae
    assert result["difference_map"].shape == result.difference_map.shape

    d = result.to_dict()
    assert "similarity_score" in d
    assert "mae" in d
    assert "mse" in d
    assert "ssim_score" in d
    assert "changed_pixel_percentage" in d
