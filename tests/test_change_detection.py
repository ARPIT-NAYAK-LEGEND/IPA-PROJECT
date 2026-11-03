import numpy as np
import pytest

from core.change_detection import ChangeDetector, ChangeDetectionConfig


def test_change_detection_complete_pipeline():
    img_a = np.zeros((200, 200, 3), dtype=np.uint8)
    img_b = img_a.copy()
    img_b[20:50, 20:60] = 220
    img_b[100:150, 100:150] = 200

    detector = ChangeDetector()
    cfg = ChangeDetectionConfig(
        blur_kernel_size=5,
        threshold_method="fixed",
        threshold_value=50,
        morph_kernel_size=3,
        min_contour_area=100
    )

    res = detector.detect_changes(img_a, img_b, cfg)

    assert res.abs_diff.shape == (200, 200)
    assert np.max(res.abs_diff) > 100
    assert res.change_mask.shape == (200, 200)
    assert np.array_equal(np.unique(res.change_mask), np.array([0, 255]))
    assert res.total_regions_count == 2
    assert len(res.detected_regions) == 2
    assert res.detected_regions[0].area >= res.detected_regions[1].area
    assert len(res.detected_regions[0].bbox) == 4
    assert res.changed_pixel_percentage > 5.0


def test_change_detection_configurable_threshold():
    img_a = np.zeros((100, 100, 3), dtype=np.uint8)
    img_b = img_a.copy()
    img_b[30:70, 30:70] = 60

    detector = ChangeDetector()

    cfg_low = ChangeDetectionConfig(threshold_method="fixed", threshold_value=40)
    res_low = detector.detect_changes(img_a, img_b, cfg_low)
    assert res_low.total_regions_count == 1

    cfg_high = ChangeDetectionConfig(threshold_method="fixed", threshold_value=80)
    res_high = detector.detect_changes(img_a, img_b, cfg_high)
    assert res_high.total_regions_count == 0


def test_change_detection_min_contour_area_filter():
    img_a = np.zeros((100, 100, 3), dtype=np.uint8)
    img_b = img_a.copy()
    img_b[10:13, 10:13] = 255
    img_b[40:60, 40:60] = 255

    detector = ChangeDetector()
    cfg = ChangeDetectionConfig(
        threshold_method="fixed",
        threshold_value=50,
        apply_opening=False,
        min_contour_area=50
    )
    res = detector.detect_changes(img_a, img_b, cfg)

    assert res.total_regions_count == 1
    assert res.detected_regions[0].area > 100
