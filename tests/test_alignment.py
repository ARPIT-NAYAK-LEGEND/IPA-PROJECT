import cv2
import numpy as np
import pytest

from core.alignment import ImageAligner, AlignmentConfig


@pytest.fixture
def rich_feature_pair():
    np.random.seed(101)
    base = np.zeros((300, 300, 3), dtype=np.uint8)

    for i in range(12):
        cv2.circle(base, (40 + i * 20, 50 + (i % 4) * 50), 10, (180 - i * 8, 40 + i * 15, 200), -1)
        cv2.rectangle(base, (50 + i * 15, 180), (80 + i * 15, 240), (60 + i * 10, 210, 100), 2)
    cv2.putText(base, "ORB CALIBRATION TEST", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    h, w = base.shape[:2]
    M = np.float32([[1, 0, 8], [0, 1, 4]])
    shifted = cv2.warpAffine(base, M, (w, h))
    return base, shifted


def test_orb_alignment_success(rich_feature_pair):
    ref, target = rich_feature_pair
    aligner = ImageAligner()
    cfg = AlignmentConfig(enabled=True, method="ORB", transform_model="homography")
    res = aligner.align(ref, target, cfg)

    assert res.success is True
    assert res.status == "SUCCESS"
    assert res.homography_matrix is not None
    assert res.homography_matrix.shape == (3, 3)
    assert res.inliers >= 4
    assert res.aligned_image.shape == ref.shape


def test_orb_alignment_skipped():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    aligner = ImageAligner()
    cfg = AlignmentConfig(enabled=False)
    res = aligner.align(img, img, cfg)

    assert res.success is True
    assert res.status == "SKIPPED"


def test_orb_alignment_failure_clear_status():
    blank_a = np.zeros((150, 150, 3), dtype=np.uint8)
    blank_b = np.full((150, 150, 3), 128, dtype=np.uint8)

    aligner = ImageAligner()
    cfg = AlignmentConfig(enabled=True, method="ORB")
    res = aligner.align(blank_a, blank_b, cfg)

    assert res.success is False
    assert res.status == "FAILED"
    assert res.homography_matrix is None
    assert np.array_equal(res.aligned_image, blank_b)

    status_dict = res.to_dict()
    assert status_dict["aligned"] is False
    assert status_dict["status"] == "FAILED"
    assert status_dict["homography_matrix"] is None
