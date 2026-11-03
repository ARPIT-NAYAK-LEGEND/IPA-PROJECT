import pytest
import numpy as np

from samples.sample_generator import (
    generate_object_pair,
    generate_document_pair,
    generate_pcb_pair,
    generate_architecture_pair,
)
from core.engine import ImageDiffEngine, ImageDiffResult, compare
from utils.image_utils import to_bytes


@pytest.mark.parametrize(
    "generator",
    [
        generate_object_pair,
        generate_document_pair,
        generate_pcb_pair,
        generate_architecture_pair,
    ],
)
def test_full_engine_across_domains(generator):
    img_a, img_b = generator()

    engine = ImageDiffEngine()
    result: ImageDiffResult = engine.process(
        img_a,
        img_b,
        align=True,
        threshold_value=30,
        min_contour_area=40,
    )

    assert 0.0 <= result.similarity_score <= 1.0
    assert 0.0 <= result.difference_score <= 1.0
    assert result.mae >= 0.0
    assert result.mse >= 0.0
    assert 0.0 <= result.ssim_score <= 1.0
    assert 0.0 <= result.changed_pixel_percentage <= 100.0

    assert result.change_mask.shape == result.difference_map.shape
    assert result.change_mask.shape == result.heatmap.shape[:2]
    assert len(result.detected_regions) >= 1

    assert result.overlay_image.shape == result.heatmap.shape
    assert result.side_by_side is not None

    png_bytes = to_bytes(result.overlay_image, format="PNG")
    assert len(png_bytes) > 500
