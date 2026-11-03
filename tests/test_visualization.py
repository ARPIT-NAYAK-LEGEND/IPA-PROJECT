import numpy as np
import pytest

from core.visualization import Visualizer, VisualizationConfig, create_visualizations
from core.change_detection import ChangeRegion


def test_visualizations_generation():
    h, w = 120, 160
    base = np.full((h, w, 3), 100, dtype=np.uint8)
    diff = np.zeros((h, w), dtype=np.uint8)
    diff[30:70, 30:70] = 200

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[30:70, 30:70] = 255

    region = ChangeRegion(
        id=1,
        bbox=(30, 30, 40, 40),
        area=1600,
        area_percentage=8.33,
        centroid=(50, 50),
        aspect_ratio=1.0,
        extent=1.0,
        contour=np.array([[[30, 30]], [[70, 30]], [[70, 70]], [[30, 70]]], dtype=np.int32),
    )

    vis = Visualizer()

    heatmap = vis.generate_heatmap(diff, colormap="turbo")
    assert heatmap.shape == (h, w, 3)
    assert heatmap.dtype == np.uint8

    mask_rgb = vis.generate_binary_mask_image(mask)
    assert mask_rgb.shape == (h, w, 3)
    assert mask_rgb.dtype == np.uint8

    overlay = vis.generate_red_overlay(base, mask, alpha=0.5, red_color=(255, 0, 0))
    assert overlay.shape == (h, w, 3)
    assert np.all(overlay[35:65, 35:65, 0] > 150)

    sbs = vis.generate_side_by_side(base, overlay)
    assert sbs.shape[0] >= h
    assert sbs.shape[1] > w * 2
    assert sbs.shape[2] == 3

    annotated = vis.generate_contour_visualization(base, [region])
    assert annotated.shape == (h, w, 3)
