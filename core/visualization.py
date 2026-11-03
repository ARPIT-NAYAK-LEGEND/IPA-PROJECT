from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2
import numpy as np

from utils.image_utils import to_rgb
from core.change_detection import ChangeRegion


@dataclass
class VisualizationConfig:
    colormap: str = "turbo"
    colormap_name: str = "turbo"
    overlay_alpha: float = 0.45
    heatmap_alpha: float = 0.55
    overlay_color: Tuple[int, int, int] = (255, 0, 0)
    box_color: Tuple[int, int, int] = (255, 40, 40)
    box_thickness: int = 2
    draw_contours: bool = True
    contour_fill_alpha: float = 0.25
    contour_color: Tuple[int, int, int] = (0, 255, 128)
    contour_thickness: int = 2
    show_labels: bool = True
    font_scale: float = 0.55
    show_centroids: bool = True


class Visualizer:
    COLORMAP_MAPPING = {
        "turbo": cv2.COLORMAP_TURBO,
        "jet": cv2.COLORMAP_JET,
        "inferno": cv2.COLORMAP_INFERNO,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "magma": cv2.COLORMAP_MAGMA,
        "plasma": cv2.COLORMAP_PLASMA,
        "bone": cv2.COLORMAP_BONE,
        "ocean": cv2.COLORMAP_OCEAN,
    }

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()

    def set_config(self, config: VisualizationConfig) -> None:
        self.config = config

    def generate_heatmap(
        self,
        diff_map: np.ndarray,
        colormap: Optional[str] = None
    ) -> np.ndarray:
        cmap_name = (colormap or self.config.colormap or self.config.colormap_name).lower()
        cmap_id = self.COLORMAP_MAPPING.get(cmap_name, cv2.COLORMAP_TURBO)

        diff_u8 = np.clip(diff_map, 0, 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(diff_u8, cmap_id)
        return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    def create_heatmap(
        self,
        diff_map: np.ndarray,
        background_img: Optional[np.ndarray] = None,
        config: Optional[VisualizationConfig] = None
    ) -> np.ndarray:
        cfg = config or self.config
        hm = self.generate_heatmap(diff_map, colormap=cfg.colormap or cfg.colormap_name)
        if background_img is None:
            return hm
        bg = to_rgb(background_img)
        if bg.shape[:2] != hm.shape[:2]:
            hm = cv2.resize(hm, (bg.shape[1], bg.shape[0]))
        a = float(np.clip(cfg.heatmap_alpha or cfg.overlay_alpha, 0.0, 1.0))
        return cv2.addWeighted(bg, 1.0 - a, hm, a, 0)

    def generate_binary_mask_image(self, change_mask: np.ndarray) -> np.ndarray:
        mask_u8 = np.clip(change_mask, 0, 255).astype(np.uint8)
        return cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2RGB)

    def generate_red_overlay(
        self,
        base_image: np.ndarray,
        change_mask: np.ndarray,
        alpha: Optional[float] = None,
        red_color: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        rgb_base = to_rgb(base_image).copy()
        h, w = rgb_base.shape[:2]

        mask = change_mask if change_mask.shape[:2] == (h, w) else cv2.resize(change_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        a = float(np.clip(alpha if alpha is not None else self.config.overlay_alpha, 0.0, 1.0))
        color = red_color or self.config.overlay_color

        changed_indices = mask > 0
        if np.any(changed_indices):
            overlay = rgb_base.copy()
            overlay[changed_indices] = color
            rgb_base[changed_indices] = (
                a * np.array(color, dtype=np.float32) +
                (1.0 - a) * rgb_base[changed_indices].astype(np.float32)
            ).astype(np.uint8)

        return rgb_base

    def generate_side_by_side(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        title_a: str = "Image A (Reference)",
        title_b: str = "Image B (Target)"
    ) -> np.ndarray:
        rgb_a = to_rgb(img_a)
        rgb_b = to_rgb(img_b)

        h1, w1 = rgb_a.shape[:2]
        h2, w2 = rgb_b.shape[:2]

        target_h = max(h1, h2)
        if h1 != target_h:
            rgb_a = cv2.resize(rgb_a, (int(w1 * target_h / h1), target_h), interpolation=cv2.INTER_AREA)
        if h2 != target_h:
            rgb_b = cv2.resize(rgb_b, (int(w2 * target_h / h2), target_h), interpolation=cv2.INTER_AREA)

        divider_w = 4
        divider = np.full((target_h, divider_w, 3), 60, dtype=np.uint8)
        banner_h = 32
        total_w = rgb_a.shape[1] + divider_w + rgb_b.shape[1]
        composite = np.zeros((target_h + banner_h, total_w, 3), dtype=np.uint8)

        composite[:banner_h, :] = (20, 24, 30)
        composite[banner_h:, :rgb_a.shape[1]] = rgb_a
        composite[banner_h:, rgb_a.shape[1]:rgb_a.shape[1] + divider_w] = divider
        composite[banner_h:, rgb_a.shape[1] + divider_w:] = rgb_b

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(composite, title_a, (12, 22), font, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(
            composite, title_b, (rgb_a.shape[1] + divider_w + 12, 22),
            font, 0.6, (220, 220, 220), 1, cv2.LINE_AA
        )

        return composite

    def create_side_by_side(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        label_a: str = "Image A (Reference)",
        label_b: str = "Image B (Target)"
    ) -> np.ndarray:
        return self.generate_side_by_side(img_a, img_b, label_a, label_b)

    def generate_contour_visualization(
        self,
        base_image: np.ndarray,
        regions: List[ChangeRegion],
        draw_boxes: bool = True,
        draw_contours: bool = True
    ) -> np.ndarray:
        canvas = to_rgb(base_image).copy()
        h, w = canvas.shape[:2]
        cfg = self.config

        if cfg.contour_fill_alpha > 0.0 and regions:
            overlay = canvas.copy()
            for r in regions:
                if r.contour is not None:
                    cv2.drawContours(overlay, [r.contour], -1, cfg.box_color, -1)
            alpha = float(np.clip(cfg.contour_fill_alpha, 0.0, 1.0))
            canvas = cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0)

        for r in regions:
            if draw_contours and cfg.draw_contours and r.contour is not None:
                cv2.drawContours(canvas, [r.contour], -1, cfg.contour_color, cfg.contour_thickness)

            if draw_boxes:
                x, y, rw, rh = r.bbox
                cv2.rectangle(canvas, (x, y), (x + rw, y + rh), cfg.box_color, cfg.box_thickness)

                if cfg.show_centroids:
                    cx, cy = r.centroid
                    cv2.circle(canvas, (cx, cy), 3, (255, 255, 0), -1)

                if cfg.show_labels:
                    label = f"#{r.id} ({r.area_percentage:.1f}%)"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (tw, th), _ = cv2.getTextSize(label, font, cfg.font_scale, 1)

                    bg_y1 = max(0, y - th - 6)
                    bg_y2 = y
                    bg_x1 = x
                    bg_x2 = min(w, x + tw + 6)

                    cv2.rectangle(canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), (15, 18, 24), -1)
                    cv2.rectangle(canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), cfg.box_color, 1)
                    cv2.putText(
                        canvas, label, (x + 3, max(th + 2, y - 4)),
                        font, cfg.font_scale, (255, 255, 255), 1, cv2.LINE_AA
                    )

        return canvas

    def draw_detections(
        self,
        base_img: np.ndarray,
        regions: List[ChangeRegion],
        config: Optional[VisualizationConfig] = None
    ) -> np.ndarray:
        return self.generate_contour_visualization(base_img, regions)

    def create_split_view(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        split_pct: float = 50.0
    ) -> np.ndarray:
        rgb_a = to_rgb(img_a)
        rgb_b = to_rgb(img_b)
        h, w = rgb_a.shape[:2]

        if rgb_b.shape[:2] != (h, w):
            rgb_b = cv2.resize(rgb_b, (w, h))

        pct = np.clip(split_pct, 0.0, 100.0) / 100.0
        split_x = int(round(w * pct))

        composite = np.zeros_like(rgb_a)
        if split_x > 0:
            composite[:, :split_x] = rgb_a[:, :split_x]
        if split_x < w:
            composite[:, split_x:] = rgb_b[:, split_x:]

        line_w = 2
        x1 = max(0, split_x - line_w)
        x2 = min(w, split_x + line_w)
        composite[:, x1:x2] = (0, 220, 255)

        return composite

    def create_quad_panel(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        heatmap: np.ndarray,
        annotated: np.ndarray
    ) -> np.ndarray:
        a = to_rgb(img_a)
        b = to_rgb(img_b)
        h = to_rgb(heatmap)
        d = to_rgb(annotated)

        target_h, target_w = a.shape[:2]
        b = cv2.resize(b, (target_w, target_h))
        h = cv2.resize(h, (target_w, target_h))
        d = cv2.resize(d, (target_w, target_h))

        def _add_header(img: np.ndarray, title: str) -> np.ndarray:
            banner = np.full((30, target_w, 3), 20, dtype=np.uint8)
            cv2.putText(banner, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
            return np.vstack([banner, img])

        p1 = _add_header(a, "1. Reference (Image A)")
        p2 = _add_header(b, "2. Target (Image B)")
        p3 = _add_header(h, "3. Difference Heatmap")
        p4 = _add_header(d, "4. Change Annotations & Mask")

        top_row = np.hstack([p1, p2])
        bottom_row = np.hstack([p3, p4])
        return np.vstack([top_row, bottom_row])


def create_visualizations(
    base_image: np.ndarray,
    diff_map: np.ndarray,
    change_mask: np.ndarray,
    regions: List[ChangeRegion],
    ref_image: Optional[np.ndarray] = None,
    config: Optional[VisualizationConfig] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    vis = Visualizer(config)
    heatmap = vis.generate_heatmap(diff_map)
    mask_rgb = vis.generate_binary_mask_image(change_mask)
    red_overlay = vis.generate_red_overlay(base_image, change_mask)
    contour_vis = vis.generate_contour_visualization(base_image, regions)

    sbs = None
    if ref_image is not None:
        sbs = vis.generate_side_by_side(ref_image, contour_vis)

    return heatmap, mask_rgb, red_overlay, contour_vis, sbs
