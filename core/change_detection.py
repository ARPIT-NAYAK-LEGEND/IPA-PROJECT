from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np

from utils.image_utils import to_gray, validate_image


@dataclass
class ChangeRegion:
    id: int
    bbox: Tuple[int, int, int, int]
    area: int
    area_percentage: float
    centroid: Tuple[int, int]
    aspect_ratio: float
    extent: float
    contour: np.ndarray

    @property
    def area_pixels(self) -> int:
        return self.area

    @property
    def mean_intensity(self) -> float:
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "bbox": self.bbox,
            "x": self.bbox[0],
            "y": self.bbox[1],
            "width": self.bbox[2],
            "height": self.bbox[3],
            "area": self.area,
            "area_percentage": round(self.area_percentage, 3),
            "centroid": self.centroid,
            "aspect_ratio": round(self.aspect_ratio, 2),
            "extent": round(self.extent, 2),
        }


@dataclass
class ChangeDetectionConfig:
    blur_kernel_size: int = 5
    blur_sigma: float = 1.0
    threshold_method: str = "fixed"
    threshold_value: int = 30
    morph_kernel_size: int = 5
    apply_opening: bool = True
    apply_closing: bool = True
    min_contour_area: int = 50
    max_contour_area_pct: float = 95.0


@dataclass
class ChangeDetectionResult:
    abs_diff: np.ndarray
    thresh_mask: np.ndarray
    change_mask: np.ndarray
    detected_regions: List[ChangeRegion]
    total_regions_count: int
    total_changed_pixels: int
    changed_pixel_percentage: float
    largest_region_area: int
    threshold_value_used: float

    @property
    def regions(self) -> List[ChangeRegion]:
        return self.detected_regions

    @property
    def clean_mask(self) -> np.ndarray:
        return self.change_mask

    @property
    def raw_mask(self) -> np.ndarray:
        return self.thresh_mask

    @property
    def changed_area_percentage(self) -> float:
        return self.changed_pixel_percentage

    def get_regions_as_dicts(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.detected_regions]


class ChangeDetector:
    def __init__(self, config: Optional[ChangeDetectionConfig] = None):
        self.config = config or ChangeDetectionConfig()

    def set_config(self, config: ChangeDetectionConfig) -> None:
        self.config = config

    def detect_changes(
        self,
        img_a: np.ndarray,
        img_b: Optional[np.ndarray] = None,
        config: Optional[ChangeDetectionConfig] = None
    ) -> ChangeDetectionResult:
        cfg = config or self.config

        if img_b is not None:
            validate_image(img_a, "Image A")
            validate_image(img_b, "Image B")

            if img_a.shape[:2] != img_b.shape[:2]:
                raise ValueError(f"Shape mismatch: {img_a.shape[:2]} vs {img_b.shape[:2]}")

            gray_a = to_gray(img_a)
            gray_b = to_gray(img_b)

            k = cfg.blur_kernel_size if cfg.blur_kernel_size % 2 == 1 else cfg.blur_kernel_size + 1
            k = max(3, k)
            blur_a = cv2.GaussianBlur(gray_a, (k, k), cfg.blur_sigma)
            blur_b = cv2.GaussianBlur(gray_b, (k, k), cfg.blur_sigma)

            abs_diff = cv2.absdiff(blur_a, blur_b)
        else:
            abs_diff = to_gray(img_a) if len(img_a.shape) == 3 else img_a.copy()

        thresh_method = cfg.threshold_method.lower()
        if thresh_method == "otsu":
            thresh_val, thresh_mask = cv2.threshold(
                abs_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            t = int(np.clip(cfg.threshold_value, 1, 254))
            thresh_val = float(t)
            _, thresh_mask = cv2.threshold(abs_diff, t, 255, cv2.THRESH_BINARY)

        mk = cfg.morph_kernel_size if cfg.morph_kernel_size % 2 == 1 else cfg.morph_kernel_size + 1
        mk = max(3, mk)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (mk, mk))

        clean_mask = thresh_mask.copy()
        if cfg.apply_opening:
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
        if cfg.apply_closing:
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

        h, w = clean_mask.shape[:2]
        total_pixels = h * w

        contours, _ = cv2.findContours(
            clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        max_allowed_area = (cfg.max_contour_area_pct / 100.0) * total_pixels
        min_allowed_area = max(1, cfg.min_contour_area)

        detected_regions: List[ChangeRegion] = []
        total_changed_pixels = int(np.count_nonzero(clean_mask))
        changed_pixel_pct = (total_changed_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

        valid_contours = []
        for cnt in contours:
            area_val = float(cv2.contourArea(cnt))
            if min_allowed_area <= area_val <= max_allowed_area:
                valid_contours.append((cnt, area_val))

        valid_contours.sort(key=lambda x: x[1], reverse=True)

        largest_area = 0
        for idx, (cnt, area_val) in enumerate(valid_contours, start=1):
            x, y, rw, rh = cv2.boundingRect(cnt)
            area_px = int(area_val)
            if area_px > largest_area:
                largest_area = area_px

            area_pct = (area_px / total_pixels) * 100.0

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = x + rw // 2
                cy = y + rh // 2

            aspect_ratio = float(rw) / float(rh) if rh > 0 else 1.0
            rect_area = rw * rh
            extent = float(area_px) / float(rect_area) if rect_area > 0 else 0.0

            region = ChangeRegion(
                id=idx,
                bbox=(x, y, rw, rh),
                area=area_px,
                area_percentage=float(round(area_pct, 4)),
                centroid=(cx, cy),
                aspect_ratio=float(round(aspect_ratio, 2)),
                extent=float(round(extent, 2)),
                contour=cnt
            )
            detected_regions.append(region)

        return ChangeDetectionResult(
            abs_diff=abs_diff,
            thresh_mask=thresh_mask,
            change_mask=clean_mask,
            detected_regions=detected_regions,
            total_regions_count=len(detected_regions),
            total_changed_pixels=total_changed_pixels,
            changed_pixel_percentage=float(round(changed_pixel_pct, 4)),
            largest_region_area=largest_area,
            threshold_value_used=float(thresh_val)
        )


def detect_changes(
    img_a: np.ndarray,
    img_b: Optional[np.ndarray] = None,
    config: Optional[ChangeDetectionConfig] = None
) -> ChangeDetectionResult:
    detector = ChangeDetector(config)
    return detector.detect_changes(img_a, img_b)
