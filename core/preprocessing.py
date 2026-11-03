from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np

from utils.image_utils import (
    to_rgb,
    to_gray,
    ensure_same_dimensions,
    validate_image,
)


@dataclass
class PreprocessingConfig:
    resize_mode: str = "resize"
    target_reference: str = "first"
    preserve_aspect_ratio: bool = True
    filter_type: str = "gaussian"
    kernel_size: int = 5
    gaussian_sigma: float = 1.0
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0
    enable_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    normalize_contrast: bool = False


class ImagePreprocessor:
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

    def set_config(self, config: PreprocessingConfig) -> None:
        self.config = config

    def apply_filter(self, img: np.ndarray, config: Optional[PreprocessingConfig] = None) -> np.ndarray:
        cfg = config or self.config
        ftype = (cfg.filter_type or "none").lower()

        if ftype == "none" or cfg.kernel_size <= 1:
            return img.copy()

        k = cfg.kernel_size if cfg.kernel_size % 2 == 1 else cfg.kernel_size + 1
        k = max(3, k)

        if ftype == "gaussian":
            return cv2.GaussianBlur(img, (k, k), sigmaX=cfg.gaussian_sigma)
        if ftype == "bilateral":
            return cv2.bilateralFilter(
                img,
                d=k,
                sigmaColor=cfg.bilateral_sigma_color,
                sigmaSpace=cfg.bilateral_sigma_space
            )
        if ftype == "median":
            return cv2.medianBlur(img, k)
        return img.copy()

    def apply_clahe(self, img: np.ndarray, config: Optional[PreprocessingConfig] = None) -> np.ndarray:
        cfg = config or self.config
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=cfg.clahe_tile_grid_size
        )

        if len(img.shape) == 2:
            return clahe.apply(img)
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = clahe.apply(l)
            merged = cv2.merge([l_enhanced, a, b])
            return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return img.copy()

    def apply_contrast_normalization(self, img: np.ndarray) -> np.ndarray:
        img_float = img.astype(np.float32)
        min_val = float(np.percentile(img_float, 1.0))
        max_val = float(np.percentile(img_float, 99.0))

        if max_val <= min_val:
            return img.copy()

        stretched = (img_float - min_val) * (255.0 / (max_val - min_val))
        return np.clip(stretched, 0, 255).astype(np.uint8)

    def preprocess_single(self, img: np.ndarray, config: Optional[PreprocessingConfig] = None) -> np.ndarray:
        cfg = config or self.config
        validate_image(img)
        current = to_rgb(img)

        if cfg.normalize_contrast:
            current = self.apply_contrast_normalization(current)

        if cfg.enable_clahe:
            current = self.apply_clahe(current, cfg)

        if cfg.filter_type != "none":
            current = self.apply_filter(current, cfg)

        return current

    def preprocess_pair(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        config: Optional[PreprocessingConfig] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        cfg = config or self.config

        aligned_a, aligned_b, dim_meta = ensure_same_dimensions(
            img_a, img_b,
            mode=cfg.resize_mode,
            target_reference=cfg.target_reference,
            preserve_aspect_ratio=cfg.preserve_aspect_ratio
        )

        proc_rgb_a = self.preprocess_single(aligned_a, cfg)
        proc_rgb_b = self.preprocess_single(aligned_b, cfg)

        gray_a = to_gray(proc_rgb_a)
        gray_b = to_gray(proc_rgb_b)

        metadata = {
            "geometric_adjustment": dim_meta,
            "filter_applied": cfg.filter_type,
            "kernel_size": cfg.kernel_size,
            "clahe_applied": cfg.enable_clahe,
            "contrast_normalized": cfg.normalize_contrast,
            "final_resolution": (proc_rgb_a.shape[1], proc_rgb_a.shape[0]),
        }

        return proc_rgb_a, proc_rgb_b, gray_a, gray_b, metadata


def preprocess_images(
    img_a: np.ndarray,
    img_b: np.ndarray,
    config: Optional[PreprocessingConfig] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    preprocessor = ImagePreprocessor(config)
    return preprocessor.preprocess_pair(img_a, img_b)
