from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np
from skimage.metrics import structural_similarity as ssim

from utils.image_utils import to_rgb, to_gray, validate_image


@dataclass
class ComparisonConfig:
    color_mode: str = "RGB"
    pixel_diff_threshold: int = 25
    ssim_win_size: int = 7
    ssim_k1: float = 0.01
    ssim_k2: float = 0.03


@dataclass
class ComparisonMetrics:
    similarity_score: float
    difference_score: float
    mae: float
    mse: float
    rmse: float
    psnr: float
    raw_ssim: float
    ssim_score: float
    pixel_difference_percentage: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "similarity_score": round(self.similarity_score, 4),
            "difference_score": round(self.difference_score, 4),
            "mae": round(self.mae, 4),
            "mse": round(self.mse, 4),
            "rmse": round(self.rmse, 4),
            "psnr": round(self.psnr, 2),
            "raw_ssim": round(self.raw_ssim, 4),
            "ssim_score": round(self.ssim_score, 4),
            "pixel_difference_percentage": round(self.pixel_difference_percentage, 2),
        }


class ImageComparator:
    def __init__(self, config: Optional[ComparisonConfig] = None):
        self.config = config or ComparisonConfig()

    def set_config(self, config: ComparisonConfig) -> None:
        self.config = config

    def calculate_pixel_difference(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        threshold: int = 25
    ) -> Tuple[np.ndarray, float, float, float, float, float]:
        validate_image(img_a, "Image A")
        validate_image(img_b, "Image B")

        if img_a.shape[:2] != img_b.shape[:2]:
            raise ValueError(f"Shape mismatch: {img_a.shape[:2]} vs {img_b.shape[:2]}")

        a_f = img_a.astype(np.float64)
        b_f = img_b.astype(np.float64)
        diff_raw = np.abs(a_f - b_f)

        diff_2d = np.mean(diff_raw, axis=2) if len(diff_raw.shape) == 3 else diff_raw

        mae = float(np.mean(diff_raw))
        mse = float(np.mean((a_f - b_f) ** 2))
        rmse = float(np.sqrt(mse))
        psnr = 100.0 if mse == 0.0 else float(10.0 * np.log10((255.0 ** 2) / mse))

        total_pixels = diff_2d.size
        changed_pixels = int(np.count_nonzero(diff_2d >= threshold))
        pixel_diff_pct = (changed_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

        diff_uint8 = np.clip(diff_2d, 0, 255).astype(np.uint8)
        return diff_uint8, mae, mse, rmse, pixel_diff_pct, psnr

    def calculate_ssim(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        win_size: int = 7,
        k1: float = 0.01,
        k2: float = 0.03
    ) -> Tuple[float, float, np.ndarray]:
        validate_image(img_a, "Image A")
        validate_image(img_b, "Image B")

        h, w = img_a.shape[:2]
        w_size = win_size if win_size % 2 == 1 else win_size + 1
        w_size = min(w_size, min(h, w))
        if w_size % 2 == 0:
            w_size -= 1
        w_size = max(3, w_size)

        is_color = (len(img_a.shape) == 3 and img_a.shape[2] == 3)

        if is_color:
            raw_score, full_map = ssim(
                img_a,
                img_b,
                win_size=w_size,
                data_range=255,
                K1=k1,
                K2=k2,
                channel_axis=2,
                full=True
            )
            ssim_2d = np.mean(full_map, axis=2)
        else:
            raw_score, full_map = ssim(
                to_gray(img_a),
                to_gray(img_b),
                win_size=w_size,
                data_range=255,
                K1=k1,
                K2=k2,
                full=True
            )
            ssim_2d = full_map

        raw_val = float(raw_score)
        normalized_ssim = max(0.0, min(1.0, raw_val))
        ssim_diff_map = np.clip((1.0 - ssim_2d) * 255.0, 0, 255).astype(np.uint8)

        return raw_val, normalized_ssim, ssim_diff_map

    def compare(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        config: Optional[ComparisonConfig] = None
    ) -> Tuple[ComparisonMetrics, np.ndarray, np.ndarray]:
        cfg = config or self.config

        if cfg.color_mode.upper() == "GRAY":
            work_a = to_gray(img_a)
            work_b = to_gray(img_b)
        else:
            work_a = to_rgb(img_a)
            work_b = to_rgb(img_b)

        diff_u8, mae, mse, rmse, pix_pct, psnr = self.calculate_pixel_difference(
            work_a, work_b, threshold=cfg.pixel_diff_threshold
        )

        raw_ssim, norm_ssim, ssim_diff = self.calculate_ssim(
            work_a, work_b,
            win_size=cfg.ssim_win_size,
            k1=cfg.ssim_k1,
            k2=cfg.ssim_k2
        )

        similarity_score = float(max(0.0, min(1.0, 1.0 - (mae / 255.0))))
        difference_score = float(max(0.0, min(1.0, mae / 255.0)))

        metrics = ComparisonMetrics(
            similarity_score=similarity_score,
            difference_score=difference_score,
            mae=float(mae),
            mse=float(mse),
            rmse=float(rmse),
            psnr=float(psnr),
            raw_ssim=float(raw_ssim),
            ssim_score=float(norm_ssim),
            pixel_difference_percentage=float(pix_pct),
        )

        return metrics, diff_u8, ssim_diff


def compare_images(
    img_a: np.ndarray,
    img_b: np.ndarray,
    config: Optional[ComparisonConfig] = None
) -> Tuple[ComparisonMetrics, np.ndarray, np.ndarray]:
    comparator = ImageComparator(config)
    return comparator.compare(img_a, img_b)
