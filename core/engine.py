from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import numpy as np

from utils.image_utils import load_image
from core.preprocessing import ImagePreprocessor, PreprocessingConfig
from core.alignment import ImageAligner, AlignmentConfig
from core.comparison import ImageComparator, ComparisonConfig
from core.change_detection import ChangeDetector, ChangeDetectionConfig
from core.visualization import Visualizer, VisualizationConfig


@dataclass
class ImageDiffResult:
    similarity_score: float
    difference_score: float
    mae: float
    mse: float
    ssim_score: float
    changed_pixel_percentage: float
    detected_regions: List[Dict[str, Any]]
    difference_map: np.ndarray
    heatmap: np.ndarray
    change_mask: np.ndarray
    overlay_image: np.ndarray
    alignment_status: Dict[str, Any]
    raw_ssim: float = 1.0
    psnr: float = 100.0
    side_by_side: Optional[np.ndarray] = None
    contour_image: Optional[np.ndarray] = None
    aligned_target: Optional[np.ndarray] = None
    reference_image: Optional[np.ndarray] = None

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(f"ImageDiffResult has no field named '{item}'")

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_dict(self, include_images: bool = False) -> Dict[str, Any]:
        d = {
            "similarity_score": round(self.similarity_score, 4),
            "difference_score": round(self.difference_score, 4),
            "mae": round(self.mae, 4),
            "mse": round(self.mse, 4),
            "ssim_score": round(self.ssim_score, 4),
            "raw_ssim": round(self.raw_ssim, 4),
            "psnr": round(self.psnr, 2),
            "changed_pixel_percentage": round(self.changed_pixel_percentage, 3),
            "detected_regions_count": len(self.detected_regions),
            "detected_regions": self.detected_regions,
            "alignment_status": self.alignment_status,
        }
        if include_images:
            d["difference_map_shape"] = self.difference_map.shape
            d["heatmap_shape"] = self.heatmap.shape
            d["change_mask_shape"] = self.change_mask.shape
            d["overlay_image_shape"] = self.overlay_image.shape
        return d


class ImageDiffEngine:
    def __init__(
        self,
        preprocessing_config: Optional[PreprocessingConfig] = None,
        alignment_config: Optional[AlignmentConfig] = None,
        comparison_config: Optional[ComparisonConfig] = None,
        change_detection_config: Optional[ChangeDetectionConfig] = None,
        visualization_config: Optional[VisualizationConfig] = None,
    ):
        self.preprocessor = ImagePreprocessor(preprocessing_config)
        self.aligner = ImageAligner(alignment_config)
        self.comparator = ImageComparator(comparison_config)
        self.detector = ChangeDetector(change_detection_config)
        self.visualizer = Visualizer(visualization_config)

    def process(
        self,
        source_a: Union[str, bytes, np.ndarray],
        source_b: Union[str, bytes, np.ndarray],
        align: bool = False,
        threshold_value: int = 30,
        threshold_method: str = "fixed",
        min_contour_area: int = 50,
        blur_kernel_size: int = 5,
        colormap: str = "turbo",
        overlay_alpha: float = 0.45,
    ) -> ImageDiffResult:
        img_a = load_image(source_a, target_mode="RGB")
        img_b = load_image(source_b, target_mode="RGB")

        rgb_a, rgb_b, gray_a, gray_b, prep_meta = self.preprocessor.preprocess_pair(img_a, img_b)

        align_cfg = AlignmentConfig(enabled=align, method="ORB", transform_model="homography")
        align_res = self.aligner.align(rgb_a, rgb_b, align_cfg)
        aligned_b = align_res.aligned_image
        alignment_status = align_res.to_dict()

        comp_metrics, pixel_diff_map, ssim_diff_map = self.comparator.compare(rgb_a, aligned_b)

        cd_cfg = ChangeDetectionConfig(
            blur_kernel_size=blur_kernel_size,
            threshold_method=threshold_method,
            threshold_value=threshold_value,
            min_contour_area=min_contour_area,
            apply_opening=True,
            apply_closing=True
        )
        cd_res = self.detector.detect_changes(rgb_a, aligned_b, cd_cfg)

        self.visualizer.config.colormap = colormap
        self.visualizer.config.overlay_alpha = overlay_alpha

        heatmap = self.visualizer.generate_heatmap(cd_res.abs_diff, colormap=colormap)
        overlay_image = self.visualizer.generate_red_overlay(aligned_b, cd_res.change_mask, alpha=overlay_alpha)
        contour_img = self.visualizer.generate_contour_visualization(aligned_b, cd_res.detected_regions)
        sbs_img = self.visualizer.generate_side_by_side(rgb_a, contour_img)

        return ImageDiffResult(
            similarity_score=comp_metrics.similarity_score,
            difference_score=comp_metrics.difference_score,
            mae=comp_metrics.mae,
            mse=comp_metrics.mse,
            ssim_score=comp_metrics.ssim_score,
            changed_pixel_percentage=cd_res.changed_pixel_percentage,
            detected_regions=cd_res.get_regions_as_dicts(),
            difference_map=cd_res.abs_diff,
            heatmap=heatmap,
            change_mask=cd_res.change_mask,
            overlay_image=overlay_image,
            alignment_status=alignment_status,
            raw_ssim=comp_metrics.raw_ssim,
            psnr=comp_metrics.psnr,
            side_by_side=sbs_img,
            contour_image=contour_img,
            aligned_target=aligned_b,
            reference_image=rgb_a
        )


def compare(
    image_a: Union[str, bytes, np.ndarray],
    image_b: Union[str, bytes, np.ndarray],
    align: bool = False,
    threshold_value: int = 30,
    min_contour_area: int = 50,
    **kwargs
) -> ImageDiffResult:
    engine = ImageDiffEngine()
    return engine.process(
        image_a,
        image_b,
        align=align,
        threshold_value=threshold_value,
        min_contour_area=min_contour_area,
        **kwargs
    )
