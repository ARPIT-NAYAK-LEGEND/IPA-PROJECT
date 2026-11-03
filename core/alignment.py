from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import cv2
import numpy as np

from utils.image_utils import to_gray, to_rgb, validate_image


@dataclass
class AlignmentConfig:
    enabled: bool = True
    method: str = "ORB"
    transform_model: str = "homography"
    max_features: int = 2000
    ratio_threshold: float = 0.75
    ransac_threshold: float = 3.0
    min_inliers_required: int = 8
    ecc_max_iterations: int = 100
    ecc_eps: float = 1e-5


@dataclass
class AlignmentResult:
    aligned_image: np.ndarray
    reference_image: np.ndarray
    homography_matrix: Optional[np.ndarray]
    total_keypoints_ref: int
    total_keypoints_target: int
    matches_found: int
    inliers: int
    inlier_ratio: float
    success: bool
    status: str
    method: str
    message: str
    matches_visualization: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aligned": self.success,
            "status": self.status,
            "message": self.message,
            "inliers": self.inliers,
            "matches_found": self.matches_found,
            "inlier_ratio": float(round(self.inlier_ratio, 4)),
            "method": self.method,
            "total_keypoints_ref": self.total_keypoints_ref,
            "total_keypoints_target": self.total_keypoints_target,
            "homography_matrix": (
                self.homography_matrix.tolist() if self.homography_matrix is not None else None
            ),
        }


class ImageAligner:
    def __init__(self, config: Optional[AlignmentConfig] = None):
        self.config = config or AlignmentConfig()

    def set_config(self, config: AlignmentConfig) -> None:
        self.config = config

    def align(
        self,
        ref_img: np.ndarray,
        target_img: np.ndarray,
        config: Optional[AlignmentConfig] = None
    ) -> AlignmentResult:
        cfg = config or self.config
        validate_image(ref_img, "Reference image")
        validate_image(target_img, "Target image")

        if not cfg.enabled:
            return AlignmentResult(
                aligned_image=target_img.copy(),
                reference_image=ref_img.copy(),
                homography_matrix=np.eye(3, dtype=np.float32),
                total_keypoints_ref=0,
                total_keypoints_target=0,
                matches_found=0,
                inliers=0,
                inlier_ratio=1.0,
                success=True,
                status="SKIPPED",
                method="None (Bypassed)",
                message="Alignment skipped by configuration."
            )

        method = cfg.method.upper()
        if method == "ECC":
            return self._align_ecc(ref_img, target_img, cfg)
        if method == "SIFT":
            return self._align_features(ref_img, target_img, cfg, method="SIFT")
        return self._align_features(ref_img, target_img, cfg, method="ORB")

    def _align_features(
        self,
        ref_img: np.ndarray,
        target_img: np.ndarray,
        cfg: AlignmentConfig,
        method: str = "ORB"
    ) -> AlignmentResult:
        ref_gray = to_gray(ref_img)
        target_gray = to_gray(target_img)
        h, w = ref_gray.shape[:2]

        if method == "SIFT":
            detector = cv2.SIFT_create(nfeatures=cfg.max_features)
            norm_type = cv2.NORM_L2
        else:
            detector = cv2.ORB_create(
                nfeatures=cfg.max_features,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=15,
                firstLevel=0,
                WTA_K=2,
                scoreType=cv2.ORB_HARRIS_SCORE,
                patchSize=31
            )
            norm_type = cv2.NORM_HAMMING

        kp_ref, desc_ref = detector.detectAndCompute(ref_gray, None)
        kp_target, desc_target = detector.detectAndCompute(target_gray, None)

        num_ref = len(kp_ref) if kp_ref is not None else 0
        num_target = len(kp_target) if kp_target is not None else 0

        if desc_ref is None or desc_target is None or num_ref < 4 or num_target < 4:
            return AlignmentResult(
                aligned_image=target_img.copy(),
                reference_image=ref_img.copy(),
                homography_matrix=None,
                total_keypoints_ref=num_ref,
                total_keypoints_target=num_target,
                matches_found=0,
                inliers=0,
                inlier_ratio=0.0,
                success=False,
                status="FAILED",
                method=method,
                message=f"Insufficient keypoints detected (Ref: {num_ref}, Target: {num_target}). Kept unaligned image."
            )

        matcher = cv2.BFMatcher(norm_type, crossCheck=False)
        try:
            raw_matches = matcher.knnMatch(desc_target, desc_ref, k=2)
        except cv2.error as e:
            return AlignmentResult(
                aligned_image=target_img.copy(),
                reference_image=ref_img.copy(),
                homography_matrix=None,
                total_keypoints_ref=num_ref,
                total_keypoints_target=num_target,
                matches_found=0,
                inliers=0,
                inlier_ratio=0.0,
                success=False,
                status="FAILED",
                method=method,
                message=f"Feature matcher failure: {str(e)}"
            )

        good_matches: List[cv2.DMatch] = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < cfg.ratio_threshold * n.distance:
                    good_matches.append(m)

        matched_count = len(good_matches)
        min_required = 4 if cfg.transform_model.lower() == "homography" else 3

        if matched_count < min_required:
            return AlignmentResult(
                aligned_image=target_img.copy(),
                reference_image=ref_img.copy(),
                homography_matrix=None,
                total_keypoints_ref=num_ref,
                total_keypoints_target=num_target,
                matches_found=matched_count,
                inliers=0,
                inlier_ratio=0.0,
                success=False,
                status="FAILED",
                method=method,
                message=f"Only {matched_count} good feature matches found (min {min_required} required). Alignment failed."
            )

        src_pts = np.float32([kp_target[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        model = cfg.transform_model.lower()
        matrix = None
        inlier_mask = None
        inlier_count = 0

        if model == "homography":
            matrix, mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, cfg.ransac_threshold
            )
            if mask is not None:
                inlier_mask = mask.ravel().tolist()
                inlier_count = int(np.sum(mask))
        elif model == "affine":
            matrix, mask = cv2.estimateAffine2D(
                src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_threshold
            )
            if mask is not None:
                inlier_mask = mask.ravel().tolist()
                inlier_count = int(np.sum(mask))
        else:
            matrix, mask = cv2.estimateAffinePartial2D(
                src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_threshold
            )
            if mask is not None:
                inlier_mask = mask.ravel().tolist()
                inlier_count = int(np.sum(mask))

        inlier_ratio = inlier_count / max(1, matched_count)

        if inlier_count < cfg.min_inliers_required:
            return AlignmentResult(
                aligned_image=target_img.copy(),
                reference_image=ref_img.copy(),
                homography_matrix=None,
                total_keypoints_ref=num_ref,
                total_keypoints_target=num_target,
                matches_found=matched_count,
                inliers=inlier_count,
                inlier_ratio=float(inlier_ratio),
                success=False,
                status="FAILED",
                method=method,
                message=f"Insufficient RANSAC inliers ({inlier_count} < {cfg.min_inliers_required}). Alignment rejected."
            )

        if matrix is None or not self._is_matrix_stable(matrix):
            return AlignmentResult(
                aligned_image=target_img.copy(),
                reference_image=ref_img.copy(),
                homography_matrix=None,
                total_keypoints_ref=num_ref,
                total_keypoints_target=num_target,
                matches_found=matched_count,
                inliers=inlier_count,
                inlier_ratio=float(inlier_ratio),
                success=False,
                status="FAILED",
                method=method,
                message="Estimated geometric transformation matrix was degenerate. Alignment rejected."
            )

        border_mode = cv2.BORDER_REFLECT101
        if model == "homography":
            aligned = cv2.warpPerspective(
                target_img, matrix, (w, h),
                flags=cv2.INTER_LINEAR, borderMode=border_mode
            )
        else:
            aligned = cv2.warpAffine(
                target_img, matrix, (w, h),
                flags=cv2.INTER_LINEAR, borderMode=border_mode
            )

        matches_vis = None
        try:
            matches_vis = cv2.drawMatches(
                to_rgb(target_img), kp_target,
                to_rgb(ref_img), kp_ref,
                good_matches[:40], None,
                matchColor=(0, 255, 128),
                singlePointColor=(255, 60, 60),
                matchesMask=inlier_mask[:40] if inlier_mask else None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
        except Exception:
            matches_vis = None

        return AlignmentResult(
            aligned_image=aligned,
            reference_image=ref_img.copy(),
            homography_matrix=matrix,
            total_keypoints_ref=num_ref,
            total_keypoints_target=num_target,
            matches_found=matched_count,
            inliers=inlier_count,
            inlier_ratio=float(inlier_ratio),
            success=True,
            status="SUCCESS",
            method=f"{method} + {model.capitalize()} (RANSAC)",
            message=f"Aligned using {inlier_count}/{matched_count} inliers ({inlier_ratio * 100:.1f}%).",
            matches_visualization=matches_vis
        )

    def _align_ecc(
        self,
        ref_img: np.ndarray,
        target_img: np.ndarray,
        cfg: AlignmentConfig
    ) -> AlignmentResult:
        ref_gray = to_gray(ref_img)
        target_gray = to_gray(target_img)
        h, w = ref_gray.shape[:2]

        model = cfg.transform_model.lower()
        if model == "homography":
            warp_mode = cv2.MOTION_HOMOGRAPHY
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        elif model == "affine":
            warp_mode = cv2.MOTION_AFFINE
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        else:
            warp_mode = cv2.MOTION_EUCLIDEAN
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            cfg.ecc_max_iterations,
            cfg.ecc_eps
        )

        try:
            cc, warp_matrix = cv2.findTransformECC(
                ref_gray, target_gray, warp_matrix, warp_mode, criteria,
                inputMask=None, gaussFiltSize=5
            )
            border_mode = cv2.BORDER_REFLECT101
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                aligned = cv2.warpPerspective(
                    target_img, warp_matrix, (w, h),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=border_mode
                )
            else:
                aligned = cv2.warpAffine(
                    target_img, warp_matrix, (w, h),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=border_mode
                )

            return AlignmentResult(
                aligned_image=aligned,
                reference_image=ref_img.copy(),
                homography_matrix=warp_matrix,
                total_keypoints_ref=0,
                total_keypoints_target=0,
                matches_found=0,
                inliers=0,
                inlier_ratio=1.0,
                success=True,
                status="SUCCESS",
                method="ECC Correlation Maximization",
                message=f"ECC converged with correlation coefficient {cc:.4f}."
            )
        except cv2.error as e:
            return AlignmentResult(
                aligned_image=target_img.copy(),
                reference_image=ref_img.copy(),
                homography_matrix=None,
                total_keypoints_ref=0,
                total_keypoints_target=0,
                matches_found=0,
                inliers=0,
                inlier_ratio=0.0,
                success=False,
                status="FAILED",
                method="ECC",
                message=f"ECC optimization failed: {str(e)}. Original image kept."
            )

    @staticmethod
    def _is_matrix_stable(matrix: np.ndarray) -> bool:
        if matrix.shape == (3, 3):
            det = float(np.linalg.det(matrix[:2, :2]))
            if det <= 0.05 or det >= 20.0 or np.isnan(det) or np.isinf(det):
                return False
        elif matrix.shape == (2, 3):
            det = float(np.linalg.det(matrix[:, :2]))
            if det <= 0.05 or det >= 20.0 or np.isnan(det) or np.isinf(det):
                return False
        return True


def align_images(
    ref_img: np.ndarray,
    target_img: np.ndarray,
    config: Optional[AlignmentConfig] = None
) -> AlignmentResult:
    aligner = ImageAligner(config)
    return aligner.align(ref_img, target_img)
