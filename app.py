import io
import json
import os
import tempfile
from typing import Optional, Tuple, Dict, Any

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from core.engine import ImageDiffEngine, ImageDiffResult
from core.alignment import AlignmentConfig
from core.change_detection import ChangeDetectionConfig
from core.comparison import ComparisonConfig
from core.preprocessing import PreprocessingConfig
from core.visualization import VisualizationConfig, Visualizer
from utils.image_utils import load_image, to_pil, to_bytes, to_rgb

st.set_page_config(
    page_title="ImageDiff — Computer Vision Analysis Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DESIGN_SYSTEM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #e2e8f0;
}

code, kbd, samp, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

.stApp {
    background-color: #0b0f17;
}

section[data-testid="stSidebar"] {
    background-color: #080c13;
    border-right: 1px solid #1e2638;
}

.platform-header {
    background-color: #111724;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 24px 28px;
    margin-bottom: 24px;
}

.platform-title {
    font-size: 1.85rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.02em;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.platform-title span.badge {
    font-size: 0.72rem;
    font-weight: 600;
    color: #38bdf8;
    background-color: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.25);
    padding: 2px 8px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.platform-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin-top: 6px;
    margin-bottom: 0;
    line-height: 1.5;
}

.metric-tile {
    background-color: #111724;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 16px 20px;
    position: relative;
    overflow: hidden;
}

.metric-tile::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 3px;
    height: 100%;
    background-color: #0284c7;
}

.metric-tile.alert::before {
    background-color: #ef4444;
}

.metric-tile.success::before {
    background-color: #10b981;
}

.metric-name {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin-bottom: 4px;
}

.metric-figure {
    font-size: 1.85rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.1;
    margin-bottom: 4px;
}

.metric-footnote {
    font-size: 0.75rem;
    color: #64748b;
}

.surface-card {
    background-color: #111724;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
}

.surface-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 12px;
}

.empty-state-box {
    background-color: #111724;
    border: 1px dashed #2d3748;
    border-radius: 10px;
    padding: 48px 32px;
    text-align: center;
    margin: 20px 0;
}

.empty-state-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 8px;
}

.empty-state-description {
    font-size: 0.95rem;
    color: #94a3b8;
    max-width: 650px;
    margin: 0 auto 24px auto;
    line-height: 1.6;
}

.step-card {
    background-color: #080c13;
    border: 1px solid #1e2638;
    border-radius: 8px;
    padding: 16px;
    text-align: left;
    height: 100%;
}

.step-number {
    font-size: 0.8rem;
    font-weight: 700;
    color: #38bdf8;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.step-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 4px;
}

.step-desc {
    font-size: 0.82rem;
    color: #64748b;
    line-height: 1.4;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 18px;
}

.stTabs [data-baseweb="tab"] {
    height: 42px;
    padding: 0 16px;
    font-size: 0.88rem;
    font-weight: 500;
    color: #94a3b8;
    background-color: transparent;
    border-radius: 6px 6px 0 0;
}

.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    background-color: #111724 !important;
    border-bottom: 2px solid #38bdf8 !important;
}

.stDownloadButton button {
    background-color: #132238;
    color: #38bdf8;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    font-weight: 500;
    transition: all 0.2s ease;
}

.stDownloadButton button:hover {
    background-color: #1e3a5f;
    border-color: #38bdf8;
    color: #ffffff;
}

.image-preview-frame {
    background-color: #080c13;
    border: 1px solid #1e2638;
    border-radius: 6px;
    padding: 8px;
}
</style>
"""

st.markdown(DESIGN_SYSTEM_CSS, unsafe_allow_html=True)

SAMPLE_PRESETS = {
    "None (Use Uploaded Images)": None,
    "Tabletop Product Objects (Addition / Removal)": "samples/desk_objects",
    "Invoice Document (Text & Audit Alteration)": "samples/document_tampering",
    "Electronics PCB (Missing Capacitor & Trace Scratch)": "samples/pcb_defect",
    "Architectural Facade (Structural Modification)": "samples/building_facade",
}


def load_preset_pair(preset_base: str) -> Tuple[np.ndarray, np.ndarray]:
    path_a = f"{preset_base}_before.png"
    path_b = f"{preset_base}_after.png"
    if not (os.path.exists(path_a) and os.path.exists(path_b)):
        from samples.sample_generator import create_all_samples
        create_all_samples()
    img_a = load_image(path_a, target_mode="RGB")
    img_b = load_image(path_b, target_mode="RGB")
    return img_a, img_b


with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 16px 0;">
            <div style="font-size: 1.4rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em;">
                ImageDiff <span style="font-size: 0.8rem; font-weight: 500; color: #38bdf8;">PRO</span>
            </div>
            <div style="font-size: 0.78rem; color: #64748b; margin-top: 2px;">
                Image Processing Analysis Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("<p style='font-size: 0.8rem; font-weight: 600; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em; margin-bottom: 8px;'>Navigation</p>", unsafe_allow_html=True)
    selected_view = st.radio(
        "Navigation",
        options=["Comparison Workspace", "Detailed Analysis", "Export Artifacts", "IPA Methodology"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("<p style='font-size: 0.8rem; font-weight: 600; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em; margin-bottom: 8px;'>Comparison Settings</p>", unsafe_allow_html=True)

    threshold_value = st.slider(
        "Change Threshold",
        min_value=5,
        max_value=120,
        value=30,
        step=5,
        help="Minimum intensity divergence (0-255) for pixels to be categorized as changed.",
    )

    enable_alignment = st.checkbox(
        "Enable Camera Alignment",
        value=False,
        help="Compensates for handheld camera jitter, scale shifts, and perspective rotation using ORB keypoint matching and RANSAC homography.",
    )

    min_region_area = st.slider(
        "Min Changed Region Area (px²)",
        min_value=10,
        max_value=500,
        value=40,
        step=10,
        help="Contours with area smaller than this threshold are filtered out as noise.",
    )

    with st.expander("Visualization Settings", expanded=False):
        colormap_name = st.selectbox(
            "Heatmap Palette",
            options=["turbo", "jet", "inferno", "viridis", "magma"],
            index=0,
            help="Colormap used to render difference fields.",
        )
        overlay_opacity = st.slider(
            "Highlight Opacity",
            min_value=0.1,
            max_value=0.8,
            value=0.45,
            step=0.05,
            help="Alpha transparency for the change highlight overlay.",
        )
        blur_kernel = st.selectbox(
            "Gaussian Denoise Kernel",
            options=[3, 5, 7],
            index=1,
            help="Size of Gaussian blur filter applied during change mask generation.",
        )

    st.markdown("---")

    st.markdown("<p style='font-size: 0.8rem; font-weight: 600; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em; margin-bottom: 6px;'>About ImageDiff</p>", unsafe_allow_html=True)
    st.caption(
        "ImageDiff is a general-purpose image comparison and change detection system designed for the "
        "**Image Processing Analysis (IPA)** academic curriculum. Built with Python 3.11, OpenCV, and scikit-image."
    )


st.markdown(
    """
    <div class="platform-header">
        <h1 class="platform-title">
            ImageDiff
            <span class="badge">Computer Vision Engine</span>
        </h1>
        <p class="platform-subtitle">
            Intelligent Image Comparison & Change Detection — Identify, visualize, and quantify structural and pixel-level 
            differences across any visual domain, including objects, buildings, rooms, documents, products, electronics, or medical scans.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

preset_col1, preset_col2 = st.columns([1, 2])
with preset_col1:
    selected_preset = st.selectbox(
        "Load Demonstration Preset",
        options=list(SAMPLE_PRESETS.keys()),
        index=0,
        help="Select a sample pair to inspect differences without uploading your own files.",
    )

with preset_col2:
    st.caption(
        "Tip: Upload your own images using the cards below, or select a sample preset for an immediate demonstration "
        "of product changes, document fraud, printed circuit board defects, or architectural modifications."
    )

upload_col_a, upload_col_b = st.columns(2)

with upload_col_a:
    st.markdown(
        """
        <div style="font-weight: 600; font-size: 0.95rem; color: #f1f5f9; margin-bottom: 6px;">
            Image A: Original / Reference Image
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file_a = st.file_uploader(
        "Upload Reference Image (Image A)",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
        key="uploader_file_a",
        label_visibility="collapsed",
    )

with upload_col_b:
    st.markdown(
        """
        <div style="font-weight: 600; font-size: 0.95rem; color: #f1f5f9; margin-bottom: 6px;">
            Image B: Modified / Comparison Image
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file_b = st.file_uploader(
        "Upload Comparison Image (Image B)",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
        key="uploader_file_b",
        label_visibility="collapsed",
    )

img_a_input: Optional[np.ndarray] = None
img_b_input: Optional[np.ndarray] = None
image_source_label = ""

if uploaded_file_a is not None and uploaded_file_b is not None:
    try:
        img_a_input = load_image(uploaded_file_a, target_mode="RGB")
        img_b_input = load_image(uploaded_file_b, target_mode="RGB")
        image_source_label = f"Uploaded files: {uploaded_file_a.name} & {uploaded_file_b.name}"
    except Exception as e:
        st.error(f"Error reading uploaded images: {str(e)}")
elif selected_preset != "None (Use Uploaded Images)" and SAMPLE_PRESETS[selected_preset] is not None:
    try:
        preset_base = SAMPLE_PRESETS[selected_preset]
        img_a_input, img_b_input = load_preset_pair(preset_base)
        image_source_label = f"Sample preset: {selected_preset}"
    except Exception as e:
        st.error(f"Error loading preset: {str(e)}")
elif uploaded_file_a is not None and uploaded_file_b is None:
    try:
        img_a_input = load_image(uploaded_file_a, target_mode="RGB")
    except Exception:
        pass
elif uploaded_file_b is not None and uploaded_file_a is None:
    try:
        img_b_input = load_image(uploaded_file_b, target_mode="RGB")
    except Exception:
        pass

if img_a_input is not None or img_b_input is not None:
    prev_col1, prev_col2 = st.columns(2)
    with prev_col1:
        if img_a_input is not None:
            st.image(to_pil(img_a_input), caption=f"Image A — {img_a_input.shape[1]}x{img_a_input.shape[0]} px", use_container_width=True)
        else:
            st.info("Image A pending upload.")
    with prev_col2:
        if img_b_input is not None:
            st.image(to_pil(img_b_input), caption=f"Image B — {img_b_input.shape[1]}x{img_b_input.shape[0]} px", use_container_width=True)
        else:
            st.info("Image B pending upload.")

if img_a_input is None or img_b_input is None:
    st.markdown(
        """
        <div class="empty-state-box">
            <div class="empty-state-title">Ready for Image Comparison</div>
            <div class="empty-state-description">
                Upload two images above or select a demonstration preset from the dropdown to start the 
                automated change detection and structural comparison pipeline.
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; max-width: 800px; margin: 0 auto;">
                <div class="step-card">
                    <div class="step-number">Step 01</div>
                    <div class="step-title">Select Reference (A)</div>
                    <div class="step-desc">Upload the baseline, original, or unaltered photograph.</div>
                </div>
                <div class="step-card">
                    <div class="step-number">Step 02</div>
                    <div class="step-title">Select Target (B)</div>
                    <div class="step-desc">Upload the modified, second, or updated comparison photograph.</div>
                </div>
                <div class="step-card">
                    <div class="step-number">Step 03</div>
                    <div class="step-title">Inspect & Quantify</div>
                    <div class="step-desc">Examine heatmaps, SSIM structural degradation, and segmented bounding boxes.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

engine = ImageDiffEngine()

with st.spinner("Processing image comparison and change detection pipeline..."):
    result: ImageDiffResult = engine.process(
        img_a_input,
        img_b_input,
        align=enable_alignment,
        threshold_value=threshold_value,
        threshold_method="fixed",
        min_contour_area=min_region_area,
        blur_kernel_size=blur_kernel,
        colormap=colormap_name,
        overlay_alpha=overlay_opacity,
    )

if enable_alignment:
    align_status = result.alignment_status
    if align_status.get("aligned", False):
        st.success(f"Camera Registration Active: {align_status.get('message', 'Aligned successfully')}")
    else:
        st.warning(f"Registration Notice: {align_status.get('message', 'Unaligned fallback preserved')}")

m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    sim_pct = result.similarity_score * 100.0
    sim_class = "success" if sim_pct >= 85 else ("alert" if sim_pct < 60 else "")
    st.markdown(
        f"""
        <div class="metric-tile {sim_class}">
            <div class="metric-name">Similarity Score</div>
            <div class="metric-figure">{sim_pct:.1f}%</div>
            <div class="metric-footnote">Normalized pixel closeness</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m_col2:
    diff_pct = result.difference_score * 100.0
    diff_class = "alert" if diff_pct > 15 else ""
    st.markdown(
        f"""
        <div class="metric-tile {diff_class}">
            <div class="metric-name">Difference Score</div>
            <div class="metric-figure">{diff_pct:.1f}%</div>
            <div class="metric-footnote">MAE: {result.mae:.2f} / 255</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m_col3:
    changed_px_pct = result.changed_pixel_percentage
    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-name">Changed Pixels</div>
            <div class="metric-figure">{changed_px_pct:.2f}%</div>
            <div class="metric-footnote">Exceeding threshold {threshold_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m_col4:
    region_count = len(result.detected_regions)
    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-name">Detected Regions</div>
            <div class="metric-figure">{region_count}</div>
            <div class="metric-footnote">Segmented anomalies</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

if selected_view == "Comparison Workspace":
    st.markdown("<div class='surface-title'>Visual Comparison Workspace</div>", unsafe_allow_html=True)

    tab_sbs, tab_slider, tab_heatmap, tab_overlay, tab_mask = st.tabs([
        "Side-by-Side Comparison",
        "Interactive Wipe Slider",
        "Difference Heatmap",
        "Change Detection Overlay",
        "Binary Change Mask",
    ])

    with tab_sbs:
        col_sbs_1, col_sbs_2 = st.columns(2)
        with col_sbs_1:
            st.caption("Reference (Image A)")
            st.image(to_pil(result.reference_image), use_container_width=True)
        with col_sbs_2:
            st.caption("Target with Detected Changes (Image B)")
            st.image(to_pil(result.contour_image), use_container_width=True)

    with tab_slider:
        st.caption("Drag the slider to wipe interactively across the boundary between Reference Image A and Target Image B.")
        slider_val = st.slider("Divider Position (%)", min_value=0, max_value=100, value=50, step=1, label_visibility="collapsed")
        vis_helper = Visualizer()
        split_img = vis_helper.create_split_view(result.reference_image, result.contour_image, split_pct=float(slider_val))
        st.image(to_pil(split_img), use_container_width=True)

    with tab_heatmap:
        col_hm_1, col_hm_2 = st.columns(2)
        with col_hm_1:
            st.caption("Thermal Difference Map (Continuous Field)")
            st.image(to_pil(result.heatmap), use_container_width=True)
            st.caption(f"Colormap: {colormap_name.upper()} — Warmer tones correspond to severe intensity variation.")
        with col_hm_2:
            st.caption("Blended Thermal Heatmap Overlay")
            blended_hm = vis_helper.create_heatmap(result.difference_map, background_img=result.aligned_target, config=VisualizationConfig(colormap=colormap_name, heatmap_alpha=overlay_opacity))
            st.image(to_pil(blended_hm), use_container_width=True)
            st.caption("Projected thermal delta overlaid onto the target image.")

    with tab_overlay:
        st.caption("Red highlight overlay identifying localized change boundaries on Image B.")
        st.image(to_pil(result.overlay_image), use_container_width=True)
        st.caption(f"Rendered at {overlay_opacity * 100:.0f}% opacity across classified changed pixels.")

    with tab_mask:
        col_m_1, col_m_2 = st.columns(2)
        with col_m_1:
            st.caption("Binary Change Mask (Cleaned via Morphological Opening & Closing)")
            mask_rgb = vis_helper.generate_binary_mask_image(result.change_mask)
            st.image(to_pil(mask_rgb), use_container_width=True)
            st.caption("White = Classified Change (255), Black = Invariant Background (0).")
        with col_m_2:
            st.caption("Raw Difference Intensity Map")
            diff_norm = vis_helper.generate_binary_mask_image(result.difference_map)
            st.image(to_pil(diff_norm), use_container_width=True)
            st.caption("Unthresholded absolute intensity differences |Image A - Image B|.")

elif selected_view == "Detailed Analysis":
    st.markdown("<div class='surface-title'>Detailed Quantitative Analysis</div>", unsafe_allow_html=True)

    col_stat1, col_stat2 = st.columns(2)

    with col_stat1:
        st.markdown(
            """
            <div class="surface-card">
                <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 12px; color: #f8fafc;">
                    Structural & Photometric Error Metrics
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(f"**Structural Similarity Index (SSIM)**: `{result.ssim_score:.4f}` (Raw SSIM: `{result.raw_ssim:.4f}`)")
        st.caption("Measures perceptual fidelity, luminance, and texture correlation using local window covariance.")

        st.write(f"**Mean Absolute Error (MAE)**: `{result.mae:.4f}`")
        st.caption("Average pixel magnitude delta across all color channels in range [0, 255].")

        st.write(f"**Mean Squared Error (MSE)**: `{result.mse:.4f}`")
        st.caption("Average squared Euclidean distance penalizing large local discrepancies heavily.")

        st.write(f"**Peak Signal-to-Noise Ratio (PSNR)**: `{result.psnr:.2f} dB`")
        st.caption("Logarithmic ratio between peak signal energy and mean squared noise.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_stat2:
        st.markdown(
            """
            <div class="surface-card">
                <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 12px; color: #f8fafc;">
                    Spatial Footprint & Segmentation Summary
                </div>
            """,
            unsafe_allow_html=True,
        )
        total_px = result.change_mask.size
        changed_px_count = int(np.count_nonzero(result.change_mask))

        st.write(f"**Total Image Pixels**: `{total_px:,} px` ({result.change_mask.shape[1]}x{result.change_mask.shape[0]})")
        st.write(f"**Pixels Classified as Changed**: `{changed_px_count:,} px`")
        st.write(f"**Percentage of Changed Pixels**: `{result.changed_pixel_percentage:.3f}%`")
        st.write(f"**Total Segmented Change Regions**: `{len(result.detected_regions)}`")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='surface-title' style='margin-top: 20px;'>Segmented Anomaly Regions Table</div>", unsafe_allow_html=True)

    if result.detected_regions:
        df_regions = pd.DataFrame(result.detected_regions)
        cols_to_show = ["id", "x", "y", "width", "height", "area", "area_percentage", "centroid", "aspect_ratio", "extent"]
        existing_cols = [c for c in cols_to_show if c in df_regions.columns]
        df_display = df_regions[existing_cols].rename(columns={
            "id": "Region ID",
            "x": "X (px)",
            "y": "Y (px)",
            "width": "Width",
            "height": "Height",
            "area": "Area (px²)",
            "area_percentage": "Area (%)",
            "centroid": "Centroid (X,Y)",
            "aspect_ratio": "Aspect Ratio",
            "extent": "Extent",
        })
        st.dataframe(df_display, use_container_width=True)
    else:
        st.info("No change regions met the configured minimum contour area threshold.")

elif selected_view == "Export Artifacts":
    st.markdown("<div class='surface-title'>Export Inspection Artifacts & Reports</div>", unsafe_allow_html=True)
    st.caption("Download generated visual artifacts, structured JSON audit logs, and region CSV datasets.")

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown(
            """
            <div class="surface-card">
                <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 14px; color: #f8fafc;">
                    Image Output Downloads
                </div>
            """,
            unsafe_allow_html=True,
        )

        st.download_button(
            label="Download Difference Heatmap (PNG)",
            data=to_bytes(result.heatmap, format="PNG"),
            file_name="imagediff_heatmap.png",
            mime="image/png",
            use_container_width=True,
        )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        mask_png = to_bytes(result.change_mask, format="PNG")
        st.download_button(
            label="Download Binary Change Mask (PNG)",
            data=mask_png,
            file_name="imagediff_change_mask.png",
            mime="image/png",
            use_container_width=True,
        )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        st.download_button(
            label="Download Red Change Overlay (PNG)",
            data=to_bytes(result.overlay_image, format="PNG"),
            file_name="imagediff_change_overlay.png",
            mime="image/png",
            use_container_width=True,
        )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        if result.side_by_side is not None:
            st.download_button(
                label="Download Side-by-Side Composite (PNG)",
                data=to_bytes(result.side_by_side, format="PNG"),
                file_name="imagediff_side_by_side.png",
                mime="image/png",
                use_container_width=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with exp_col2:
        st.markdown(
            """
            <div class="surface-card">
                <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 14px; color: #f8fafc;">
                    Structured Audit Logs & Tabular Data
                </div>
            """,
            unsafe_allow_html=True,
        )

        audit_payload = {
            "application": "ImageDiff — Computer Vision Analysis Platform",
            "source_info": image_source_label,
            "configuration": {
                "threshold_value": threshold_value,
                "camera_alignment_enabled": enable_alignment,
                "min_region_area_px": min_region_area,
                "colormap": colormap_name,
            },
            "metrics": {
                "similarity_score": round(result.similarity_score, 4),
                "difference_score": round(result.difference_score, 4),
                "mae": round(result.mae, 4),
                "mse": round(result.mse, 4),
                "psnr_db": round(result.psnr, 2),
                "ssim_score": round(result.ssim_score, 4),
                "raw_ssim": round(result.raw_ssim, 4),
                "changed_pixel_percentage": round(result.changed_pixel_percentage, 3),
            },
            "alignment_diagnostics": result.alignment_status,
            "detected_regions": result.detected_regions,
        }
        json_report_str = json.dumps(audit_payload, indent=2)

        st.download_button(
            label="Download Full Audit Report (JSON)",
            data=json_report_str,
            file_name="imagediff_comparison_report.json",
            mime="application/json",
            use_container_width=True,
        )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        if result.detected_regions:
            csv_str = pd.DataFrame(result.detected_regions).to_csv(index=False)
            st.download_button(
                label="Download Detected Regions Data (CSV)",
                data=csv_str,
                file_name="imagediff_detected_regions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

elif selected_view == "IPA Methodology":
    st.markdown("<div class='surface-title'>Image Processing Analysis (IPA) — Algorithm Methodology</div>", unsafe_allow_html=True)
    st.caption("Mathematical and algorithmic reference of computer vision techniques used in ImageDiff.")

    with st.expander("1. Image Preprocessing & Format Harmonization", expanded=True):
        st.markdown(
            """
            Image ingestion accepts diverse image sources (Grayscale, RGB, and RGBA).
            - **Alpha Channel Compositing**: RGBA inputs are composited over a neutral white canvas:
              $$I_{RGB} = I_{RGB} \\cdot \\alpha + I_{white} \\cdot (1 - \\alpha)$$
            - **Aspect-Ratio Preserving Normalization**: When comparing two images of disparate resolutions $(H_1, W_1) \\ne (H_2, W_2)$, 
              images are scaled with area-based interpolation (`cv2.INTER_AREA`) and padded to avoid geometric distortion.
            - **Spatial Filtering**: Gaussian convolution suppresses high-frequency sensor noise:
              $$G(x, y) = \\frac{1}{2\\pi\\sigma^2} e^{-\\frac{x^2+y^2}{2\\sigma^2}}$$
            """
        )

    with st.expander("2. Feature-Based Alignment (ORB + RANSAC Homography)"):
        st.markdown(
            """
            Handheld cameras introduce translation, rotation, and perspective skew between photographs.
            ImageDiff aligns Image B to the coordinate frame of Image A:
            1. **ORB Feature Detection**: Detects multiscale corners using FAST and computes orientation-compensated BRIEF binary descriptors.
            2. **Hamming Distance Matching & Lowe's Ratio Test**: Discards ambiguous feature correspondences where the closest match distance exceeds 75% of the second closest match.
            3. **RANSAC Homography Estimation**: Solves the 8-degrees-of-freedom perspective transformation matrix $\\mathbf{H}$:
               $$\\begin{bmatrix} x' \\\\ y' \\\\ 1 \\end{bmatrix} \\sim \\mathbf{H} \\begin{bmatrix} x \\\\ y \\\\ 1 \\end{bmatrix}$$
            4. **Perspective Warping**: Warps target Image B onto the coordinate plane of Image A using bilinear interpolation.
            """
        )

    with st.expander("3. Absolute Pixel Differencing & Error Formulation"):
        st.markdown(
            """
            Pixel differences are computed on floating-point matrices:
            $$\\text{diff}(x, y) = |I_A(x, y) - I_B(x, y)|$$
            From this difference field, the following metrics are computed:
            - **Mean Absolute Error (MAE)**:
              $$\\text{MAE} = \\frac{1}{M \\cdot N} \\sum_{x=0}^{M-1} \\sum_{y=0}^{N-1} |I_A(x, y) - I_B(x, y)|$$
            - **Mean Squared Error (MSE)**:
              $$\\text{MSE} = \\frac{1}{M \\cdot N} \\sum_{x=0}^{M-1} \\sum_{y=0}^{N-1} (I_A(x, y) - I_B(x, y))^2$$
            - **Normalized Similarity Score**:
              $$\\text{Similarity} = 1.0 - \\frac{\\text{MAE}}{255.0} \\in [0.0, 1.0]$$
            """
        )

    with st.expander("4. Structural Similarity Index (SSIM)"):
        st.markdown(
            """
            SSIM models perceptual quality across three dimensions: **Luminance** ($l$), **Contrast** ($c$), and **Structure** ($s$):
            $$\\text{SSIM}(x, y) = \\frac{(2\\mu_x\\mu_y + C_1)(2\\sigma_{xy} + C_2)}{(\\mu_x^2 + \\mu_y^2 + C_1)(\\sigma_x^2 + \\sigma_y^2 + C_2)}$$
            Where:
            - $\\mu_x, \\mu_y$: Local mean intensities within a Gaussian spatial window.
            - $\\sigma_x^2, \\sigma_y^2$: Local sample variance within window patches.
            - $\\sigma_{xy}$: Local cross-covariance between windows $x$ and $y$.
            - $C_1 = (K_1 L)^2, C_2 = (K_2 L)^2$: Numerical stability constants with dynamic range $L = 255$.
            """
        )

    with st.expander("5. Decision Thresholding & Mathematical Morphology"):
        st.markdown(
            """
            Converting continuous difference fields into a binary change mask $M(x, y) \\in \\{0, 255\\}$:
            $$M(x, y) = \\begin{cases} 255 & \\text{if } \\text{diff}(x, y) > \\tau \\\\ 0 & \\text{otherwise} \\end{cases}$$
            Mathematical morphology with structuring element $B$ refines the mask:
            - **Morphological Opening** ($M \\circ B = (M \\ominus B) \\oplus B$): Erosion followed by Dilation. Eliminates isolated single-pixel noise.
            - **Morphological Closing** ($M \\bullet B = (M \\oplus B) \\ominus B$): Dilation followed by Erosion. Fills internal pinholes and bridges fractures.
            """
        )

    with st.expander("6. Contour Detection & Region Geometry Extraction"):
        st.markdown(
            """
            Contour tracing extracts closed polygons from the refined binary mask using Suzuki's border following algorithm (`cv2.findContours`):
            - **Bounding Rectangle**: $(x, y, w, h) = \\text{cv2.boundingRect}(C)$
            - **Contour Area**: $A = \\text{cv2.contourArea}(C)$ (regions with $A < \\text{min\\_area}$ are discarded)
            - **Spatial Centroid**: Derived via image spatial moments:
              $$c_x = \\frac{M_{10}}{M_{00}}, \\quad c_y = \\frac{M_{01}}{M_{00}}$$
            """
        )

st.markdown("---")
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: #64748b; padding: 4px 0 16px 0;">
        <div><strong>ImageDiff</strong> &bull; Computer Vision Engine</div>
        <div>Image Processing Analysis (IPA) Project</div>
    </div>
    """,
    unsafe_allow_html=True,
)
