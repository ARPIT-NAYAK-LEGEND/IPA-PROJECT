# ImageDiff — Image Comparison and Change Detection System

ImageDiff is an image comparison and change detection system designed for the Image Processing Analysis (IPA) course. It compares two related images across arbitrary domains (objects, documents, printed circuit boards, architecture, aerial photography), calculates structural and pixel-level divergence metrics, and segments changed regions using classical computer vision techniques.

## Features

- **Domain-Agnostic Comparison**: Works on any two related images without domain-specific assumptions.
- **Camera Alignment**: ORB feature matching with Lowe's ratio test and RANSAC homography estimation to align slightly translated or rotated images.
- **Perceptual and Pixel Metrics**: Structural Similarity Index (SSIM), Mean Absolute Error (MAE), Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and normalized similarity/difference scores.
- **Change Segmentation**: Intensity thresholding, mathematical morphology (opening and closing), and contour extraction with bounding boxes and centroid coordinates.
- **Visualizations**: Continuous thermal heatmaps (Turbo, Jet, Inferno, Viridis, Magma), binary change masks, red highlight overlays, side-by-side views, and an interactive wipe slider.
- **Data Export**: Export results to PNG images, CSV region summaries, and JSON audit reports.

## Project Structure

```
x:/IPA/
├── app.py                      # Streamlit dashboard interface
├── core/
│   ├── __init__.py             # Core package initialization
│   ├── preprocessing.py        # Aspect ratio preservation, resizing, and format conversion
│   ├── alignment.py            # ORB keypoint matching and RANSAC homography
│   ├── comparison.py           # SSIM, MAE, MSE, and PSNR calculations
│   ├── change_detection.py     # Thresholding, morphology, and contour detection
│   ├── visualization.py        # Heatmaps, overlays, split views, and contour drawing
│   └── engine.py               # ImageDiffEngine orchestrator and ImageDiffResult
├── utils/
│   ├── __init__.py             # Utilities package initialization
│   └── image_utils.py          # Format conversions, validation, and file I/O
├── samples/
│   ├── sample_generator.py     # Test pairs generator for multiple domains
│   └── ...                     # Pre-generated sample images
├── tests/
│   ├── test_preprocessing.py   # Preprocessing unit tests
│   ├── test_alignment.py       # Alignment unit tests
│   ├── test_comparison.py      # Metric calculation unit tests
│   ├── test_change_detection.py# Morphology and contour unit tests
│   ├── test_visualization.py   # Visualization generator tests
│   ├── test_engine.py          # Engine and result contract tests
│   ├── test_image_utils.py     # Image utility tests
│   └── test_integration.py     # Multi-domain end-to-end integration tests
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Algorithms and Methodology

### 1. Structural Similarity Index (SSIM)
SSIM evaluates visual degradation by comparing luminance, contrast, and structure over local Gaussian patches:

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

where $\mu$ is local mean, $\sigma^2$ is local variance, $\sigma_{xy}$ is cross-covariance, and $C_1, C_2$ are stabilization constants.

### 2. Feature-Based Alignment (ORB + RANSAC)
To map points from Image B to Image A:

$$\mathbf{x}' \sim \mathbf{H} \mathbf{x} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

ORB detects oriented FAST corners and computes BRIEF descriptors. Matches passing Lowe's ratio test ($d_1 < 0.75 \cdot d_2$) are used with RANSAC to compute the homography matrix $\mathbf{H}$. Image B is then warped using bilinear interpolation.

### 3. Absolute Pixel Differencing
Pixel difference is computed on floating-point matrices:

$$\text{diff}(x, y) = |I_A(x, y) - I_B(x, y)|$$

- Mean Absolute Error (MAE): $\frac{1}{MN} \sum |I_A - I_B|$
- Mean Squared Error (MSE): $\frac{1}{MN} \sum (I_A - I_B)^2$
- Similarity Score: $1.0 - \frac{\text{MAE}}{255.0} \in [0.0, 1.0]$

### 4. Mathematical Morphology and Contours
Thresholding converts the difference map into a binary mask:

$$M(x, y) = \begin{cases} 255 & \text{if } \text{diff}(x, y) \ge \tau \\ 0 & \text{otherwise} \end{cases}$$

- Morphological Opening ($M \circ B$): Erosion followed by dilation to eliminate isolated noise.
- Morphological Closing ($M \bullet B$): Dilation followed by erosion to bridge gaps and fill holes.
- Contour Extraction: Suzuki's border following algorithm (`cv2.findContours`) segments individual changed regions and calculates bounding boxes and centroids.

## Installation

### Prerequisites
- Python 3.11

### Virtual Environment Setup

Using `uv`:
```powershell
uv venv .venv --python 3.11
.\.venv\Scripts\activate
uv pip install -r requirements.txt
```

Or using standard `venv` and `pip`:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Application

Launch the Streamlit web dashboard:

```powershell
streamlit run app.py
```

The application will be accessible at:
```
http://localhost:8501
```

## Running Tests

Run the test suite using `pytest`:

```powershell
pytest -v tests
```

All 27 unit and integration tests should pass.

## Limitations

- **3D Parallax and Non-Planar Scenes**: A single 2D planar homography assumes approximately flat surfaces or pure rotational camera motion. Complex non-planar scenes with heavy depth variation cannot be fully registered with 2D homography alone.
- **Global Illumination Changes**: Large shifts in overall lighting (such as daylight vs. night) create global intensity deltas. Using higher threshold values helps isolate localized geometric changes from ambient lighting shifts.
- **Non-Semantic Processing**: ImageDiff uses classical image processing algorithms (pixel statistics, structural correlation, and morphology) rather than deep learning object recognition models.
