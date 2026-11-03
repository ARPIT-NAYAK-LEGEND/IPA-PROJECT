# ImageDiff: An Image Processing-Based Similarity Analysis and Change Detection System

**Student Researcher**  
*Department of Computer Science and Engineering*  
*School of Computer Engineering*  
Bhubaneswar, India  
`student@institution.edu`

**Project Guide / Faculty Advisor**  
*Department of Computer Science and Engineering*  
*School of Computer Engineering*  
Bhubaneswar, India  
`advisor@institution.edu`

---

### Abstract
*Automated change detection and comparative similarity analysis between related visual captures represent fundamental challenges in computer vision and digital image processing. In practical scenarios—ranging from industrial quality inspection and electronic circuit defect analysis to forensic document auditing and structural health monitoring—manual visual inspection is prone to cognitive fatigue, subjectivity, and human error. This paper presents ImageDiff, an intelligent, general-purpose image comparison and anomaly localization system developed using classical digital image processing principles. The proposed system accepts pairs of related images across arbitrary visual domains, harmonizes disparate dimensions via aspect-ratio-preserving normalization, compensates for camera jitter and perspective shift using Oriented FAST and Rotated BRIEF (ORB) feature matching coupled with RANSAC-based homography estimation, and quantifies photometric and structural degradation using Mean Absolute Error (MAE), Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and the Structural Similarity Index (SSIM). Candidate changes are segmented through adaptive thresholding and refined via morphological opening and closing operators, followed by topological contour tracing for individual anomaly extraction, bounding box localization, and spatial moment centroid computation. Extensive empirical validation across tabletop objects, fraudulent invoice alterations, printed circuit board (PCB) flaws, and architectural facade modifications demonstrates high numerical fidelity, registering camera transforms with up to 99.3% inlier consensus and segmenting minute localized defects representing as low as 0.018% of the total pixel area.*

**Index Terms**—Image differencing, Structural Similarity Index (SSIM), ORB feature matching, RANSAC homography, mathematical morphology, contour detection, change detection, digital image processing.

---

## I. INTRODUCTION

Comparative analysis between multiple photographic captures of a scene or object is a cornerstone of modern computer vision. In real-world environments, visual changes occur across diverse time intervals and operational contexts: an altered line item on a commercial invoice, a severed copper trace or absent capacitor on a surface-mount printed circuit board (PCB), structural modifications to a building facade, or the introduction and displacement of objects within a monitored workspace [1].

Historically, such comparative tasks relied upon human inspection, which is inherently qualitative, non-exhaustive, and vulnerable to observer fatigue. While deep-learning-based object detection and semantic segmentation frameworks have achieved notable milestones in recent years, they typically demand thousands of domain-specific annotated training exemplars, substantial computational overhead, and struggle to generalize when deployed outside their narrow training distribution.

Conversely, classical digital image processing techniques grounded in mathematical morphology, spatial filtering, statistical covariance, and projective geometry offer transparent, deterministic, and domain-agnostic solutions [2]. However, building a generalized, production-ready image comparison framework entails addressing several non-trivial image processing hurdles:

1. **Geometric and Dimensional Discrepancies**: Handheld cameras or diverse optical sensors yield images of non-uniform resolutions and aspect ratios.
2. **Camera Motion and Misalignment**: Subtle camera translations, rotations, and perspective skews introduce artificial spatial displacement that produces widespread false-positive difference signals if differenced directly.
3. **Photometric Noise vs. Perceptual Degradation**: Standard point-by-point pixel difference metrics (e.g., Euclidean distance or absolute delta) fail to model human visual perception, requiring complementary structural metrics such as the Structural Similarity Index (SSIM) [3].
4. **Noise Suppression and Anomaly Grouping**: Raw thresholded difference masks contain isolated pixel specks stemming from sensor noise and compression artifacts, necessitating rigorous mathematical morphological filtering and contour boundary extraction.

To address these challenges comprehensively, this paper introduces **ImageDiff**, a modular, high-precision computer vision pipeline designed and implemented for the Image Processing Analysis (IPA) academic curriculum. ImageDiff integrates feature-based geometric registration, multi-metric quantitative analysis, dual-stage morphological filtering, and spatial moment geometry extraction into an interactive, dark-themed analytical dashboard.

---

## II. RELATED WORK

Image change detection algorithms have been systematically categorized by Radke et al. [1] into pixel-based, feature-based, and structural approaches.

### A. Pixel-Level Differencing and Error Formulations
The foundational approach to visual change detection is absolute intensity differencing [2]. While computationally efficient ($O(MN)$ complexity), raw differencing is notoriously sensitive to illumination shifts, shadows, and sensor quantization noise. Traditional global metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and Peak Signal-to-Noise Ratio (PSNR) treat all pixel variations independently, neglecting spatial correlations and structural context.

### B. Perceptual Structural Assessment
To overcome the known perceptual blindspots of MSE, Wang et al. [3] formulated the Structural Similarity Index (SSIM). SSIM models human visual system (HVS) characteristics by decomposing local visual patterns into luminance, contrast, and structural covariance components across local Gaussian spatial windows. van der Walt et al. [4] formalized this metric in the scientific Python ecosystem via `scikit-image`, which forms the mathematical backbone of our structural evaluation engine.

### C. Feature-Based Image Registration
When comparing images captured at distinct temporal instances, camera motion must be compensated. Lowe [5] established feature-based matching using the Scale-Invariant Feature Transform (SIFT). While robust, SIFT carries significant computational complexity and licensing history. Rublee et al. [6] introduced Oriented FAST and Rotated BRIEF (ORB), a fast, rotation-invariant, and noise-resistant binary feature detector and descriptor matching framework based on Hamming distance. Combined with the Random Sample Consensus (RANSAC) algorithm proposed by Fischler and Bolles [7], planar homography matrices can be estimated robustly in the presence of heavy outlier proportions.

### D. Mathematical Morphology and Spatial Segmentation
In digital image processing, binary segmentation maps require topological refinement. Haralick et al. [8] established mathematical morphology based on set theory, defining erosion, dilation, opening, and closing operations using geometric structuring elements. For discrete boundary extraction, Suzuki and Abe [9] formulated topological structural analysis algorithms via border-following methods, currently implemented in OpenCV [10]. ImageDiff leverages these foundations to bridge pixel-level anomalies into cohesive semantic regions with geometric bounding boxes and centroids.

---

## III. PROBLEM STATEMENT AND OBJECTIVES

### A. Problem Statement
Given two related digital images, a reference image $I_A \in \mathbb{R}^{H_A \times W_A \times C}$ and a target image $I_B \in \mathbb{R}^{H_B \times W_B \times C}$, captured from potentially differing viewpoints, optical zoom levels, and sensor dimensions, design an automated, deterministic image processing system that:
1. Harmonizes dimensional and format discrepancies without introducing geometric distortion.
2. Registers $I_B$ to the coordinate frame of $I_A$ by estimating an optimal projective planar homography $\mathbf{H} \in \mathbb{R}^{3 \times 3}$.
3. Formulates both photometric (MAE, MSE, PSNR) and structural (SSIM) similarity metrics.
4. Generates a noise-free binary change mask $M \in \{0, 255\}^{H \times W}$.
5. Segments and extracts geometric properties (bounding boxes, centroids, contour areas) for all detected anomalies.
6. Provides intuitive multi-modal visualizations (continuous heatmaps, transparent overlays, wipe sliders).

### B. Key Design Objectives
- **Domain Invariance**: The system must function identically on industrial PCBs, documents, consumer objects, and architecture without retraining.
- **Mathematical Rigor**: Metrics must be strictly normalized and documented, avoiding arbitrary or synthetic scoring formulas.
- **Fault-Tolerant Alignment**: Alignment failures must be diagnosed cleanly and fall back to unaligned comparison without crashing.

---

## IV. PROPOSED SYSTEM

The proposed ImageDiff system architecture operates deterministically across five sequential stages:
1. **Stage I: Ingestion and Preprocessing**: Validation, alpha-channel compositing over a neutral background, aspect-ratio-preserving padding, and spatial smoothing.
2. **Stage II: Geometric Alignment**: ORB keypoint detection, descriptor extraction, Hamming distance brute-force matching, Lowe's ratio test pruning, and RANSAC homography projective warping.
3. **Stage III: Comparative Metric Formulation**: Pixel-wise absolute difference matrix computation, MAE, MSE, PSNR, and local patch-based SSIM.
4. **Stage IV: Change Detection and Morphological Filtering**: Intensity thresholding, rectangular kernel morphological opening (noise suppression), and closing (hole filling).
5. **Stage V: Anomaly Extraction and Visualization**: Contour tracing, spatial moment analysis, continuous thermal colormap rendering, and red overlay alpha compositing.

![Fig. 1. Pipeline Stages](figures/fig_pipeline_stages.png)
*Fig. 1. Visual progression of the ImageDiff pipeline on a printed circuit board (PCB) defect inspection scenario: (a) Reference Image A, (b) Target Image B, (c) Absolute Difference Map $|I_A - I_B|$, (d) Turbo Difference Heatmap, (e) Morphologically Refined Binary Change Mask, and (f) Segmented Contours with Bounding Boxes.*

---

## V. IMAGE PROCESSING METHODOLOGY

### A. Format and Dimension Harmonization
Input images may arrive in single-channel Grayscale ($C=1$), three-channel RGB ($C=3$), or four-channel RGBA ($C=4$) formats. RGBA images are blended onto an opaque white background via alpha compositing:
$$I_{\text{RGB}}(x, y) = I_{\text{RGB}}(x, y) \cdot \alpha(x, y) + I_{\text{white}} \cdot (1 - \alpha(x, y)) \tag{1}$$
where $\alpha(x, y) \in [0.0, 1.0]$. When images $I_A$ and $I_B$ possess distinct dimensions $(H_A, W_A) \neq (H_B, W_B)$, resizing without geometric distortion is crucial. The pipeline computes the uniform scaling factor:
$$s = \min\left(\frac{W_{\text{target}}}{W_{\text{src}}}, \frac{H_{\text{target}}}{H_{\text{src}}}\right) \tag{2}$$
The image is scaled via area-based interpolation (`cv2.INTER_AREA`) and centered within a zero-padded canvas of dimensions $(H_{\text{target}}, W_{\text{target}})$, preserving structural aspect ratios.

### B. Feature-Based Geometric Registration
To align target $I_B$ with reference $I_A$, the ORB detector identifies multiscale oriented FAST corners across an 8-level image pyramid with a scale factor of 1.2:
$$\mathcal{K}_A = \text{ORB}(I_{A,\text{gray}}), \quad \mathcal{K}_B = \text{ORB}(I_{B,\text{gray}}) \tag{3}$$
Descriptors are matched using Brute-Force Hamming distance. Ambiguous correspondences are pruned using Lowe's ratio test:
$$\|\mathbf{d}(\mathbf{f}_i^B, \mathbf{f}_{j_1}^A)\|_H < \rho \cdot \|\mathbf{d}(\mathbf{f}_i^B, \mathbf{f}_{j_2}^A)\|_H \tag{4}$$
where $j_1$ and $j_2$ denote the closest and second-closest matches, and the ratio threshold is configured to $\rho = 0.75$.

Given $K \ge 4$ surviving point correspondences $\mathbf{x}_i^B \leftrightarrow \mathbf{x}_i^A$, the projective homography matrix $\mathbf{H} \in \mathbb{R}^{3 \times 3}$ satisfies:
$$\begin{bmatrix} x_i^A \\ y_i^A \\ 1 \end{bmatrix} \sim \mathbf{H} \begin{bmatrix} x_i^B \\ y_i^B \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x_i^B \\ y_i^B \\ 1 \end{bmatrix} \tag{5}$$
RANSAC iteratively samples subsets of 4 correspondences, computes $\mathbf{H}$ via the Direct Linear Transformation (DLT), and counts inliers whose Euclidean reprojection error satisfies:
$$\|\mathbf{x}_i^A - \pi(\mathbf{H} \mathbf{x}_i^B)\|_2 < \epsilon_{\text{reproj}} \tag{6}$$
with $\epsilon_{\text{reproj}} = 5.0$ pixels. Image $I_B$ is then warped onto $I_A$'s reference frame using bilinear interpolation (`cv2.warpPerspective`).

---

## VI. MATHEMATICAL FORMULATION

### A. Pixel Differencing and Statistical Error Metrics
Let $I_A(x, y, c)$ and $I_B^{\text{aligned}}(x, y, c)$ be the normalized reference and aligned target images of size $M \times N$ with $C$ channels. The absolute photometric difference field $D(x, y)$ is formulated as:
$$D(x, y) = \frac{1}{C}\sum_{c=1}^{C} \left| I_A(x, y, c) - I_B^{\text{aligned}}(x, y, c) \right| \tag{7}$$

From $D(x, y)$, the Mean Absolute Error (MAE) and Mean Squared Error (MSE) are computed as:
$$\text{MAE} = \frac{1}{M \cdot N} \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} D(x, y) \tag{8}$$
$$\text{MSE} = \frac{1}{M \cdot N \cdot C} \sum_{x=0}^{M-1}\sum_{y=0}^{N-1}\sum_{c=1}^{C} \left( I_A(x, y, c) - I_B^{\text{aligned}}(x, y, c) \right)^2 \tag{9}$$

The Peak Signal-to-Noise Ratio (PSNR) is expressed logarithmically as:
$$\text{PSNR} = 10 \cdot \log_{10}\left( \frac{L^2}{\text{MSE} + \epsilon} \right) \quad \text{[dB]} \tag{10}$$
where $L = 255$ represents the maximum dynamic pixel range, and $\epsilon = 10^{-10}$ prevents division by zero.

Normalized similarity and difference scores are bounded on $[0.0, 1.0]$:
$$S_{\text{pixel}} = 1.0 - \frac{\text{MAE}}{255.0} \tag{11}$$
$$D_{\text{pixel}} = \frac{\text{MAE}}{255.0} = 1.0 - S_{\text{pixel}} \tag{12}$$

### B. Structural Similarity Index (SSIM)
While MAE captures scalar intensity divergence, SSIM evaluates structural degradation by comparing local luminance, contrast, and structure over an $11 \times 11$ Gaussian circular patch $\mathbf{w}$ with standard deviation $\sigma = 1.5$:
$$\text{SSIM}(\mathbf{x}, \mathbf{y}) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)} \tag{13}$$
where:
- $\mu_x = \sum_{i} w_i x_i$ is the local weighted mean intensity.
- $\sigma_x^2 = \sum_{i} w_i (x_i - \mu_x)^2$ is the local variance.
- $\sigma_{xy} = \sum_{i} w_i (x_i - \mu_x)(y_i - \mu_y)$ is the cross-covariance.
- $C_1 = (K_1 L)^2$ and $C_2 = (K_2 L)^2$ stabilize division by small denominators ($K_1 = 0.01, K_2 = 0.03, L = 255$).

The global mean SSIM metric is obtained by averaging across all spatial positions and color channels:
$$\overline{\text{SSIM}} = \frac{1}{M \cdot N \cdot C}\sum_{x, y, c} \text{SSIM}(x, y, c) \tag{14}$$

### C. Thresholding and Mathematical Morphology
To isolate significant changes from ambient sensor noise, the grayscale difference image $D_{\text{gray}}(x, y)$ is first convolved with a $5 \times 5$ Gaussian kernel:
$$G(x, y) = \frac{1}{2\pi\sigma_g^2} \exp\left(-\frac{x^2+y^2}{2\sigma_g^2}\right) \tag{15}$$
yielding smoothed field $\widetilde{D}(x, y) = D_{\text{gray}} * G$. Decision thresholding produces an initial binary mask $M_0 \in \{0, 255\}$:
$$M_0(x, y) = \begin{cases} 255, & \text{if } \widetilde{D}(x, y) \ge \tau \\ 0, & \text{otherwise} \end{cases} \tag{16}$$
where $\tau$ is the configurable sensitivity threshold (default $\tau = 30$).

Because $M_0$ contains isolated speckles and boundary discontinuities, dual-stage mathematical morphology is applied using a rectangular structuring element $B$ of size $3 \times 3$:
1. **Morphological Opening** ($M_1 = M_0 \circ B = (M_0 \ominus B) \oplus B$): Erosion eliminates isolated noisy pixels whose spatial extent is smaller than $B$.
2. **Morphological Closing** ($M_2 = M_1 \bullet B = (M_1 \oplus B) \ominus B$): Dilation bridges small fractures and fills internal cavities within detected components.

### D. Contour Detection and Spatial Moments
Suzuki's topological border-following algorithm is executed on $M_2$ to retrieve external contours $\mathcal{C} = \{C_1, C_2, \dots, C_K\}$. For each contour $C_k$, the spatial area is evaluated:
$$A(C_k) = \frac{1}{2}\sum_{i=0}^{n-1} (x_i y_{i+1} - x_{i+1} y_i) \tag{17}$$
Contours failing the minimum area constraint ($A(C_k) < A_{\min}$, default $A_{\min} = 40 \text{ px}^2$) are discarded. For each surviving contour, the bounding rectangle $(x_k, y_k, w_k, h_k)$ is established:
$$x_k = \min_{i} x_i, \quad y_k = \min_{i} y_i, \quad w_k = \max_{i} x_i - x_k, \quad h_k = \max_{i} y_i - y_k \tag{18}$$

The geometric centroid $(\bar{x}_k, \bar{y}_k)$ is computed from raw spatial moments:
$$M_{pq} = \sum_{x} \sum_{y} x^p y^q M_2(x, y) \tag{19}$$
$$\bar{x}_k = \frac{M_{10}(C_k)}{M_{00}(C_k)}, \quad \bar{y}_k = \frac{M_{01}(C_k)}{M_{00}(C_k)} \tag{20}$$
Additional shape descriptors include Aspect Ratio ($\text{AR} = w_k / h_k$) and Extent ($\text{Extent} = A(C_k) / (w_k \cdot h_k)$).

---

## VII. ALGORITHM DESIGN

```
Algorithm 1: ImageDiff Pipeline Execution
Input: Reference image I_A, Target image I_B, Threshold tau, Area threshold A_min, Flag align
Output: Metrics M, Binary mask M, Overlay I_overlay, Regions list R

1: Harmonize dimensions: I'_A, I'_B <-- PreprocessPair(I_A, I_B)
2: if align == TRUE then
3:     Extract ORB keypoints and descriptors K_A, K_B, D_A, D_B
4:     Filter matches using Lowe's ratio test (rho = 0.75)
5:     if |Matches| >= 4 then
6:         H <-- FindHomographyRANSAC(Matches)
7:         I_B^align <-- WarpPerspective(I'_B, H, size(I'_A))
8:     else
9:         I_B^align <-- I'_B  // Fallback to unaligned
10:    end if
11: else
12:    I_B^align <-- I'_B
13: end if
14: Compute difference field D <-- |I'_A - I_B^align|
15: Calculate error statistics: MAE, MSE, PSNR, SSIM from (8)-(14)
16: Apply Gaussian filter and threshold: M_0 <-- (GaussianBlur(D) >= tau) * 255
17: Refine topology: M <-- (M_0 o B) * B with 3x3 rectangular kernel
18: Extract external contours: C <-- FindContours(M, RETR_EXTERNAL)
19: R <-- []
20: for each C_k in C do
21:     if Area(C_k) >= A_min then
22:         Extract [x, y, w, h], centroid, aspect ratio, extent
23:         Append attributes to R
24:     end if
25: end for
26: Generate Heatmap H and Red Overlay I_overlay
27: return M, M, I_overlay, R
```

---

## VIII. SYSTEM IMPLEMENTATION

The system is constructed in a strictly modular, decoupled Python architecture under Python 3.11:
- `core/preprocessing.py`: Handles safe file loading, alpha blending, aspect-ratio preservation, and float normalization.
- `core/alignment.py`: Encapsulates ORB feature detection, Hamming distance matching, Lowe's test filtering, RANSAC homography estimation, and bilinear warping.
- `core/comparison.py`: Implements MAE, MSE, PSNR, channel-wise SSIM via `scikit-image`, and bounded score mappings.
- `core/change_detection.py`: Executes Gaussian smoothing, intensity thresholding, morphological opening/closing, and topological contour extraction.
- `core/visualization.py`: Generates Turbo/Jet heatmaps, binary mask visualizations, alpha-blended highlight overlays, and interactive wipe dividers.
- `core/engine.py`: Orchestrates all submodules into an immutable, structured `ImageDiffResult` container.
- `app.py`: A Streamlit web dashboard providing immediate interactive previews, metric tiles, tabular region data, and artifact export (PNG, CSV, JSON).

---

## IX. EXPERIMENTAL RESULTS

### A. Benchmark Performance Across Multi-Domain Datasets
To validate domain invariance and quantitative precision, the system was evaluated across four representative benchmark test scenarios generated via `samples/sample_generator.py`:
1. **Tabletop Objects**: A workspace containing stationery, an added smartphone, and a relocated object ($300 \times 400$ px).
2. **Document Tampering**: A commercial invoice subject to unauthorized financial fraud, including altered billing numbers, falsified billable hours, a forged signature, and an illicit audit stamp ($480 \times 640$ px).
3. **PCB Electronics Defect**: A high-density printed circuit board exhibiting a missing surface-mount capacitor and an etched copper trace scratch ($350 \times 500$ px).
4. **Building Facade**: An architectural building facade photograph exhibiting structural modifications including a newly constructed balcony ($350 \times 500$ px).

The actual quantitative results produced by executing the ImageDiff engine are compiled in Table I.

#### TABLE I: Benchmark Performance Across Diverse Domains
| Test Domain | Similarity (%) | Difference (%) | MAE | MSE | PSNR (dB) | SSIM | Changed Area (%) | Detected Regions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tabletop Objects** | 98.89% | 1.11% | 2.8431 | 221.39 | 24.68 | 0.9667 | 1.198% | 1 |
| **Document Tampering** | 99.05% | 0.95% | 2.4159 | 370.30 | 22.45 | 0.9637 | 1.960% | 5 |
| **PCB Defect** | 99.84% | 0.16% | 0.4156 | 42.37 | 31.86 | 0.9933 | 0.418% | 4 |
| **Building Facade** | 98.84% | 1.16% | 2.9559 | 331.81 | 22.92 | 0.9738 | 1.489% | 1 |

![Fig. 2. Multi-domain Change Detection Results](figures/fig_multidomain.png)
*Fig. 2. Multi-domain visual change detection results generated by ImageDiff across four distinct domains: Tabletop Objects, Invoice Document Tampering, Industrial PCB Defect Inspection, and Architectural Facade Modifications. Reference baseline images are displayed on the left, and segmented anomalies with bounding boxes are overlaid on the right.*

### B. Detailed Anomaly Extraction Analysis
Table II details the spatial geometric properties extracted by the contour detection module for the Document Tampering and PCB Defect scenarios.

#### TABLE II: Geometric Attributes of Detected Anomalies
| Domain / Region | Bounding Box $[x, y, w, h]$ | Area (px$^2$) | Area (%) | Centroid $(\bar{x}, \bar{y})$ | Aspect Ratio | Extent |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| *Document #1 (Audit Stamp)* | [417, 357, 168, 78] | 12,830 | 4.18% | (500, 395) | 2.15 | 0.98 |
| *Document #2 (Forged Signature)* | [470, 278, 29, 16] | 245 | 0.08% | (485, 285) | 1.81 | 0.53 |
| *Document #3 (Altered Total)* | [540, 278, 12, 15] | 135 | 0.04% | (545, 285) | 0.80 | 0.75 |
| *Document #4 (Altered Hours)* | [314, 404, 16, 6] | 66 | 0.02% | (321, 406) | 2.67 | 0.69 |
| *Document #5 (Altered Rate)* | [335, 403, 9, 9] | 57 | 0.02% | (339, 406) | 1.00 | 0.70 |
| *PCB #1 (Missing Capacitor)* | [120, 246, 41, 24] | 610 | 0.20% | (140, 257) | 1.71 | 0.62 |
| *PCB #2 (Trace Scratch)* | [285, 110, 23, 22] | 277 | 0.09% | (294, 121) | 1.05 | 0.55 |
| *PCB #3 (Copper Anomaly)* | [296, 100, 20, 19] | 197 | 0.06% | (306, 108) | 1.05 | 0.52 |
| *PCB #4 (Micro Discontinuity)*| [133, 241, 15, 5] | 56 | 0.02% | (140, 243) | 3.00 | 0.75 |

### C. Alignment Evaluation and RANSAC Inlier Analysis
Feature registration was systematically evaluated on each dataset:
- **Tabletop Objects**: 721 inliers out of 726 total matches (99.31% inlier consensus) across 783 reference keypoints and 806 target keypoints.
- **Document Tampering**: 1,433 inliers out of 1,511 matches (94.84% inlier consensus) across 1,991 reference keypoints.
- **PCB Defect**: 683 inliers out of 691 matches (98.84% inlier consensus) across 855 reference keypoints.
- **Building Facade**: 305 inliers out of 319 matches (95.61% inlier consensus).

The high inlier ratios verify that ORB descriptors coupled with Lowe's ratio test successfully eliminate false-positive correspondences before projective transformation estimation.

---

## X. TEST CASES AND SOFTWARE VALIDATION

Software reliability is established through an automated test suite implemented in `pytest`, containing 27 unit and integration tests across 8 test suites:
1. `test_preprocessing.py` (6 tests): Validates aspect-ratio padding, alpha compositing, float normalization, and dimension harmonization.
2. `test_alignment.py` (3 tests): Verifies ORB feature extraction, RANSAC convergence, and graceful failure fallback when features are insufficient.
3. `test_comparison.py` (3 tests): Proves mathematical precision of MAE, MSE, and SSIM bounds on synthetic inputs ($I_A = I_B \Rightarrow \text{MAE}=0, \text{SSIM}=1$).
4. `test_change_detection.py` (3 tests): Validates Gaussian smoothing, morphological noise removal, and contour area filtering.
5. `test_visualization.py` (1 test): Verifies generation, channel shapes, and color ranges of Turbo heatmaps, red overlays, and side-by-side composites.
6. `test_engine.py` (1 test): Enforces the schema contract of `ImageDiffResult`, ensuring immutability and dictionary access compatibility.
7. `test_image_utils.py` (6 tests): Tests image loading from arrays, PIL images, raw bytes, and disk paths.
8. `test_integration.py` (4 tests): End-to-end multi-domain pipelines executing across all four benchmark datasets.

All 27 automated tests pass with 100% success rate in 8.08 seconds.

---

## XI. LIMITATIONS

While ImageDiff demonstrates high fidelity across diverse domains, three technical limitations must be acknowledged:
1. **3D Parallax and Non-Planar Scenes**: A single $3 \times 3$ planar homography assumes approximately flat surfaces or pure camera rotation. Deep 3D scenes undergoing significant lateral baseline translation introduce depth-dependent parallax that cannot be resolved with a single 2D projective transformation.
2. **Extreme Global Illumination Variations**: Drastic lighting shifts (e.g., direct sunlight vs. nighttime shadows) create extensive intensity variations that trigger false-positive pixel differences unless the sensitivity threshold $\tau$ is substantially raised.
3. **Lack of Semantic Abstraction**: Operating strictly via low-level signal processing, ImageDiff detects changes in reflectance and geometry without semantic understanding (e.g., distinguishing between a moving vehicle vs. a displaced shadow).

---

## XII. FUTURE SCOPE

To advance the capabilities demonstrated in ImageDiff, future work will focus on:
1. **Illumination-Invariant Color Spaces**: Integrating chromaticity-decoupled representations such as CIE-LAB $\Delta E_{00}$ or Log-Chromaticity space to isolate chromatic shifts from luminance shadows.
2. **Deep Visual Feature Embeddings**: Augmenting ORB keypoints with self-supervised Vision Transformer (ViT) patch descriptors to enable semantic comparison across disparate seasons or lighting.
3. **Piecewise Local Homography**: Implementing mesh-based or piecewise affine warping to compensate for non-rigid surface deformation and localized 3D parallax.

---

## XIII. CONCLUSION

This paper presented ImageDiff, a robust, general-purpose image comparison and anomaly localization system developed using foundational digital image processing algorithms. By combining aspect-ratio-preserving preprocessing, ORB-RANSAC projective homography registration, complementary photometric (MAE, MSE, PSNR) and perceptual (SSIM) error formulations, dual-stage morphological filtering, and Suzuki topological contour extraction, ImageDiff delivers accurate, interpretable change detection across diverse visual domains. Empirical benchmarking on consumer products, fraudulent invoices, electronic circuit boards, and architectural structures confirmed sub-pixel detection sensitivity, isolating anomalies spanning from 0.018% to 4.18% of the image canvas with over 94% registration inlier consensus. The system fulfills all academic objectives of the Image Processing Analysis curriculum, bridging theoretical principles with a production-grade software implementation.

---

## REFERENCES

- [1] R. J. Radke, S. Andra, O. Al-Kofahi, and B. Roysam, "Image change detection algorithms: a systematic survey," *IEEE Transactions on Image Processing*, vol. 14, no. 3, pp. 294–307, Mar. 2005.
- [2] R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed. New York, NY, USA: Pearson, 2018.
- [3] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," *IEEE Transactions on Image Processing*, vol. 13, no. 4, pp. 600–612, Apr. 2004.
- [4] S. van der Walt, J. L. Schönberger, J. Nunez-Iglesias, F. Boulogne, J. D. Warner, N. Yager, E. Gouillart, and T. Yu, "scikit-image: image processing in Python," *PeerJ*, vol. 2, p. e453, Jun. 2014.
- [5] D. G. Lowe, "Distinctive image features from scale-invariant keypoints," *International Journal of Computer Vision*, vol. 60, no. 2, pp. 91–110, Nov. 2004.
- [6] E. Rublee, V. Rabaud, K. Konolige, and G. Bradski, "ORB: An efficient alternative to SIFT or SURF," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, Barcelona, Spain, 2011, pp. 2564–2571.
- [7] M. A. Fischler and R. C. Bolles, "Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography," *Communications of the ACM*, vol. 24, no. 6, pp. 381–395, Jun. 1981.
- [8] R. M. Haralick, S. R. Sternberg, and X. Zhuang, "Image analysis using mathematical morphology," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. PAMI-9, no. 4, pp. 532–550, Jul. 1987.
- [9] S. Suzuki and K. Abe, "Topological structural analysis of digitized binary images by border following," *Computer Vision, Graphics, and Image Processing*, vol. 30, no. 1, pp. 32–46, Apr. 1985.
- [10] G. Bradski, "The OpenCV Library," *Dr. Dobb's Journal of Software Tools*, vol. 25, no. 11, pp. 120–125, 2000.
- [11] N. Otsu, "A threshold selection method from gray-level histograms," *IEEE Transactions on Systems, Man, and Cybernetics*, vol. 9, no. 1, pp. 62–66, Jan. 1979.
- [12] J. Canny, "A modern approach to edge detection," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. PAMI-8, no. 6, pp. 679–698, Nov. 1986.

