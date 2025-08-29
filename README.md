# PyHE - Python Histogram Equalization Library

A comprehensive Python library for histogram equalization techniques, great for messing around with medical imaging applications like CT and MRI grayscale images. This is a personal practice project, so have fun with it!

## Features

PyHE implements multiple advanced histogram equalization methods:

-   **Standard Histogram Equalization (HE)** - The classic global histogram equalization.
-   **Brightness Preserving Bi-Histogram Equalization (BBHE)** - Keeps the mean brightness the same while enhancing contrast.
-   **Recursive Mean Separated Histogram Equalization (RMSHE)** - A recursive approach for better detail preservation.
-   **Dynamic Histogram Equalization (DHE)** - Plateau-limited enhancement that you can control.
-   **Contrast Limited Adaptive Histogram Equalization (CLAHE)** - Helps prevent over-amplification of noise.
-   **Adaptive CLAHE** - Automatically adjusts parameters based on image characteristics.
-   **Medical Enhancement** - A specialized enhancement for medical images that preserves the original intensity range.

## Installation

1.  Clone or download the repository.
2.  Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

The core library requires the following:

-   numpy >= 1.19.0
-   opencv-python >= 4.5.0
-   scipy >= 1.7.0
-   matplotlib >= 3.3.0 (for visualization)
-   scikit-image >= 0.18.0 (optional)
-   Pillow >= 8.0.0 (optional)

For the 3D segmentation case study, you'll also need:

-   MONAI
-   PyTorch
-   NiBabel (for NIfTI file handling)

## Quick Start

```python
import numpy as np
from PyHE import PyHE
import cv2

# Load your medical image
image = cv2.imread('medical_image.png', cv2.IMREAD_GRAYSCALE)

# Apply different enhancement methods
he_result = PyHE.histogram_equalization(image)
bbhe_result = PyHE.bbhe(image)
clahe_result = PyHE.clahe(image)
medical_result = PyHE.medical_enhancement(image, method='clahe')

# Compare all methods
results = PyHE.compare_methods(image)

# Calculate quality metrics
metrics = PyHE.calculate_metrics(image, clahe_result)
print(f"PSNR: {metrics['psnr']:.2f} dB")
print(f"SSIM: {metrics['ssim']:.4f}")
```

## Detailed Usage

### Standard Histogram Equalization

```python
# Basic histogram equalization
enhanced = PyHE.histogram_equalization(image)
```

### Brightness Preserving Bi-Histogram Equalization (BBHE)

```python
# BBHE preserves the mean brightness of the original image
enhanced = PyHE.bbhe(image)
```

### Recursive Mean Separated Histogram Equalization (RMSHE)

```python
# RMSHE with different recursion levels
enhanced_level2 = PyHE.rmshe(image, recursion_level=2)
enhanced_level3 = PyHE.rmshe(image, recursion_level=3)
```

### Dynamic Histogram Equalization (DHE)

```python
# DHE with plateau limit parameter
enhanced = PyHE.dhe(image, x=0.5)  # x should be between 0 and 1
```

### Contrast Limited Adaptive Histogram Equalization (CLAHE)

```python
# Standard CLAHE
enhanced = PyHE.clahe(image, clip_limit=2.0, tile_grid_size=(8, 8))

# Adaptive CLAHE for medical images
enhanced = PyHE.adaptive_clahe(image, window_size=64, clip_limit=2.0)
```

### Medical Image Enhancement

```python
# Specialized enhancement for medical images
enhanced = PyHE.medical_enhancement(
    image,
    method='clahe',  # 'clahe', 'bbhe', 'dhe', 'rmshe'
    preserve_range=True  # Preserve original intensity range
)
```

### Comprehensive Analysis

```python
# Compare all methods
results = PyHE.compare_methods(image)

# Calculate metrics for each method
for method_name, enhanced_image in results.items():
    if method_name != 'original':
        metrics = PyHE.calculate_metrics(image, enhanced_image)
        print(f"{method_name}: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")
```

## API Reference

### PyHE Class Methods

#### `histogram_equalization(image)`

Standard histogram equalization.

**Parameters:**

-   `image` (np.ndarray): Input grayscale image

**Returns:**

-   `np.ndarray`: Enhanced image

#### `bbhe(image)`

Brightness Preserving Bi-Histogram Equalization.

#### `rmshe(image, recursion_level=2)`

Recursive Mean Separated Histogram Equalization.

#### `dhe(image, x=0.5)`

Dynamic Histogram Equalization.

#### `clahe(image, clip_limit=2.0, tile_grid_size=(8, 8))`

Contrast Limited Adaptive Histogram Equalization.

#### `adaptive_clahe(image, window_size=64, clip_limit=2.0)`

Adaptive CLAHE with automatic parameter adjustment.

#### `medical_enhancement(image, method='clahe', preserve_range=True)`

Medical image specific enhancement.

#### `compare_methods(image)`

Compare all enhancement methods on the same image.

#### `calculate_metrics(original, enhanced)`

Calculate quality metrics for enhanced images.


## Case Study: 3D Liver Segmentation

To see PyHE in action, I used it to see how 3D histogram equalization (specifically, 3D BBHE) would affect liver segmentation performance with a few deep learning models.

### Project Overview

The study compares four 3D segmentation models on liver CT data, with and without histogram equalization preprocessing:

-   **UNet**: Classic encoder-decoder architecture
-   **SegResNet**: Residual network-based segmentation
-   **SwinUNETR**: A transformer-based approach
-   **VNet**: Volumetric convolutional network

### Key Findings & Unexpected Results

Surprisingly, histogram equalization did **not** consistently improve segmentation performance. Actually, it often made it worse.

| Model     | Original Dice   | Enhanced Dice   | Improvement |
| --------- | --------------- | --------------- | ----------- |
| UNet      | 0.8112 ± 0.1763 | 0.8254 ± 0.1293 | **+1.75%**  |
| SegResNet | 0.8804 ± 0.0496 | 0.8775 ± 0.0567 | **-0.32%**  |
| SwinUNETR | 0.8989 ± 0.0313 | 0.8836 ± 0.0613 | **-1.71%**  |
| VNet      | 0.5748 ± 0.1179 | 0.4195 ± 0.2294 | **-27.02%** |

**Overall Average**: The performance was **-6.83% ± 11.72%** worse with enhancement.

### Analysis

1.  **Only UNet showed a small improvement** (+1.75%).
2.  **SegResNet and SwinUNETR showed slight degradation**.
3.  **VNet's performance completely tanked** (-27.02%).
4.  The high variance in results suggests that each model is sensitive to this kind of preprocessing in its own way.

### Dataset

-   **Source**: [3D Liver segmentation Dataset on Kaggle](https://www.kaggle.com/datasets/prathamgrover/3d-liver-segmentation/data)
-   **Classes**: Background vs. Liver
-   **Total Samples**: 123 volumes (split into 80 training, 18 validation, 25 test)

### Project Structure

```
liver_segmentation_project/
├── src/
│   ├── PyHE.py
│   ├── comprehensive_segmentation_pipeline.py
│   └── ...
├── results/
│   ├── histogram_equalization_enhancement/
│	│	├── liver_0_comparison.png
│	│	├── ...
│	│	└── README.md
│   └── segmentation/
│	│	├── test_results.json
│	│	└── ...
└── requirements.txt
└── README.md
```

### Usage (for the Case Study)

```bash
# To run the comprehensive multi-model training pipeline
python src/comprehensive_segmentation_pipeline.py

# To run training sequentially (more memory friendly)
python src/sequential_training_pipeline.py --model UNet --dataset original
```

### Conclusions from the Case Study

1.  **Histogram equalization isn't a silver bullet** for medical image segmentation.
2.  **Model architecture really matters**. Different models react differently to the same preprocessing.
3.  **Advanced models** like SwinUNETR and SegResNet might already be good at handling contrast variations, making HE less useful or even harmful.
4.  **Always test preprocessing** on your specific model and dataset before assuming it will help.

## License

This project is open source. Check the LICENSE file for details.

## Acknowledgments

- This project utilized a dataset following **CC BY-SA 4.0** (Creative Commons Attribution-ShareAlike 4.0 International Public License, https://creativecommons.org/licenses/by-sa/4.0/) by Pratham Grover for training.
- Please note that the comparison pictures in results/histogram_equalization_enhancement follows CC BY-SA 4.0 LISENCE.
