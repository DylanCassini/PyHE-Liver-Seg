# 3D Liver Segmentation with Histogram Equalization Enhancement

This project investigates the impact of 3D histogram equalization (BBHE - Bi-Histogram Equalization) on liver segmentation performance using deep learning models.

## Project Overview

The study compares four state-of-the-art 3D segmentation models on liver CT data with and without histogram equalization preprocessing:
- **UNet**: Classic encoder-decoder architecture
- **SegResNet**: Residual network-based segmentation
- **SwinUNETR**: Vision transformer-based approach
- **VNet**: Volumetric convolutional network

## Key Findings

### Unexpected Results

Contrary to expectations, histogram equalization did **not** consistently improve segmentation performance:

| Model | Original Dice | Enhanced Dice | Improvement |
|-------|---------------|---------------|-------------|
| UNet | 0.8112 ± 0.1763 | 0.8254 ± 0.1293 | **+1.75%** |
| SegResNet | 0.8804 ± 0.0496 | 0.8775 ± 0.0567 | **-0.32%** |
| SwinUNETR | 0.8989 ± 0.0313 | 0.8836 ± 0.0613 | **-1.71%** |
| VNet | 0.5748 ± 0.1179 | 0.4195 ± 0.2294 | **-27.02%** |

**Overall Average**: -6.83% ± 11.72% (worse with enhancement)

### Analysis

1. **Only UNet showed improvement** (+1.75%) with histogram equalization
2. **SegResNet and SwinUNETR showed slight degradation** (-0.32% and -1.71%)
3. **VNet showed significant performance drop** (-27.02%)
4. **High variance in results** suggests model-dependent sensitivity to preprocessing

### Histogram Equalization Effects

The 3D BBHE enhancement did improve image contrast:
- **Contrast Enhancement**: 4.372x improvement over 2D methods
- **Artifact Measure**: Slightly higher artifacts (4.699 vs 3.619)
- **Consistency**: Better slice-to-slice consistency

## Dataset

- **Source**: 3D Liver segmentation Dataset: [3D Liver segmentation](https://www.kaggle.com/datasets/prathamgrover/3d-liver-segmentation/data)
- **Classes**: Segmentation (background, liver)
- **Total Samples**: 123 volumes
- **Split**: 80 training, 18 validation, 25 test

## Project Structure

```
liver_segmentation_project/
├── src/
│   ├── PyHE.py                              # Histogram equalization implementation
│   ├── comprehensive_segmentation_pipeline.py # Multi-model training pipeline
│   ├── sequential_training_pipeline.py      # Sequential model training
│   └── simple_segmentation_test.py         # Basic segmentation test
├── results/
│   ├── comprehensive_report.txt            # Detailed experimental results
│   ├── test_results.json                  # Raw test metrics
│   ├── training_results.json              # Training progress data
│   ├── model_comparison.png               # Performance comparison chart
│   ├── training_progress.png              # Training curves
│   ├── enhancement_comparison_summary.png  # Enhancement visualization
│   └── comparison_summary.txt             # Enhancement analysis
├── docs/
└── requirements.txt                        # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

### Key Dependencies
- MONAI (Medical Open Network for AI)
- PyTorch
- NumPy
- Matplotlib
- NiBabel (for NIfTI file handling)

## Usage

### Basic Histogram Equalization
```python
from src.PyHE import PyHE

# Apply 3D BBHE enhancement
enhancer = PyHE()
enhanced_volume = enhancer.enhance_3d(original_volume)
```

### Run Segmentation Pipeline
```bash
# Comprehensive multi-model training
python src/comprehensive_segmentation_pipeline.py

# Sequential training (memory efficient)
python src/sequential_training_pipeline.py --model UNet --dataset original

# Simple test
python src/simple_segmentation_test.py
```

## Conclusions

### Main Insights

1. **Histogram equalization is not universally beneficial** for medical image segmentation
2. **Model architecture matters**: Different models respond differently to preprocessing
3. **UNet showed resilience** to enhancement, while VNet was highly sensitive
4. **Advanced models** (SwinUNETR, SegResNet) may already handle contrast variations effectively

### Recommendations

1. **Test preprocessing effects** on specific model architectures before deployment
2. **Consider model-specific preprocessing** rather than universal enhancement
3. **Evaluate on multiple metrics** beyond Dice score for comprehensive assessment
4. **Use validation sets** to tune preprocessing parameters per model

## Contact

For questions about the implementation or results, please refer to the source code and experimental reports in the `results/` directory.