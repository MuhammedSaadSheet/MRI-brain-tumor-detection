# Brain Tumor MRI Segmentation

A Python-based medical image processing project that explores and compares multiple filtering, enhancement, and segmentation techniques for identifying potential brain tumor regions in MRI scans.

## Project Overview

This project focuses on the analysis of brain MRI images using classical computer vision and image processing methods. The objective is to evaluate how different preprocessing and segmentation techniques influence the detection of tumor-like regions.

The workflow includes image acquisition, grayscale conversion, noise reduction, contrast enhancement, histogram analysis, and segmentation. Multiple methods are implemented and compared to assess their effectiveness in highlighting potential tumor areas.

## Features

### Image Filtering

* Mean Filter
* Gaussian Filter
* Median Filter
* Bilateral Filter
* Non-Local Means Denoising

### Image Enhancement

* CLAHE (Contrast Limited Adaptive Histogram Equalization)
* Histogram Equalization
* Gamma Correction
* Contrast Stretching

### Segmentation Methods

* Otsu Thresholding
* Region Growing
* Region Merging
* Split and Merge
* Canny Edge Detection

### Analysis

* Histogram generation and comparison
* Visualization of preprocessing results
* Comparison of segmentation outputs
* Contour overlay visualization

## Dataset

This project uses the Brain Tumor MRI Dataset from Kaggle:

```text
masoudnickparvar/brain-tumor-mri-dataset
```

The dataset is downloaded automatically using KaggleHub.

## Installation

Install the required dependencies:

```bash
pip install opencv-python kagglehub numpy matplotlib
```

## Usage

Run the project:

```bash
python main.py
```

## Configuration

The following parameters can be modified:

```python
DATASET_NAME = "masoudnickparvar/brain-tumor-mri-dataset"
NUM_IMAGES = 5
SELECTED_IMAGE_INDEX = 0
GAMMA = 1.5
```

## Processing Pipeline

1. Download MRI dataset
2. Load MRI images
3. Convert images to grayscale
4. Apply image filtering techniques
5. Enhance image contrast
6. Generate intensity histograms
7. Apply segmentation algorithms
8. Visualize and compare results

## Technologies

* Python
* OpenCV
* NumPy
* Matplotlib
* KaggleHub

## Future Improvements

* Quantitative evaluation metrics
* Machine learning-based segmentation
* Deep learning approaches (CNN/U-Net)
* Automatic tumor classification
* Interactive user interface

## Disclaimer

This project is intended for educational and research purposes only. It is not a medical diagnostic tool and must not be used for clinical decision-making.

## Author

Muhammed Saad