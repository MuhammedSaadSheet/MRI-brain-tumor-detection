#==============================
#   Brain Tumor MRI Image Processing
#==============================

#==============================
# Author: Muhammed Saad
#==============================

#==============================
# Description:
# This script processes MRI images of brain tumors.
# It applies various filters, preprocessing techniques, and segmentation methods to analyze the images.
# The results are visualized using matplotlib.
#==============================

#=============================
# Importing Required Libraries
#=============================

import os
import cv2
import kagglehub
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

#=============================
#   Constants
#=============================

DATASET_NAME = "masoudnickparvar/brain-tumor-mri-dataset"
NUM_IMAGES = 5
SELECTED_IMAGE_INDEX = 0
GAMMA = 1.5

LOWER_THRESHOLD = 100
UPPER_THRESHOLD = 200


#=============================
#   Functions
#=============================

def load_dataset(dataset_name):
    return kagglehub.dataset_download(dataset_name)


def collect_image_paths(dataset_path):
    image_paths = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, file))

    if len(image_paths) == 0:
        raise Exception("no images found in the dataset")

    image_paths.sort()
    return image_paths


def load_images(image_paths, num_images):
    images = []

    for path in image_paths[:num_images]:
        img = cv2.imread(path)

        if img is not None:
            images.append(img)

    if len(images) == 0:
        raise Exception("no images could be loaded from the dataset")

    return images


def convert_to_gray(images):
    return [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]


def apply_filters(gray_images):
    return {
        "Original Gray": gray_images,
        "Mean": [cv2.blur(img, (5, 5)) for img in gray_images],
        "Gaussian": [cv2.GaussianBlur(img, (5, 5), 0) for img in gray_images],
        "Median": [cv2.medianBlur(img, 5) for img in gray_images],
        "Bilateral": [cv2.bilateralFilter(img, 9, 75, 75) for img in gray_images],
        "Non-Local Means": [
            cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
            for img in gray_images
        ]
    }


def apply_preprocessing(filtered_images, image_index, gamma):
    first_filter = next(iter(filtered_images.values()))

    if image_index < 0 or image_index >= len(first_filter):
        raise IndexError("SELECTED_IMAGE_INDEX is out of bounds")

    selected_images = {
        filter_name: images[image_index]
        for filter_name, images in filtered_images.items()
    }

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    results = {}

    for filter_name, img in selected_images.items():
        clahe_img = clahe.apply(img)
        equalized_img = cv2.equalizeHist(img)

        gamma_img = cv2.pow(img / 255.0, gamma) * 255.0
        gamma_img = gamma_img.astype(np.uint8)

        stretched_img = cv2.normalize(
            img,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX
        )

        results[filter_name] = {
            "Filtered": img,
            "CLAHE": clahe_img,
            "Histogram Equalization": equalized_img,
            "Gamma Correction": gamma_img,
            "Contrast Stretching": stretched_img
        }

    return results


def calculate_histograms(preprocessing_results):
    histograms = {}

    for filter_name, methods in preprocessing_results.items():
        histograms[filter_name] = {}

        for method_name, img in methods.items():
            histograms[filter_name][method_name] = cv2.calcHist(
                [img],
                [0],
                None,
                [256],
                [0, 256]
            )

    return histograms


def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    return mask


def segment_manual_threshold(img, lower_threshold, upper_threshold):
    mask = np.where(
        (img >= lower_threshold) & (img <= upper_threshold),
        255,
        0
    ).astype(np.uint8)

    return clean_mask(mask)


def segment_region_growing(img, threshold=40):
    h, w = img.shape

    seed_x = w // 2
    seed_y = int(h * 0.4)

    seed_value = int(img[seed_y, seed_x])

    mask = np.zeros((h, w), dtype=np.uint8)
    visited = np.zeros((h, w), dtype=bool)

    queue = deque()
    queue.append((seed_y, seed_x))

    while queue:
        y, x = queue.popleft()

        if y < 0 or y >= h or x < 0 or x >= w:
            continue

        if visited[y, x]:
            continue

        visited[y, x] = True

        if abs(int(img[y, x]) - seed_value) <= threshold:
            mask[y, x] = 255

            queue.append((y + 1, x))
            queue.append((y - 1, x))
            queue.append((y, x + 1))
            queue.append((y, x - 1))

    return clean_mask(mask)


def show_filter_grid(filtered_images):
    for filter_name, images in filtered_images.items():
        plt.figure(figsize=(15, 4))

        for col, img in enumerate(images):
            plt.subplot(1, len(images), col + 1)
            plt.imshow(img, cmap="gray")
            plt.title(f"Bild {col + 1}", fontsize=8)
            plt.axis("off")

        plt.suptitle(f"Filter Vergleich: {filter_name}")
        plt.tight_layout()
        plt.show()


def show_preprocessing_grid(preprocessing_results):
    for filter_name, methods in preprocessing_results.items():
        method_names = list(methods.keys())

        plt.figure(figsize=(18, 4))

        for col, method_name in enumerate(method_names):
            plt.subplot(1, len(method_names), col + 1)
            plt.imshow(methods[method_name], cmap="gray")
            plt.title(method_name, fontsize=8)
            plt.axis("off")

        plt.suptitle(f"Grauwert-Bearbeitung: {filter_name}")
        plt.tight_layout()
        plt.show()


def show_histograms(histograms):
    method_names = list(next(iter(histograms.values())).keys())

    for method_name in method_names:
        plt.figure(figsize=(10, 6))

        for filter_name in histograms:
            plt.plot(
                histograms[filter_name][method_name].flatten(),
                label=f"{filter_name} + {method_name}"
            )

        plt.title(f"Histogramm Vergleich: {method_name}")
        plt.xlabel("Intensität")
        plt.ylabel("Pixelanzahl")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.show()


def show_segmentation_on_original(original_image):
    segmentation_results = {
        "Manual Thresholding": segment_manual_threshold(
            original_image,
            LOWER_THRESHOLD,
            UPPER_THRESHOLD
        ),
        "Region Growing": segment_region_growing(original_image)
    }

    plt.figure(figsize=(10, 4))

    for i, (name, mask) in enumerate(segmentation_results.items(), start=1):
        plt.subplot(1, 2, i)
        plt.imshow(mask, cmap="gray")
        plt.title(name, fontsize=8)
        plt.axis("off")

    plt.suptitle("Segmentierung auf dem ursprünglichen Grauwertbild")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))

    for i, (name, mask) in enumerate(segmentation_results.items(), start=1):
        overlay = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        plt.subplot(1, 2, i)
        plt.imshow(overlay)
        plt.title(name, fontsize=8)
        plt.axis("off")

    plt.suptitle("Overlay auf dem ursprünglichen Grauwertbild")
    plt.tight_layout()
    plt.show()

#=============================
#   Main Execution
#=============================

dataset_path = load_dataset(DATASET_NAME)
image_paths = collect_image_paths(dataset_path)

images = load_images(image_paths, NUM_IMAGES)
gray_images = convert_to_gray(images)

filtered_images = apply_filters(gray_images)
show_filter_grid(filtered_images)

preprocessing_results = apply_preprocessing(
    filtered_images,
    SELECTED_IMAGE_INDEX,
    GAMMA
)

show_preprocessing_grid(preprocessing_results)

histograms = calculate_histograms(preprocessing_results)
show_histograms(histograms)

original_selected_image = gray_images[SELECTED_IMAGE_INDEX]
show_segmentation_on_original(original_selected_image)