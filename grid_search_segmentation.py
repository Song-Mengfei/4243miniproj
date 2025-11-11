"""
Grid Search for Segmentation Hyperparameters

This script performs grid search on spatial_weight_x, spatial_weight_y, and contour_weight
parameters for the segment_characters function. It evaluates performance using a trained
character recognition model and reports both sequence-level and character-level accuracy.

Usage:
    python grid_search_segmentation.py
    python grid_search_segmentation.py --model best_char_recognition_model.pth
    python grid_search_segmentation.py --data processed/test --output grid_search_results.csv
"""

import argparse
import csv
import time
from pathlib import Path
from itertools import product
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from segmentation_v2 import get_dominant_clusters

# =============================================================================
# Model Architecture (must match training)
# =============================================================================

class InvertedResidualBlock(nn.Module):
    """MobileNetV2-style block with depthwise separable convolutions."""
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=4):
        super().__init__()
        hidden_ch = in_ch * expand_ratio
        self.use_residual = (stride == 1 and in_ch == out_ch)

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, 3, stride=stride, padding=1, 
                     groups=hidden_ch, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class EfficientCaptchaCNN(nn.Module):
    def __init__(self, num_classes=36, dropout=0.5):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.features = nn.Sequential(
            InvertedResidualBlock(32, 32, stride=1, expand_ratio=1),
            InvertedResidualBlock(32, 64, stride=2, expand_ratio=4),
            InvertedResidualBlock(64, 64, stride=1, expand_ratio=4),
            InvertedResidualBlock(64, 128, stride=2, expand_ratio=4),
            InvertedResidualBlock(128, 128, stride=1, expand_ratio=4),
            InvertedResidualBlock(128, 256, stride=2, expand_ratio=4),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# =============================================================================
# Segmentation Function (with configurable parameters)
# =============================================================================

def segment_characters_custom(image_path, min_pixels=3, top_components=5,
                               spatial_weight_x=1.5, spatial_weight_y=0.4, 
                               contour_weight=100):
    """
    Segment CAPTCHA characters using configurable hyperparameters.
    Returns list of character images.
    """
    filename = Path(image_path).stem
    chars_part = filename.split("-")[0]
    n_clusters = len(chars_part) + 1

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    threshold = 250
    max_threshold = 250
    step = 10
    valid_bounding_boxes = []

    while threshold <= max_threshold:
        clustered_masks, mask_foreground = get_dominant_clusters(
            image, n_clusters, threshold,
            spatial_weight_x=spatial_weight_x, 
            spatial_weight_y=spatial_weight_y, 
            contour_weight=contour_weight
        )
        if clustered_masks is None:
            threshold += step
            continue

        bounding_boxes = []
        for idx, mask in enumerate(clustered_masks):
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue
            x_min, x_max = np.min(xs), np.max(xs)
            y_min, y_max = np.min(ys), np.max(ys)
            w, h = x_max - x_min + 1, y_max - y_min + 1
            bounding_boxes.append((x_min, y_min, w, h, mask))

        if len(bounding_boxes) >= n_clusters:
            valid_bounding_boxes = bounding_boxes
            break
        threshold += step

    if not valid_bounding_boxes:
        return []

    cleaned_clusters = []
    cluster_x_positions = []
    cluster_pixel_counts = []

    # Clean clusters
    for i, (x, y, w, h, mask) in enumerate(valid_bounding_boxes):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            sorted_idx = np.argsort(areas)[::-1]

            keep_mask = np.zeros_like(mask)
            count_kept = 0
            first_center = None
            distance_threshold = 25

            for idx2 in sorted_idx:
                if count_kept >= top_components:
                    break
                area = areas[idx2]
                if area < min_pixels:
                    continue

                label = idx2 + 1
                cx, cy = centroids[label]

                if first_center is None:
                    first_center = (cx, cy)
                    keep_mask[labels == label] = 255
                    count_kept += 1
                else:
                    dist = np.sqrt((cx - first_center[0]) ** 2 + (cy - first_center[1]) ** 2)
                    if dist < distance_threshold:
                        keep_mask[labels == label] = 255
                        count_kept += 1

            cleaned_mask = keep_mask
        else:
            cleaned_mask = mask.copy()

        ys, xs = np.where(cleaned_mask > 0)
        x_leftmost = np.min(xs) if len(xs) > 0 else 99999
        pixel_count = np.sum(cleaned_mask > 0)

        cluster_pixel_counts.append(pixel_count)
        cleaned_clusters.append((x, y, w, h, cleaned_mask))
        cluster_x_positions.append(x_leftmost)

    # Identify and skip smallest cluster (noise)
    min_idx = int(np.argmin(cluster_pixel_counts))
    sorted_indices = np.argsort(cluster_x_positions)

    char_images = []
    for i in sorted_indices:
        if i == min_idx:
            continue

        x, y, w, h, cleaned_mask = cleaned_clusters[i]
        roi = image[y:y + h, x:x + w]
        mask_roi = cleaned_mask[y:y + h, x:x + w]
        char_img = np.full_like(roi, 255)
        char_img[mask_roi == 255] = roi[mask_roi == 255]

        # Tight crop
        gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        ys_fg, xs_fg = np.where(gray < 250)
        if len(xs_fg) > 0 and len(ys_fg) > 0:
            x_min_fg, x_max_fg = np.min(xs_fg), np.max(xs_fg)
            y_min_fg, y_max_fg = np.min(ys_fg), np.max(ys_fg)
            char_img = char_img[y_min_fg:y_max_fg + 1, x_min_fg:x_max_fg + 1]

        char_images.append(char_img)

    return char_images


# =============================================================================
# Recognition Functions
# =============================================================================

CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"
id2ch = {i: ch for i, ch in enumerate(CHARS)}


def preprocess_char(char_img, target_size=(32, 32)):
    """Preprocess a single character image for model input."""
    if len(char_img.shape) == 3:
        char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    
    h, w = char_img.shape
    target_h, target_w = target_size
    
    if h <= 0 or w <= 0:
        return np.ones((target_h, target_w), dtype=np.uint8) * 255
    
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    new_h = max(1, new_h)
    new_w = max(1, new_w)
    
    char_img = cv2.resize(char_img, (new_w, new_h))
    
    canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = char_img
    
    return canvas


def recognize_captcha(model, image_path, device, 
                      spatial_weight_x, spatial_weight_y, contour_weight):
    """
    Recognize a full CAPTCHA using specified segmentation parameters.
    """
    model.eval()
    
    # Segment with custom parameters
    char_images = segment_characters_custom(
        image_path,
        spatial_weight_x=spatial_weight_x,
        spatial_weight_y=spatial_weight_y,
        contour_weight=contour_weight
    )
    
    if not char_images:
        return ""
    
    predictions = []
    
    with torch.no_grad():
        for char_img in char_images:
            # Preprocess
            canvas = preprocess_char(char_img)
            
            # Convert to tensor
            char_tensor = torch.from_numpy(canvas).float() / 255.0
            char_tensor = char_tensor.unsqueeze(0).repeat(3, 1, 1)
            char_tensor = char_tensor.unsqueeze(0).to(device)
            
            # Predict
            output = model(char_tensor)
            _, predicted = output.max(1)
            pred_char = id2ch[predicted.item()]
            predictions.append(pred_char)
    
    return ''.join(predictions)


def evaluate_with_params(model, img_paths, device, 
                         spatial_weight_x, spatial_weight_y, contour_weight,
                         verbose=False):
    """
    Evaluate model on dataset with specified segmentation parameters.
    """
    model.eval()
    
    total = 0
    correct = 0
    char_total = 0
    char_correct = 0
    
    iterator = tqdm(img_paths, desc="Evaluating") if verbose else img_paths
    
    for img_path in iterator:
        try:
            filename = Path(img_path).stem
            gt_text = filename.split('-')[0].lower()
            
            pred_text = recognize_captcha(
                model, img_path, device,
                spatial_weight_x, spatial_weight_y, contour_weight
            )
            pred_text = pred_text.lower()
            
            total += 1
            if gt_text == pred_text:
                correct += 1
            
            # Character-level accuracy
            min_len = min(len(gt_text), len(pred_text))
            for i in range(min_len):
                char_total += 1
                if gt_text[i] == pred_text[i]:
                    char_correct += 1
                    
        except Exception as e:
            if verbose:
                print(f"Error processing {img_path}: {e}")
            continue
    
    seq_acc = 100 * correct / total if total > 0 else 0
    char_acc = 100 * char_correct / char_total if char_total > 0 else 0
    
    return seq_acc, char_acc, total


# =============================================================================
# Grid Search
# =============================================================================

def grid_search(model, img_paths, device, param_grid, output_csv):
    """
    Perform grid search over segmentation parameters.
    """
    results = []
    
    # Generate all parameter combinations
    param_combinations = list(product(
        param_grid['spatial_weight_x'],
        param_grid['spatial_weight_y'],
        param_grid['contour_weight']
    ))
    
    print(f"\n{'='*70}")
    print(f"Starting Grid Search")
    print(f"{'='*70}")
    print(f"Total combinations: {len(param_combinations)}")
    print(f"Test images: {len(img_paths)}")
    print(f"\nParameter ranges:")
    print(f"  spatial_weight_x: {param_grid['spatial_weight_x']}")
    print(f"  spatial_weight_y: {param_grid['spatial_weight_y']}")
    print(f"  contour_weight: {param_grid['contour_weight']}")
    print(f"{'='*70}\n")
    
    best_seq_acc = 0
    best_params = None
    
    for idx, (swx, swy, cw) in enumerate(param_combinations, 1):
        print(f"\n[{idx}/{len(param_combinations)}] Testing parameters:")
        print(f"  spatial_weight_x={swx}, spatial_weight_y={swy}, contour_weight={cw}")
        
        start_time = time.time()
        
        seq_acc, char_acc, total = evaluate_with_params(
            model, img_paths, device, swx, swy, cw, verbose=False
        )
        
        elapsed = time.time() - start_time
        
        print(f"  → Sequence Acc: {seq_acc:.2f}%")
        print(f"  → Character Acc: {char_acc:.2f}%")
        print(f"  → Time: {elapsed:.1f}s")
        
        results.append({
            'spatial_weight_x': swx,
            'spatial_weight_y': swy,
            'contour_weight': cw,
            'sequence_accuracy': seq_acc,
            'character_accuracy': char_acc,
            'total_images': total,
            'time_seconds': elapsed
        })
        
        if seq_acc > best_seq_acc:
            best_seq_acc = seq_acc
            best_params = (swx, swy, cw)
            print(f"  ★ NEW BEST!")
    
    # Save results to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['spatial_weight_x', 'spatial_weight_y', 'contour_weight',
                      'sequence_accuracy', 'character_accuracy', 'total_images', 'time_seconds']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n{'='*70}")
    print(f"Grid Search Complete!")
    print(f"{'='*70}")
    print(f"Results saved to: {output_csv}")
    print(f"\nBest Parameters:")
    print(f"  spatial_weight_x={best_params[0]}")
    print(f"  spatial_weight_y={best_params[1]}")
    print(f"  contour_weight={best_params[2]}")
    print(f"  Sequence Accuracy: {best_seq_acc:.2f}%")
    print(f"{'='*70}\n")
    
    return results, best_params


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Grid search for segmentation hyperparameters')
    parser.add_argument('--model', type=str, default='best_char_recognition_model.pth',
                        help='Path to trained model weights')
    parser.add_argument('--data', type=str, default='processed/test',
                        help='Directory with test images')
    parser.add_argument('--output', type=str, default='grid_search_results.csv',
                        help='Output CSV file for results')
    
    # Parameter ranges (customize these!)
    parser.add_argument('--swx-values', nargs='+', type=float,
                        default=[0.5, 1.0, 1.5, 2.0, 2.5],
                        help='Values for spatial_weight_x')
    parser.add_argument('--swy-values', nargs='+', type=float,
                        default=[0.2, 0.4, 0.6, 0.8],
                        help='Values for spatial_weight_y')
    parser.add_argument('--cw-values', nargs='+', type=float,
                        default=[50, 75, 100, 125, 150],
                        help='Values for contour_weight')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = EfficientCaptchaCNN(num_classes=36).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print("Model loaded successfully!")
    
    # Get test images
    data_dir = Path(args.data)
    img_paths = [
        str(f) for f in data_dir.glob("*.png")
    ]
    
    if not img_paths:
        print(f"Error: No images found in {data_dir}")
        return
    
    print(f"Found {len(img_paths)} test images")
    
    # Define parameter grid
    param_grid = {
        'spatial_weight_x': args.swx_values,
        'spatial_weight_y': args.swy_values,
        'contour_weight': args.cw_values
    }
    
    # Run grid search
    results, best_params = grid_search(model, img_paths, device, param_grid, args.output)


if __name__ == '__main__':
    main()
