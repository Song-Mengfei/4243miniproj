"""
Bayesian Optimization for Segmentation Hyperparameters

This script optimizes spatial_weight_x, spatial_weight_y, and contour_weight
parameters for the segment_characters_custom function using Bayesian Optimization.
It evaluates performance with a trained model and reports both sequence-level and
character-level accuracy.

Usage:
    python BO_segmentation.py
    python BO_segmentation.py --model best_char_recognition_model.pth
    python BO_segmentation.py --data processed/train --output bo_search_results.csv
"""

# =============================================================================
# Imports
# =============================================================================
import argparse
import csv
import os
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# Suppress warnings
warnings.filterwarnings("ignore")

# Import your segmentation function
from segmentation_v2 import get_dominant_clusters

# =============================================================================
# Bootstrap Sampling
# =============================================================================
def bootstrap_sample(img_paths, n_samples=2000, random_state=42):
    rng = np.random.default_rng(random_state)
    if len(img_paths) <= n_samples:
        return img_paths.copy()
    indices = rng.choice(len(img_paths), size=n_samples, replace=True)
    return [img_paths[i] for i in indices]

# =============================================================================
# Model Definition
# =============================================================================

class InvertedResidualBlock(nn.Module):
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
# Character Segmentation
# =============================================================================

def segment_characters_custom(image_path, min_pixels=3, top_components=5,
                               spatial_weight_x=1.5, spatial_weight_y=0.4,
                               contour_weight=100):
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
        gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        ys_fg, xs_fg = np.where(gray < 250)
        if len(xs_fg) > 0 and len(ys_fg) > 0:
            x_min_fg, x_max_fg = np.min(xs_fg), np.max(xs_fg)
            y_min_fg, y_max_fg = np.min(ys_fg), np.max(ys_fg)
            char_img = char_img[y_min_fg:y_max_fg + 1, x_min_fg:x_max_fg + 1]
        char_images.append(char_img)

    return char_images

# =============================================================================
# Recognition and Evaluation
# =============================================================================

CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"
id2ch = {i: ch for i, ch in enumerate(CHARS)}

def preprocess_char(char_img, target_size=(32, 32)):
    if len(char_img.shape) == 3:
        char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    h, w = char_img.shape
    target_h, target_w = target_size
    scale = min(target_h / h, target_w / w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    char_img = cv2.resize(char_img, (new_w, new_h))
    canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = char_img
    return canvas

def recognize_captcha(model, image_path, device, spatial_weight_x, spatial_weight_y, contour_weight):
    model.eval()
    char_images = segment_characters_custom(image_path, spatial_weight_x=spatial_weight_x,
                                            spatial_weight_y=spatial_weight_y,
                                            contour_weight=contour_weight)
    if not char_images:
        return ""
    preds = []
    with torch.no_grad():
        for char_img in char_images:
            canvas = preprocess_char(char_img)
            tensor = torch.from_numpy(canvas).float() / 255.0
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
            output = model(tensor)
            _, predicted = output.max(1)
            preds.append(id2ch[predicted.item()])
    return ''.join(preds)

def evaluate_with_params(model, img_paths, device, swx, swy, cw):
    model.eval()
    total, correct, char_total, char_correct = 0, 0, 0, 0
    for img_path in tqdm(img_paths, desc=f"Evaluating swx={swx:.2f}, swy={swy:.2f}, cw={cw:.1f}", leave=False):
        filename = Path(img_path).stem
        gt_text = filename.split('-')[0].lower()
        pred_text = recognize_captcha(model, img_path, device, swx, swy, cw).lower()
        total += 1
        if gt_text == pred_text:
            correct += 1
        for a, b in zip(gt_text, pred_text):
            char_total += 1
            if a == b:
                char_correct += 1
    seq_acc = 100 * correct / total if total > 0 else 0
    char_acc = 100 * char_correct / char_total if char_total > 0 else 0
    return seq_acc, char_acc, total

# =============================================================================
# Bayesian Optimization
# =============================================================================

def bayesian_optimize(model, img_paths, device, output_csv):
    print("\n[INFO] Starting Bayesian Optimization...")
    space = [
        Real(0.1, 5.0, name='spatial_weight_x'),
        Real(0.1, 2.0, name='spatial_weight_y'),
        Real(20, 300, name='contour_weight')
    ]
    results = []
    call_counter = {'n': 0}

    @use_named_args(space)
    def objective(spatial_weight_x, spatial_weight_y, contour_weight):
        call_counter['n'] += 1
        print(f"\n[Step {call_counter['n']}/30] Testing params: swx={spatial_weight_x:.2f}, swy={spatial_weight_y:.2f}, cw={contour_weight:.1f}")

        # Bootstrap sample 2000 images for evaluation
        sampled_paths = bootstrap_sample(img_paths, n_samples=2000)
        seq_acc, char_acc, total = evaluate_with_params(
            model, sampled_paths, device, spatial_weight_x, spatial_weight_y, contour_weight)

        results.append({
            'spatial_weight_x': spatial_weight_x,
            'spatial_weight_y': spatial_weight_y,
            'contour_weight': contour_weight,
            'sequence_accuracy': seq_acc,
            'character_accuracy': char_acc,
            'total_images': total
        })
        print(f"→ SeqAcc={seq_acc:.2f}% | CharAcc={char_acc:.2f}% (Total={total})")
        return -seq_acc  # minimize negative accuracy

    opt_result = gp_minimize(objective, space, n_calls=30, n_random_starts=8, random_state=42, verbose=False)
    pd.DataFrame(results).to_csv(output_csv, index=False)

    best_params = opt_result.x
    print("\n[RESULT] Optimization Complete ✅")
    print(f"Best Params: swx={best_params[0]:.3f}, swy={best_params[1]:.3f}, cw={best_params[2]:.3f}")
    print(f"Best Sequence Accuracy: {-opt_result.fun:.2f}%")
    return results, best_params

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Bayesian optimization for segmentation hyperparameters')
    parser.add_argument('--model', type=str, default='best_char_recognition_model.pth')
    parser.add_argument('--data', type=str, default='processed/train')
    parser.add_argument('--output', type=str, default='bo_search_results.csv')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = EfficientCaptchaCNN(num_classes=36).to(device)
    print(f"[INFO] Loading model from: {args.model}")
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    img_paths = [str(f) for f in Path(args.data).glob("*.png")]
    if not img_paths:
        print("[ERROR] No images found in dataset.")
        return

    print(f"[INFO] Found {len(img_paths)} images for evaluation.")
    bayesian_optimize(model, img_paths, device, args.output)
    print(f"[INFO] Results saved to {args.output}")


if __name__ == '__main__':
    main()


