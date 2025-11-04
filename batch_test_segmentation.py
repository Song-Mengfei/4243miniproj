"""
Test segmentation accuracy on a folder of CAPTCHA images.
Similar to batch_process_train.py but using segmentation.py method.

Usage (PowerShell):
    python batch_test_segmentation.py                      # uses default 'processed/train'
    python batch_test_segmentation.py -d processed/test    # specify folder

Outputs:
- Saved segmented characters under OUTPUT_DIR
- CSV summary with segmentation statistics
- Prints proportion of correctly segmented images
"""

import argparse
from pathlib import Path
import csv
import time
import cv2
import numpy as np
from segmentation import get_dominant_clusters


def segment_characters_batch(image_path, min_pixels=3, top_components=5, output_dir=None):
    """
    Segment CAPTCHA characters and optionally save them.
    Returns (char_images, expected_chars, found_chars, min_idx)
    """
    filename = Path(image_path).stem
    chars_part = filename.split("-")[0]
    expected_chars = len(chars_part)
    n_clusters = expected_chars + 1  # +1 for noise cluster

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    threshold = 250
    max_threshold = 250
    step = 10
    valid_bounding_boxes = []

    # Try increasing threshold until enough clusters found
    while threshold <= max_threshold:
        clustered_masks, mask_foreground = get_dominant_clusters(image, n_clusters, threshold)
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
        return [], expected_chars, 0, -1

    # Sort clusters left-to-right
    valid_bounding_boxes.sort(key=lambda b: b[0])

    cluster_pixel_counts = []
    cleaned_clusters = []

    # Clean clusters (keep top N largest connected parts)
    for i, (x, y, w, h, mask) in enumerate(valid_bounding_boxes):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            sorted_idx = np.argsort(areas)[::-1]  # largest → smallest

            keep_mask = np.zeros_like(mask)
            count_kept = 0
            for idx2 in sorted_idx:
                if count_kept >= top_components:
                    break
                area = areas[idx2]
                if area < min_pixels:
                    continue
                label = idx2 + 1
                keep_mask[labels == label] = 255
                count_kept += 1
            cleaned_mask = keep_mask
        else:
            cleaned_mask = mask.copy()

        pixel_count = np.sum(cleaned_mask > 0)
        cluster_pixel_counts.append(pixel_count)
        cleaned_clusters.append((x, y, w, h, cleaned_mask))

    # Identify cluster with fewest pixels (noise)
    min_idx = int(np.argmin(cluster_pixel_counts))

    # Extract character images (excluding the noise cluster)
    char_images = []
    saved_paths = []
    
    for i, (x, y, w, h, cleaned_mask) in enumerate(cleaned_clusters):
        roi = image[y:y + h, x:x + w]
        mask_roi = cleaned_mask[y:y + h, x:x + w]
        char_img = np.full_like(roi, 255)  # White background
        char_img[mask_roi == 255] = roi[mask_roi == 255]
        
        if i != min_idx:  # Skip noise cluster for counting
            char_images.append(char_img)
            
            # Save if output_dir provided
            if output_dir:
                char_idx = len(char_images) - 1
                if char_idx < len(chars_part):
                    true_label = chars_part[char_idx]
                    save_path = output_dir / f"{filename}_char{char_idx}_{true_label}.png"
                else:
                    save_path = output_dir / f"{filename}_char{char_idx}_unknown.png"
                cv2.imwrite(str(save_path), char_img)
                saved_paths.append(save_path)

    found_chars = len(char_images)
    return char_images, expected_chars, found_chars, min_idx


def find_images(folder: Path):
    """Find all image files in folder"""
    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]


def process_folder(folder: Path, output_root: Path, save_chars=False):
    """Process all images in folder and generate summary"""
    images = find_images(folder)
    print(f"[INFO] Found {len(images)} images in {folder}")

    summary = []
    correct_count = 0
    total = 0
    start_time_all = time.time()

    # Create output directory
    out_dir = output_root / folder.name
    if save_chars:
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)  # Still need for CSV

    for img_path in images:
        total += 1
        t0 = time.time()
        try:
            char_imgs, expected, found, min_idx = segment_characters_batch(
                img_path,
                output_dir=out_dir if save_chars else None
            )
            elapsed = time.time() - t0
            
            ok = (found == expected)
            if ok:
                correct_count += 1
            
            status = 'OK' if ok else f'MISMATCH'
            summary.append([
                img_path.name,
                expected,
                found,
                f"{elapsed:.3f}",
                status,
                min_idx
            ])
            
            status_symbol = '✓' if ok else '✗'
            print(f"{status_symbol} {img_path.name}: expected={expected} found={found} time={elapsed:.3f}s noise_cluster={min_idx}")
            
        except Exception as e:
            elapsed = time.time() - t0
            summary.append([
                img_path.name,
                '',
                0,
                f"{elapsed:.3f}",
                f'ERROR',
                str(e)
            ])
            print(f"✗ {img_path.name}: ERROR - {e}")

    total_time = time.time() - start_time_all
    
    # Write CSV
    csv_path = out_dir / 'segmentation_summary.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'expected_chars', 'found_chars', 'time_s', 'status', 'noise_cluster_idx'])
        writer.writerows(summary)

    prop = correct_count / total * 100 if total > 0 else 0
    
    print('\n' + '='*70)
    print(f"SUMMARY:")
    print(f"  Total images: {total}")
    print(f"  Correctly segmented: {correct_count} / {total} ({prop:.1f}%)")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time per image: {total_time/total:.3f}s")
    print(f"  Summary CSV: {csv_path.resolve()}")
    print('='*70)

    return csv_path, prop


def main():
    parser = argparse.ArgumentParser(description='Test segmentation accuracy on CAPTCHA images')
    parser.add_argument('-d', '--dir', type=str, default='processed/train',
                        help='Folder to process (default: processed/train)')
    parser.add_argument('-o', '--out', type=str, default='segmentation_test_output',
                        help='Output root folder (default: segmentation_test_output)')
    parser.add_argument('-s', '--save', action='store_true',
                        help='Save segmented characters to disk')
    args = parser.parse_args()

    folder = Path(args.dir)
    if not folder.exists() or not folder.is_dir():
        print(f"[ERROR] Folder {folder} does not exist or is not a directory")
        return

    output_root = Path(args.out)
    
    print(f"\n{'='*70}")
    print(f"Testing Segmentation Accuracy")
    print(f"  Input folder: {folder.resolve()}")
    print(f"  Output folder: {output_root.resolve()}")
    print(f"  Save characters: {args.save}")
    print(f"{'='*70}\n")
    
    csv_path, accuracy = process_folder(folder, output_root, save_chars=args.save)
    
    print(f"\n✓ Done! Accuracy: {accuracy:.1f}%\n")


if __name__ == '__main__':
    main()
