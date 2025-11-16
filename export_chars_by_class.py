"""
Export segmented CAPTCHA characters into class-specific folders using Color KMeans.

- Assumes filename encodes ground-truth string before the first '-' (e.g., '1cfu8x-0.png').
- Segments left-to-right and aligns each crop to the corresponding character in the label.
- Writes crops to: <out_root>/<group>/<char>/, where group is 'digits', 'letters', or 'others'.

Usage (PowerShell):
    # Export from current directory
    python export_chars_by_class.py

    # Export from a specific folder (e.g., processed/train)
    python export_chars_by_class.py -d processed/train

    # Change output root
    python export_chars_by_class.py -d processed/train -o chars_by_class
"""

import argparse
from pathlib import Path
import shutil
import csv
import time
import cv2
from segmentation import segment_characters

DEFAULT_OUT = 'chars_by_class'
STAGING_DIR = '_staging_chars'
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}


def group_of_char(c: str) -> str:
    if c.isdigit():
        return 'digits'
    if c.isalpha():
        return 'letters'
    return 'others'


essential_cols = ['image', 'label', 'expected_chars', 'found_chars', 'time_s', 'status']


def find_images(input_dir: Path):
    files = []
    for p in input_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return files


def infer_label_from_name(path: Path):
    stem = path.stem
    if '-' in stem:
        return stem.split('-')[0]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='.', help='Input directory with CAPTCHA images')
    parser.add_argument('-o', '--out', type=str, default=DEFAULT_OUT, help='Output root for class folders')
    parser.add_argument('--spatial-weight-x', type=float, default=1.5, help='Spatial weight for x-coordinate')
    parser.add_argument('--spatial-weight-y', type=float, default=0.4, help='Spatial weight for y-coordinate')
    parser.add_argument('--contour-weight', type=float, default=100, help='Weight for contour penalty')
    parser.add_argument('--min-pixels', type=int, default=3, help='Minimum pixels for component')
    parser.add_argument('--top-components', type=int, default=5, help='Number of top components to keep')
    args = parser.parse_args()

    input_dir = Path(args.dir)
    out_root = Path(args.out)
    out_root.mkdir(exist_ok=True)

    images = find_images(input_dir)
    print(f"[INFO] Found {len(images)} images in {input_dir.resolve()}")

    summary = []
    correct = 0

    for img in images:
        label = infer_label_from_name(img)
        expected = len(label) if label else None
        print(f"\n{'='*60}\nProcessing: {img.name} label={label} expected={expected}\n{'='*60}")

        start = time.time()
        try:
            # Segment characters using segmentation_v2 method
            char_images, char_labels = segment_characters(
                str(img),
                min_pixels=args.min_pixels,
                top_components=args.top_components,
                spatial_weight_x=args.spatial_weight_x,
                spatial_weight_y=args.spatial_weight_y,
                contour_weight=args.contour_weight
            )
            elapsed = time.time() - start
            found = len(char_images)

            # Align to label and export
            status = 'OK'
            if expected is None:
                status = 'NO_LABEL'
            elif found != expected:
                status = f'MISMATCH (found {found})'
            else:
                correct += 1

            if label and found == expected:
                for i, char_img in enumerate(char_images):
                    c = label[i]
                    group = group_of_char(c)
                    dest_dir = out_root / group / c
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_name = f"{img.stem}_char{i}.png"
                    dest_path = dest_dir / dest_name
                    cv2.imwrite(str(dest_path), char_img)

            summary.append([
                str(img), label if label else '', expected if expected is not None else '',
                found, f"{elapsed:.3f}", status
            ])

            print(f"[DONE] {img.name}: expected={expected} found={found} -> {status}")
        except Exception as e:
            elapsed = time.time() - start
            summary.append([str(img), label if label else '', expected if expected is not None else '',
                            0, f"{elapsed:.3f}", f"ERROR: {e}"])
            print(f"[ERROR] {img.name}: {e}")

    # Write summary CSV
    csv_path = out_root / 'export_summary.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(essential_cols)
        writer.writerows(summary)

    total = len(images)
    prop = (correct / total * 100) if total else 0.0
    print('\n' + '='*60)
    print(f"Export finished. Correct: {correct}/{total} ({prop:.1f}%)")
    print(f"Summary CSV: {csv_path.resolve()}")


if __name__ == '__main__':
    main()
