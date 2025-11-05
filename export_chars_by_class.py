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
from improved_segmentation import segment_captcha

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
    parser.add_argument('-m', '--method', type=str, default='color_kmeans', help='Segmentation method to use')
    parser.add_argument('--keep-staging', action='store_true', help='Keep staging crops instead of deleting')
    args = parser.parse_args()

    input_dir = Path(args.dir)
    out_root = Path(args.out)
    staging_root = Path(STAGING_DIR)
    staging_root.mkdir(exist_ok=True)
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
            # Stage crops (this returns paths in left-to-right order)
            staged_dir = staging_root / img.stem
            staged_dir.mkdir(exist_ok=True)
            saved_paths = segment_captcha(
                str(img),
                method=args.method,
                expected_chars=expected,
                output_dir=str(staged_dir),
                visualize=False,
            )
            elapsed = time.time() - start
            found = len(saved_paths)

            # Align to label and export
            status = 'OK'
            if expected is None:
                status = 'NO_LABEL'
            elif found != expected:
                status = f'MISMATCH (found {found})'
            else:
                correct += 1

            if label and found == expected:
                for i, crop_path in enumerate(saved_paths):
                    c = label[i]
                    group = group_of_char(c)
                    dest_dir = out_root / group / c
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_name = f"{img.stem}_char{i}.png"
                    shutil.copy2(crop_path, dest_dir / dest_name)

            summary.append([
                str(img), label if label else '', expected if expected is not None else '',
                found, f"{elapsed:.3f}", status
            ])

            if not args.keep_staging:
                try:
                    shutil.rmtree(staged_dir)
                except Exception:
                    pass

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
