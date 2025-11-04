"""
Process images under a training folder with Color KMeans segmentation and produce a summary.

Usage (PowerShell):
    python batch_process_train.py                 # uses default 'processed/train'
    python batch_process_train.py -d data/train    # specify folder

Outputs:
- Saved crops under OUTPUT_DIR (default: 'processed_kmeans_output/<foldername>')
- CSV summary at OUTPUT_DIR/summary.csv
- Prints proportion of images where found_chars == expected_chars
"""

import argparse
from pathlib import Path
import csv
import time
from improved_segmentation import segment_captcha


def infer_expected_chars(path: Path):
    stem = path.stem
    if '-' in stem:
        return len(stem.split('-')[0])
    return None


def find_images(folder: Path):
    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]


def process_folder(folder: Path, output_root: Path, method='color_kmeans'):
    images = find_images(folder)
    print(f"[INFO] Found {len(images)} images in {folder}")

    summary = []
    correct_count = 0
    total = 0
    start_time_all = time.time()

    out_dir = output_root / folder.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for img in images:
        expected = infer_expected_chars(img)
        total += 1
        t0 = time.time()
        try:
            saved = segment_captcha(
                str(img),
                method=method,
                expected_chars=expected,
                output_dir=str(out_dir),
                visualize=False,
            )
            elapsed = time.time() - t0
            found = len(saved)
            ok = (expected is None) or (found == expected)
            if ok:
                correct_count += 1
            status = 'OK' if ok else f'MISMATCH (found {found})'
            summary.append([str(img), expected if expected is not None else '', found, f"{elapsed:.3f}", status])
            print(f"[DONE] {img.name}: expected={expected} found={found} time={elapsed:.3f}s -> {status}")
        except Exception as e:
            elapsed = time.time() - t0
            summary.append([str(img), expected if expected is not None else '', 0, f"{elapsed:.3f}", f'ERROR: {e}'])
            print(f"[ERROR] {img.name}: {e}")

    total_time = time.time() - start_time_all
    # Write CSV
    csv_path = out_dir / 'summary.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'expected_chars', 'found_chars', 'time_s', 'status'])
        writer.writerows(summary)

    prop = correct_count / total * 100 if total > 0 else 0
    print('\n' + '='*60)
    print(f"Processed {total} images in {total_time:.2f}s")
    print(f"Correct count: {correct_count} / {total} ({prop:.1f}%)")
    print(f"Summary CSV: {csv_path.resolve()}")

    return csv_path, prop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='processed/train', help='Folder to process')
    parser.add_argument('-o', '--out', type=str, default='processed_kmeans_output', help='Output root folder')
    parser.add_argument('-m', '--method', type=str, default='color_kmeans', help='Segmentation method')
    args = parser.parse_args()

    folder = Path(args.dir)
    if not folder.exists() or not folder.is_dir():
        print(f"Folder {folder} does not exist or is not a directory")
        return

    output_root = Path(args.out)
    csv_path, prop = process_folder(folder, output_root, method=args.method)


if __name__ == '__main__':
    main()
