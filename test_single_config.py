"""
Test Single Parameter Configuration

Quick test of a specific parameter combination to verify before full grid search.
Useful for debugging or testing specific values.

Usage:
    # Test default values
    python test_single_config.py

    # Test specific values
    python test_single_config.py --swx 1.5 --swy 0.4 --cw 100

    # Test on specific images
    python test_single_config.py --data processed/test --limit 50
"""

import argparse
from pathlib import Path
import time

import torch
from tqdm import tqdm

from grid_search_segmentation import (
    EfficientCaptchaCNN, recognize_captcha
)


def test_configuration(model, img_paths, device, swx, swy, cw, limit=None):
    """Test a single parameter configuration."""
    if limit:
        img_paths = img_paths[:limit]
    
    print(f"\nTesting configuration:")
    print(f"  spatial_weight_x = {swx}")
    print(f"  spatial_weight_y = {swy}")
    print(f"  contour_weight = {cw}")
    print(f"  Test images: {len(img_paths)}")
    print()
    
    model.eval()
    
    total = 0
    correct = 0
    char_total = 0
    char_correct = 0
    errors = []
    
    start_time = time.time()
    
    for img_path in tqdm(img_paths, desc="Testing"):
        try:
            filename = Path(img_path).stem
            gt_text = filename.split('-')[0].lower()
            
            pred_text = recognize_captcha(
                model, img_path, device, swx, swy, cw
            )
            pred_text = pred_text.lower()
            
            total += 1
            is_correct = (gt_text == pred_text)
            if is_correct:
                correct += 1
            else:
                errors.append((Path(img_path).name, gt_text, pred_text))
            
            # Character-level accuracy
            min_len = min(len(gt_text), len(pred_text))
            for i in range(min_len):
                char_total += 1
                if gt_text[i] == pred_text[i]:
                    char_correct += 1
                    
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            continue
    
    elapsed = time.time() - start_time
    
    seq_acc = 100 * correct / total if total > 0 else 0
    char_acc = 100 * char_correct / char_total if char_total > 0 else 0
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print('='*70)
    print(f"Total images: {total}")
    print(f"Processing time: {elapsed:.1f}s ({elapsed/total:.2f}s per image)")
    print()
    print(f"Sequence Accuracy: {seq_acc:.2f}% ({correct}/{total})")
    print(f"Character Accuracy: {char_acc:.2f}% ({char_correct}/{char_total})")
    print('='*70)
    
    # Show some errors if any
    if errors and len(errors) <= 10:
        print(f"\nAll {len(errors)} errors:")
        for fname, gt, pred in errors:
            print(f"  {fname}: GT='{gt}' PRED='{pred}'")
    elif errors:
        print(f"\nFirst 10 of {len(errors)} errors:")
        for fname, gt, pred in errors[:10]:
            print(f"  {fname}: GT='{gt}' PRED='{pred}'")
    else:
        print("\n🎉 Perfect accuracy on this test set!")
    
    return seq_acc, char_acc


def main():
    parser = argparse.ArgumentParser(description='Test single parameter configuration')
    parser.add_argument('--model', type=str, default='best_char_recognition_model.pth',
                       help='Path to trained model weights')
    parser.add_argument('--data', type=str, default='processed/test',
                       help='Directory with test images')
    parser.add_argument('--swx', type=float, default=1.5,
                       help='spatial_weight_x value')
    parser.add_argument('--swy', type=float, default=0.4,
                       help='spatial_weight_y value')
    parser.add_argument('--cw', type=float, default=100,
                       help='contour_weight value')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of test images (for quick testing)')
    
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
    img_paths = [str(f) for f in data_dir.glob("*.png")]
    
    if not img_paths:
        print(f"Error: No images found in {data_dir}")
        return
    
    print(f"Found {len(img_paths)} test images")
    
    # Test configuration
    test_configuration(
        model, img_paths, device,
        args.swx, args.swy, args.cw,
        limit=args.limit
    )


if __name__ == '__main__':
    main()
