"""
Quick Grid Search - Coarse parameter sweep for fast initial tuning

This is a faster version with a coarser grid for initial hyperparameter exploration.
Once you identify promising regions, use grid_search_segmentation.py with finer granularity.

Usage:
    python quick_grid_search.py
    python quick_grid_search.py --data processed/test
"""

import argparse
from pathlib import Path
import torch
from grid_search_segmentation import (
    EfficientCaptchaCNN, grid_search
)


def main():
    parser = argparse.ArgumentParser(description='Quick coarse grid search')
    parser.add_argument('--model', type=str, default='best_char_recognition_model.pth',
                        help='Path to trained model weights')
    parser.add_argument('--data', type=str, default='processed/test',
                        help='Directory with test images')
    parser.add_argument('--output', type=str, default='quick_grid_search_results.csv',
                        help='Output CSV file for results')
    
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
    
    # Coarse parameter grid for quick exploration
    param_grid = {
        'spatial_weight_x': [0.5, 1.5, 2.5],      # 3 values
        'spatial_weight_y': [0.2, 0.4, 0.6],      # 3 values
        'contour_weight': [50, 100, 150],          # 3 values
    }
    # Total: 3 × 3 × 3 = 27 combinations
    
    print("\nQuick Grid Search Configuration:")
    print("  This will test 27 combinations (much faster than full grid)")
    
    # Run grid search
    results, best_params = grid_search(model, img_paths, device, param_grid, args.output)
    
    print("\nNext Steps:")
    print("  1. Review results in", args.output)
    print("  2. Identify promising parameter ranges")
    print("  3. Run full grid search with finer granularity around best parameters")


if __name__ == '__main__':
    main()
