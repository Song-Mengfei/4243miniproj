# Grid Search for Segmentation Hyperparameters

This document explains how to use the grid search scripts to optimize the segmentation parameters (`spatial_weight_x`, `spatial_weight_y`, and `contour_weight`) for CAPTCHA character segmentation.

## Overview

The grid search system evaluates different combinations of segmentation parameters by:
1. Segmenting CAPTCHA images with each parameter combination
2. Recognizing characters using the trained CNN model
3. Computing sequence-level and character-level accuracy
4. Identifying the best parameter combination

## Files

- **`grid_search_segmentation.py`**: Main grid search script with customizable parameter ranges
- **`quick_grid_search.py`**: Fast coarse grid search for initial exploration (27 combinations)
- **`GRID_SEARCH_README.md`**: This documentation file

## Quick Start

### 1. Run Quick Grid Search (Recommended First Step)

Start with a coarse grid to quickly identify promising parameter regions:

```powershell
python quick_grid_search.py
```

This tests 27 combinations (3×3×3 grid):
- `spatial_weight_x`: [0.5, 1.5, 2.5]
- `spatial_weight_y`: [0.2, 0.4, 0.6]
- `contour_weight`: [50, 100, 150]

**Expected time:** ~5-15 minutes for 1000 test images

### 2. Review Results

The results are saved to `quick_grid_search_results.csv`:

```csv
spatial_weight_x,spatial_weight_y,contour_weight,sequence_accuracy,character_accuracy,total_images,time_seconds
0.5,0.2,50,75.3,88.2,1000,12.5
1.5,0.4,100,82.1,92.7,1000,13.2
...
```

Open in Excel or use Python to find the best:
```python
import pandas as pd
df = pd.read_csv('quick_grid_search_results.csv')
best = df.loc[df['sequence_accuracy'].idxmax()]
print(best)
```

### 3. Run Detailed Grid Search

Once you identify a promising region, refine the search:

```powershell
# Example: If best params were around swx=1.5, swy=0.4, cw=100
python grid_search_segmentation.py `
    --swx-values 1.0 1.25 1.5 1.75 2.0 `
    --swy-values 0.3 0.35 0.4 0.45 0.5 `
    --cw-values 80 90 100 110 120 `
    --output refined_grid_search.csv
```

This tests 125 combinations (5×5×5 grid) around the promising region.

## Full Usage Examples

### Basic Usage

```powershell
# Use default parameters (100 combinations)
python grid_search_segmentation.py

# Specify test data directory
python grid_search_segmentation.py --data processed/test

# Use custom model
python grid_search_segmentation.py --model my_model.pth
```

### Custom Parameter Ranges

```powershell
# Test specific values
python grid_search_segmentation.py `
    --swx-values 0.5 1.0 1.5 2.0 `
    --swy-values 0.2 0.4 0.6 `
    --cw-values 50 100 150 `
    --output custom_results.csv

# Fine-grained search around current best (1.5, 0.4, 100)
python grid_search_segmentation.py `
    --swx-values 1.3 1.4 1.5 1.6 1.7 `
    --swy-values 0.35 0.40 0.45 `
    --cw-values 90 95 100 105 110 `
    --output fine_tuned_results.csv
```

### Test on Different Datasets

```powershell
# Test on validation set
python grid_search_segmentation.py --data processed/validation

# Test on specific subset
python grid_search_segmentation.py --data my_test_output/test
```

## Command-Line Arguments

### grid_search_segmentation.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `best_char_recognition_model.pth` | Path to trained model weights |
| `--data` | `processed/test` | Directory with test CAPTCHA images |
| `--output` | `grid_search_results.csv` | Output CSV file for results |
| `--swx-values` | `[0.5, 1.0, 1.5, 2.0, 2.5]` | Values for spatial_weight_x |
| `--swy-values` | `[0.2, 0.4, 0.6, 0.8]` | Values for spatial_weight_y |
| `--cw-values` | `[50, 75, 100, 125, 150]` | Values for contour_weight |

**Default grid size:** 5 × 4 × 5 = 100 combinations

## Parameter Descriptions

### spatial_weight_x
- **Range:** 0.5 - 3.0 (typical)
- **Effect:** Controls how much horizontal position influences clustering
- **Higher values:** Characters separated more by horizontal position
- **Lower values:** More emphasis on color similarity

### spatial_weight_y
- **Range:** 0.2 - 1.0 (typical)
- **Effect:** Controls how much vertical position influences clustering
- **Higher values:** Characters separated more by vertical position
- **Lower values:** More tolerance for vertical alignment variations

### contour_weight
- **Range:** 50 - 200 (typical)
- **Effect:** Penalty for pixels in different connected components
- **Higher values:** Stronger separation of disconnected regions
- **Lower values:** More likely to group nearby disconnected regions

## Interpreting Results

### Metrics

- **Sequence Accuracy:** Percentage of CAPTCHAs where ALL characters are correct
  - Most important metric for end-to-end performance
  - Example: 85% means 850 out of 1000 CAPTCHAs fully correct

- **Character Accuracy:** Percentage of individual characters recognized correctly
  - Shows per-character performance
  - Example: 95% means 5700 out of 6000 characters correct

- **Total Images:** Number of images successfully processed
  - Should equal your test set size
  - Lower values indicate segmentation failures

### Analysis Tips

1. **Sort by sequence accuracy** to find the best overall combination
2. **Look for stability** - good parameters should have similar performance across runs
3. **Consider character accuracy** - high char accuracy with low sequence accuracy suggests inconsistent segmentation
4. **Check processing time** - some parameter combinations may be slower

### Example Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('grid_search_results.csv')

# Find top 10 configurations
top10 = df.nlargest(10, 'sequence_accuracy')
print(top10)

# Visualize parameter effects
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

df.groupby('spatial_weight_x')['sequence_accuracy'].mean().plot(ax=axes[0], marker='o')
axes[0].set_title('Effect of spatial_weight_x')
axes[0].set_xlabel('spatial_weight_x')
axes[0].set_ylabel('Mean Sequence Accuracy (%)')

df.groupby('spatial_weight_y')['sequence_accuracy'].mean().plot(ax=axes[1], marker='o')
axes[1].set_title('Effect of spatial_weight_y')
axes[1].set_xlabel('spatial_weight_y')

df.groupby('contour_weight')['sequence_accuracy'].mean().plot(ax=axes[2], marker='o')
axes[2].set_title('Effect of contour_weight')
axes[2].set_xlabel('contour_weight')

plt.tight_layout()
plt.savefig('parameter_effects.png')
plt.show()
```

## Recommended Workflow

### Phase 1: Quick Exploration (27 combinations)
```powershell
python quick_grid_search.py
# Expected time: 5-15 minutes
```

### Phase 2: Identify Best Region
```python
import pandas as pd
df = pd.read_csv('quick_grid_search_results.csv')
best = df.nlargest(5, 'sequence_accuracy')
print(best)
# Look for clusters of good parameters
```

### Phase 3: Refined Search (125 combinations)
```powershell
# Based on Phase 1 results, refine around best region
python grid_search_segmentation.py `
    --swx-values 1.0 1.25 1.5 1.75 2.0 `
    --swy-values 0.3 0.35 0.4 0.45 0.5 `
    --cw-values 80 90 100 110 120 `
    --output refined_search.csv
# Expected time: 30-60 minutes
```

### Phase 4: Fine-Tuning (64 combinations)
```powershell
# Very fine-grained search around the best parameters
python grid_search_segmentation.py `
    --swx-values 1.4 1.45 1.5 1.55 `
    --swy-values 0.38 0.40 0.42 0.44 `
    --cw-values 95 100 105 110 `
    --output final_tuning.csv
# Expected time: 15-30 minutes
```

## Performance Tips

### Reduce Test Set Size
For faster iteration during exploration:
```powershell
# Create a smaller test subset
mkdir processed/test_small
# Copy 200 random images to processed/test_small
python grid_search_segmentation.py --data processed/test_small
```

### Parallel Processing
If you have multiple GPUs or want to run multiple searches:
```powershell
# Terminal 1 - Search spatial_weight_x
python grid_search_segmentation.py `
    --swx-values 0.5 1.0 1.5 `
    --swy-values 0.4 `
    --cw-values 100 `
    --output search_swx.csv

# Terminal 2 - Search spatial_weight_y
python grid_search_segmentation.py `
    --swx-values 1.5 `
    --swy-values 0.2 0.4 0.6 `
    --cw-values 100 `
    --output search_swy.csv
```

## Troubleshooting

### Error: "No images found"
```powershell
# Check the data directory path
python grid_search_segmentation.py --data "processed/test"
# Use absolute path if needed
python grid_search_segmentation.py --data "C:/Users/Jerry Jian/Desktop/CS4243_repo/processed/test"
```

### Error: "Cannot load model"
```powershell
# Verify model file exists
ls best_char_recognition_model.pth
# Specify full path if needed
python grid_search_segmentation.py --model "C:/path/to/model.pth"
```

### Low Accuracy Results
- Ensure you're using the correct trained model
- Verify test images have proper naming format: `{label}-{id}.png`
- Check that segmentation is working on individual images first

### Out of Memory
- Reduce test set size
- Use CPU instead of GPU: Set `CUDA_VISIBLE_DEVICES=-1`

## Next Steps After Grid Search

Once you find the best parameters:

1. **Update your segmentation code:**
   ```python
   # In segmentation_v2.py or your pipeline
   segment_characters(
       image_path,
       spatial_weight_x=1.5,    # Use your best values
       spatial_weight_y=0.4,
       contour_weight=100
   )
   ```

2. **Update export script:**
   ```powershell
   python export_chars_by_class.py `
       --spatial-weight-x 1.5 `
       --spatial-weight-y 0.4 `
       --contour-weight 100
   ```

3. **Re-run full pipeline with optimal parameters**

4. **Document the best parameters** in your main README or config file

## Output Files

All results are saved as CSV files with the following columns:
- `spatial_weight_x`: Parameter value
- `spatial_weight_y`: Parameter value
- `contour_weight`: Parameter value
- `sequence_accuracy`: % of fully correct CAPTCHAs
- `character_accuracy`: % of correct individual characters
- `total_images`: Number of processed images
- `time_seconds`: Processing time for this combination

## Support

If you encounter issues or need to modify the search:
- Check the inline comments in `grid_search_segmentation.py`
- Adjust parameter ranges in the script
- Contact the development team with your results CSV for analysis
