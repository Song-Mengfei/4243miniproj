# Grid Search System - Quick Reference

## 📋 Overview

This system helps you find the optimal segmentation parameters for CAPTCHA character recognition by systematically testing different combinations and evaluating them using your trained CNN model.

## 🚀 Quick Start (3 Steps)

### Step 1: Quick Test (5 minutes)
Test current parameters on a small sample:
```powershell
python test_single_config.py --limit 100
```

### Step 2: Coarse Grid Search (15 minutes)
Test 27 combinations to find promising regions:
```powershell
python quick_grid_search.py
```

### Step 3: Analyze Results
```powershell
python visualize_grid_search.py quick_grid_search_results.csv
```

## 📁 Files Created

| File | Purpose |
|------|---------|
| `grid_search_segmentation.py` | Main grid search with full customization (100 combinations default) |
| `quick_grid_search.py` | Fast coarse search (27 combinations) |
| `test_single_config.py` | Test a single parameter set quickly |
| `visualize_grid_search.py` | Generate analysis plots and statistics |
| `GRID_SEARCH_README.md` | Comprehensive documentation |
| `GRID_SEARCH_QUICKSTART.md` | This file |

## 🎯 Common Use Cases

### "I want to quickly test my current parameters"
```powershell
python test_single_config.py --swx 1.5 --swy 0.4 --cw 100 --limit 50
```

### "I want to find the best parameters quickly"
```powershell
python quick_grid_search.py
python visualize_grid_search.py quick_grid_search_results.csv
```

### "I want a thorough search"
```powershell
python grid_search_segmentation.py
python visualize_grid_search.py grid_search_results.csv
```

### "I want to fine-tune around known good values"
```powershell
python grid_search_segmentation.py `
    --swx-values 1.3 1.4 1.5 1.6 1.7 `
    --swy-values 0.35 0.40 0.45 `
    --cw-values 90 95 100 105 110 `
    --output fine_tuned.csv
```

## 📊 Understanding Results

Open the CSV in Excel or Python:
```python
import pandas as pd
df = pd.read_csv('grid_search_results.csv')

# Find best configuration
best = df.loc[df['sequence_accuracy'].idxmax()]
print(f"Best: swx={best['spatial_weight_x']}, swy={best['spatial_weight_y']}, cw={best['contour_weight']}")
print(f"Accuracy: {best['sequence_accuracy']:.2f}%")
```

## 🔧 Applying Best Parameters

Once you find the best parameters, update your code:

### Option 1: Update segmentation_v2.py defaults
```python
def segment_characters(image_path, min_pixels=3, top_components=5,
                      spatial_weight_x=1.5,  # ← Update these
                      spatial_weight_y=0.4,  # ← Update these
                      contour_weight=100):   # ← Update these
```

### Option 2: Use in export script
```powershell
python export_chars_by_class.py `
    --spatial-weight-x 1.5 `
    --spatial-weight-y 0.4 `
    --contour-weight 100
```

### Option 3: Use in your pipeline
```python
from segmentation_v2 import segment_characters

char_images, char_labels = segment_characters(
    image_path,
    spatial_weight_x=1.5,  # Use your best values
    spatial_weight_y=0.4,
    contour_weight=100
)
```

## ⚡ Performance Tips

### Faster iteration
- Use `--limit` to test on fewer images: `python test_single_config.py --limit 100`
- Use `quick_grid_search.py` first (27 combinations vs 100)

### Comprehensive search
- Run `grid_search_segmentation.py` overnight
- Test on full dataset for final validation

## 🔍 Troubleshooting

### "No images found"
Check your path:
```powershell
ls processed/test/*.png
python grid_search_segmentation.py --data "processed/test"
```

### "Cannot load model"
Verify the model exists:
```powershell
ls best_char_recognition_model.pth
python grid_search_segmentation.py --model "best_char_recognition_model.pth"
```

### "Import errors"
Make sure dependencies are installed:
```powershell
pip install torch torchvision opencv-python numpy pandas matplotlib seaborn scikit-learn tqdm
```

## 📈 Typical Workflow

```
1. Test current params     → test_single_config.py (1 min)
2. Quick exploration       → quick_grid_search.py (15 min)
3. Visualize & analyze     → visualize_grid_search.py (instant)
4. Identify best region    → Read CSV or plots
5. Refined search          → grid_search_segmentation.py with focused ranges (30 min)
6. Final validation        → test_single_config.py with best params (5 min)
7. Update production code  → Apply best parameters
```

## 🎓 Parameter Meanings

- **spatial_weight_x** (0.5-2.5): How much horizontal position matters
  - Higher = more horizontal separation
  
- **spatial_weight_y** (0.2-0.8): How much vertical position matters
  - Higher = more vertical separation
  
- **contour_weight** (50-150): Penalty for disconnected regions
  - Higher = stronger separation of disconnected parts

## 📞 Need More Help?

- **Full documentation**: See `GRID_SEARCH_README.md`
- **Code details**: Check inline comments in each script
- **Visualization**: Run `visualize_grid_search.py` on your results

## 🎉 Example Success Story

```powershell
# Before: Using default params (swx=1.5, swy=0.4, cw=100)
python test_single_config.py
# Result: 78.5% sequence accuracy

# After: Running grid search
python quick_grid_search.py
python visualize_grid_search.py quick_grid_search_results.csv
# Found: swx=1.8, swy=0.35, cw=110

# Validation with best params
python test_single_config.py --swx 1.8 --swy 0.35 --cw 110
# Result: 85.2% sequence accuracy 🎉 (6.7% improvement!)
```

---
**Good luck with your grid search! 🚀**
