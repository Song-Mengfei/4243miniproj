import cv2
import numpy as np

def remove_black_lines_mode(img_path, threshold=50):
    """
    Removes black/noise lines from CAPTCHA images by replacing dark pixels
    with the mode of their 8 non-black neighboring pixels.
    If all neighbors are black, use global median color as fallback.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Identify black pixels
    mask = np.all(img < threshold, axis=2)
    padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='reflect')

    # Precompute a global fallback color (median of all non-black pixels)
    non_black_pixels = img[~mask]
    if len(non_black_pixels) == 0:
        global_median_color = np.array([128, 128, 128], dtype=np.uint8)
    else:
        global_median_color = np.median(non_black_pixels, axis=0).astype(np.uint8)

    y_idx, x_idx = np.where(mask)

    for y, x in zip(y_idx, x_idx):
        # Get 3x3 neighborhood (excluding center)
        neighbors = padded[y:y+3, x:x+3].reshape(-1, 3)
        neighbors = np.delete(neighbors, 4, axis=0)

        # Filter out dark (black) neighbors
        bright_neighbors = np.array([c for c in neighbors if not np.all(c < threshold)])
        
        # If no bright neighbors, use global median color
        if len(bright_neighbors) == 0:
            img[y, x] = global_median_color
            continue

        # Compute mode of remaining bright neighbors
        neighbor_tuples = [tuple(c) for c in bright_neighbors]
        unique_colors, counts = np.unique(neighbor_tuples, axis=0, return_counts=True)
        mode_color = unique_colors[np.argmax(counts)]
        img[y, x] = mode_color

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# Example usage
if __name__ == "__main__":
    cleaned = remove_black_lines_mode("0abe-0.png")
    cv2.imwrite("captcha_cleaned_mode.png", cleaned)
    print("Saved cleaned image to captcha_cleaned_mode.png")
