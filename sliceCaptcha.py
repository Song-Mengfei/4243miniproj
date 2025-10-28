import re
import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter


def get_dominant_color(roi):
    """get the most common non-background color in the region"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask = gray < 240  # ignore white background


    if not np.any(mask):
        return np.mean(roi.reshape(-1, 3), axis=0)  # fallback to mean if all white

    pixels = roi[mask]  # all non-white pixels
    pixels = (pixels // 10) * 10  # quantize colors to reduce small variations
    pixels_tuples = [tuple(p) for p in pixels]
    dominant_color = Counter(pixels_tuples).most_common(1)[0][0]

    return np.array(dominant_color)


def get_second_dominant_color(roi, first_color):
    """get the second most common color"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask = gray < 240

    if not np.any(mask):
        return np.mean(roi.reshape(-1, 3), axis=0)
    
    pixels = roi[mask]
    pixels = (pixels // 10) * 10
    pixels_tuples = [tuple(p) for p in pixels]
    counts = Counter(pixels_tuples)
    counts.pop(tuple(first_color), None)  # remove first color

    if counts:
        return np.array(counts.most_common(1)[0][0])
    return first_color  # fallback


def is_contained(inner, outer):
    """check if inner box is fully contained within outer box"""
    xi, yi, wi, hi = inner
    xo, yo, wo, ho = outer
    return xi >= xo and yi >= yo and xi+wi <= xo+wo and yi+hi <= yo+ho



def check_and_merge_regions(img, merged_regions, color_threshold):
    """Iteratively merge regions with similar effective colors."""

    # Show regions before merging
    print("[INFO] Regions before final check_and_merge_regions:")
    for idx, (_, (x, y, w, h), color) in enumerate(merged_regions, 1):
        print(f" Box {idx}: position=({x},{y},{w},{h}), effective_color={color.astype(int)}")

    # Copy list to avoid modifying input
    regions = merged_regions.copy()
    has_merged = True

    while has_merged:
        has_merged = False
        merged_indices = set()
        new_regions = []

        for i in range(len(regions)):
            if i in merged_indices:
                continue

            _, (x1, y1, w1, h1), color1 = regions[i]
            color1 = color1.astype(np.float32)  # prevent uint8 overflow
            merged_box = [x1, y1, x1 + w1, y1 + h1]

            for j in range(len(regions)):
                if i == j or j in merged_indices:
                    continue

                _, (x2, y2, w2, h2), color2 = regions[j]
                color2 = color2.astype(np.float32)  # prevent overflow

                # calc color difference
                color_diff = np.linalg.norm(color1 - color2)

                if color_diff < color_threshold:
                    # Merge bounding boxes
                    merged_box[0] = min(merged_box[0], x2)
                    merged_box[1] = min(merged_box[1], y2)
                    merged_box[2] = max(merged_box[2], x2 + w2)
                    merged_box[3] = max(merged_box[3], y2 + h2)
                    merged_indices.add(j)
                    has_merged = True

            merged_indices.add(i)
            x_min, y_min, x_max, y_max = map(int, merged_box)
            cropped = img[y_min:y_max, x_min:x_max]
            new_regions.append((cropped, (x_min, y_min, x_max - x_min, y_max - y_min), color1))

        regions = new_regions

    print("\n[INFO] Regions after merge:")
    for idx, (_, (x, y, w, h), color) in enumerate(regions, 1):
        print(f" Box {idx}: position=({x},{y},{w},{h}), effective_color={color.astype(int)}")

    return regions




def merge_same_color_boxes(img, char_regions, color_threshold):
    """
    Merge boxes with similar dominant colors.
    If a box has the same dominant color as its container, use its second dominant color instead.
    """
    boxes = [bbox for (_, bbox) in char_regions]
    
    # Step 1: Determine the effective color for each box
    box_colors = []
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        roi = img[y:y+h, x:x+w]
        dominant = get_dominant_color(roi)
        original_dominant = dominant.copy()
        
        # check if contained in any other box with same color
        for j in range(len(boxes)):
            if i == j:
                continue
            if is_contained(boxes[i], boxes[j]):
                xj, yj, wj, hj = boxes[j]
                roi_container = img[yj:yj+hj, xj:xj+wj]
                container_color = get_dominant_color(roi_container)
                
                # If same color as container, use second dominant color
                if np.linalg.norm(dominant - container_color) < color_threshold:
                    dominant = get_second_dominant_color(roi, dominant)
                    print(f"[DEBUG] Box {i+1} at ({x},{y},{w},{h}) is contained in Box {j+1}")
                    print(f"        Original color: {original_dominant.astype(int)}, Changed to: {dominant.astype(int)}")
                    break
        
        box_colors.append(dominant)
    
    # Step 2: Merge boxes with similar colors iteratively
    has_merged = True
    while has_merged:
        has_merged = False
        merged_indices = set()
        new_boxes = []
        new_colors = []
        
        for i in range(len(boxes)):
            if i in merged_indices:
                continue
            
            x1, y1, w1, h1 = boxes[i]
            color1 = box_colors[i]
            merged_box = [x1, y1, x1+w1, y1+h1]
            
            # Find all boxes with similar color to merge
            for j in range(i+1, len(boxes)):
                if j in merged_indices:
                    continue
                
                color2 = box_colors[j]
                color_diff = np.linalg.norm(color1 - color2)
                
                # Merge if colors are similar
                if color_diff < color_threshold:
                    x2, y2, w2, h2 = boxes[j]
                    merged_box[0] = min(merged_box[0], x2)
                    merged_box[1] = min(merged_box[1], y2)
                    merged_box[2] = max(merged_box[2], x2 + w2)
                    merged_box[3] = max(merged_box[3], y2 + h2)
                    merged_indices.add(j)
                    has_merged = True
            
            merged_indices.add(i)
            x_min, y_min, x_max, y_max = merged_box
            new_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
            new_colors.append(color1)
        
        boxes = new_boxes
        box_colors = new_colors
    
    # Step 3: Create merged regions
    merged_regions = []
    for idx, (x, y, w, h) in enumerate(boxes):
        cropped = img[y:y+h, x:x+w]
        color = box_colors[idx]  
        merged_regions.append((cropped, (x, y, w, h), color))

    # Final merge using effective colors
    res = check_and_merge_regions(img, merged_regions, color_threshold)
    
    return res


def segment_captcha_by_color(img_path, min_area=30, output_dir="test_char"):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {img_path}")
        
    h, w, _ = img.shape
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract target string and number of characters
    base_name = os.path.basename(img_path)
    match = re.match(r"([A-Za-z0-9]+)-", base_name)
    if not match:
        raise ValueError("Filename must contain target string before '-'")
    captcha_text = match.group(1)
    n_chars = len(captcha_text)
    print(f"[INFO] CAPTCHA text: '{captcha_text}' (expected {n_chars} characters)")

    # Apply KMeans with some extra clusters for noise
    data = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters = n_chars + 1, n_init = 10, random_state = 42)
    labels = kmeans.fit_predict(data)
    label_img = labels.reshape((h, w))

    pixel_counts = [np.sum(label_img == c) for c in range(n_chars + 1)]
    bg_cluster = np.argmax(pixel_counts)
    print(f"[INFO] Ignoring background cluster: {bg_cluster}")

    char_regions = []

    for cluster_idx in range(n_chars + 1):
        if cluster_idx == bg_cluster:
            continue

        mask = np.uint8(label_img == cluster_idx) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            if w_box * h_box >= min_area:
                cropped = img[y:y+h_box, x:x+w_box]
                char_regions.append((cropped, (x, y, w_box, h_box)))

    char_regions.sort(key=lambda c: c[1][0])

    print("[INFO] Dominant colors before merging:")
    for idx, (char_img, (x, y, w, h)) in enumerate(char_regions):
        color = get_dominant_color(char_img).astype(int)
        print(f"Box {idx+1}: position=({x},{y},{w},{h}), dominant_color={color}")

    # Merge same color boxes
    merged_regions = merge_same_color_boxes(img, char_regions, color_threshold=20)
    merged_regions.sort(key=lambda c: c[1][0])

    # Save cropped images
    for i, (char_img, (x, y, w_box, h_box), color) in enumerate(merged_regions):
        # Convert color to integers and format as RGB string
        color_str = "_".join(map(str, color.astype(int)))
        filename = f"char_{i+1}_x{x}_color_{color_str}.png"
        cv2.imwrite(os.path.join(output_dir, filename), char_img)

    # Visualization
    vis = img.copy()
    for (_, (x, y, w_box, h_box), color) in merged_regions:
        cv2.rectangle(vis, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
    plt.figure(figsize=(10,4))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f"Segmented {len(merged_regions)} Characters (after merge)")
    plt.axis("off")
    plt.show()

    return merged_regions


if __name__ == "__main__":
    test_path = "fa-0.png"
    chars = segment_captcha_by_color(test_path)
