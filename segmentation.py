import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def get_dominant_clusters(image, n_clusters, threshold):
    """
    Apply KMeans clustering on all non-background pixels
    for a given threshold. Assign every foreground pixel to a cluster.
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Foreground: any pixel darker than threshold
    mask_foreground = np.any(img_rgb < threshold, axis=2)
    pixels = img_rgb[mask_foreground].reshape(-1, 3)

    if len(pixels) < n_clusters:
        return None, None  # Not enough pixels to form clusters

    # Run KMeans on all foreground pixels
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(pixels)
    labels = kmeans.labels_

    # Map labels back to full image
    label_map = np.full(mask_foreground.shape, -1, dtype=int)
    label_map[mask_foreground] = labels

    # Convert to individual masks
    clustered_masks = []
    for i in range(n_clusters):
        cluster_mask = (label_map == i).astype(np.uint8) * 255
        clustered_masks.append(cluster_mask)

    return clustered_masks, mask_foreground


def segment_and_save_characters(image_path, min_pixels=3, top_components=5):
    """
    Segment CAPTCHA characters using adaptive KMeans thresholding.
    Clean each cluster by keeping the top N (default 5) largest connected components,
    skipping any component that is too far from the main one.
    Then find and mark the one with the smallest pixel count as deleted.
    Save cropped character images sorted left-to-right.
    """
    filename = Path(image_path).stem
    chars_part = filename.split("-")[0]
    n_clusters = len(chars_part) + 1
    print(f"[INFO] Detected {n_clusters} characters — using {n_clusters} KMeans clusters.")

    image = cv2.imread(image_path)
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
        print("[ERROR] Failed to find enough clusters even at max threshold.")
        return

    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    output_vis = image.copy()

    cluster_pixel_counts = []
    cleaned_clusters = []
    cluster_x_positions = []  # store leftmost pixel x position of largest component

    # --- Step 1: Clean clusters (keep top N largest connected parts) ---
    for i, (x, y, w, h, mask) in enumerate(valid_bounding_boxes):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            sorted_idx = np.argsort(areas)[::-1]  # largest → smallest

            keep_mask = np.zeros_like(mask)
            count_kept = 0
            first_center = None
            distance_threshold = 25  # tune for your CAPTCHA size

            for idx2 in sorted_idx:
                if count_kept >= top_components:
                    break
                area = areas[idx2]
                if area < min_pixels:
                    continue

                label = idx2 + 1
                cx, cy = centroids[label]

                if first_center is None:
                    # Always keep the first (largest) component
                    first_center = (cx, cy)
                    keep_mask[labels == label] = 255
                    count_kept += 1
                else:
                    # Compute Euclidean distance to the first component
                    dist = np.sqrt((cx - first_center[0]) ** 2 + (cy - first_center[1]) ** 2)
                    if dist < distance_threshold:
                        keep_mask[labels == label] = 255
                        count_kept += 1

            cleaned_mask = keep_mask
        else:
            cleaned_mask = mask.copy()

        # record x position of largest connected component
        ys, xs = np.where(cleaned_mask > 0)
        if len(xs) > 0:
            x_leftmost = np.min(xs)
        else:
            x_leftmost = 99999  # fallback in case mask empty

        pixel_count = np.sum(cleaned_mask > 0)
        cluster_pixel_counts.append(pixel_count)
        cleaned_clusters.append((x, y, w, h, cleaned_mask))
        cluster_x_positions.append(x_leftmost)

    # --- Step 2: Identify cluster with fewest pixels ---
    min_idx = int(np.argmin(cluster_pixel_counts))
    min_pixels_count = cluster_pixel_counts[min_idx]
    print(f"[INFO] Cluster {min_idx} marked as deleted (fewest pixels: {min_pixels_count}).")

    # --- Step 3: Sort by leftmost pixel (x position of biggest component) ---
    sorted_indices = np.argsort(cluster_x_positions)

    # --- Step 4: Save all clusters ---
    for order, i in enumerate(sorted_indices):
        x, y, w, h, cleaned_mask = cleaned_clusters[i]
        color = (0, 255, 0) if i != min_idx else (0, 0, 255)  # Red for deleted
        cv2.rectangle(output_vis, (x, y), (x + w, y + h), color, 2)

        roi = image[y:y + h, x:x + w]
        mask_roi = cleaned_mask[y:y + h, x:x + w]
        char_img = np.full_like(roi, 255)
        char_img[mask_roi == 255] = roi[mask_roi == 255]

        # --- Crop tightly to foreground ---
        gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(gray < 250)  # find non-white pixels
        if len(xs) > 0 and len(ys) > 0:
            x_min, x_max = np.min(xs), np.max(xs)
            y_min, y_max = np.min(ys), np.max(ys)
            char_img = char_img[y_min:y_max + 1, x_min:x_max + 1]

        # --- Save ---
        if i == min_idx:
            save_path = output_dir / f"deleted_char_pixels_{cluster_pixel_counts[i]}.png"
        else:
            save_path = output_dir / f"char_{order}_pixels_{cluster_pixel_counts[i]}.png"

        cv2.imwrite(save_path, char_img)
        print(f"[SAVED] {save_path}")

    # --- Visualization ---
    plt.figure(figsize=(10, 4))
    plt.imshow(cv2.cvtColor(output_vis, cv2.COLOR_BGR2RGB))
    plt.title(f"Segmented CAPTCHA (Threshold={threshold}) — Deleted cluster {min_idx}")
    plt.axis("off")
    plt.show()


def main():
    image_path = "1cfu8x-0.png"
    segment_and_save_characters(image_path)


if __name__ == "__main__":
    main()


