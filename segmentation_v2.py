import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def get_dominant_clusters(image, n_clusters, threshold,
                          spatial_weight_x=0.5, spatial_weight_y=0.5, contour_weight=50):
    """
    Apply KMeans clustering on all non-background pixels.
    Combine RGB, spatial distances, and contour label as features.
    Pixels belonging to different connected regions (contours) are penalized.
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    # Foreground mask
    mask_foreground = np.any(img_rgb < threshold, axis=2)
    ys, xs = np.where(mask_foreground)
    pixels = img_rgb[mask_foreground].reshape(-1, 3)

    if len(pixels) < n_clusters:
        return None, None

    # --- Connected component labeling for contour penalty ---
    mask_uint8 = (mask_foreground.astype(np.uint8)) * 255
    num_labels, labels = cv2.connectedComponents(mask_uint8, connectivity=8)
    contour_labels = labels[ys, xs].astype(np.float32)

    # Normalize coordinates to 0–255 range
    xs_scaled = (xs / w) * 255
    ys_scaled = (ys / h) * 255

    # --- Construct feature vector ---
    features = np.column_stack([
        pixels,
        xs_scaled * spatial_weight_x,
        ys_scaled * spatial_weight_y,
        contour_labels * contour_weight
    ])

    # Run KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(features)
    labels = kmeans.labels_

    # Map back to full image
    label_map = np.full(mask_foreground.shape, -1, dtype=int)
    label_map[mask_foreground] = labels

    clustered_masks = []
    for i in range(n_clusters):
        cluster_mask = (label_map == i).astype(np.uint8) * 255
        clustered_masks.append(cluster_mask)

    return clustered_masks, mask_foreground

# Method used in the pipeline notebook
def segment_characters(image_path, min_pixels=3, top_components=5,
                                spatial_weight_x=1.5, spatial_weight_y=0.4, contour_weight=100):
    """
    Segment CAPTCHA characters using RGB + (x, y)-weighted + contour-aware KMeans.
    Keep top components and delete the smallest cluster (fewest pixels).
    """
    filename = Path(image_path).stem
    chars_part = filename.split("-")[0]
    n_clusters = len(chars_part) + 1
    print(f"[INFO] Detected {len(chars_part)} characters — using {n_clusters} KMeans clusters.")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    threshold = 250
    max_threshold = 250
    step = 10
    valid_bounding_boxes = []

    # Try until enough clusters found
    while threshold <= max_threshold:
        clustered_masks, mask_foreground = get_dominant_clusters(
            image, n_clusters, threshold,
            spatial_weight_x=spatial_weight_x, spatial_weight_y=spatial_weight_y, contour_weight=contour_weight
        )
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

    cleaned_clusters = []
    cluster_x_positions = []
    cluster_pixel_counts = []

    # --- Step 1: Clean clusters ---
    for i, (x, y, w, h, mask) in enumerate(valid_bounding_boxes):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            sorted_idx = np.argsort(areas)[::-1]

            keep_mask = np.zeros_like(mask)
            count_kept = 0
            first_center = None
            distance_threshold = 25

            for idx2 in sorted_idx:
                if count_kept >= top_components:
                    break
                area = areas[idx2]
                if area < min_pixels:
                    continue

                label = idx2 + 1
                cx, cy = centroids[label]

                if first_center is None:
                    first_center = (cx, cy)
                    keep_mask[labels == label] = 255
                    count_kept += 1
                else:
                    dist = np.sqrt((cx - first_center[0]) ** 2 + (cy - first_center[1]) ** 2)
                    if dist < distance_threshold:
                        keep_mask[labels == label] = 255
                        count_kept += 1

            cleaned_mask = keep_mask
        else:
            cleaned_mask = mask.copy()

        ys, xs = np.where(cleaned_mask > 0)
        x_leftmost = np.min(xs) if len(xs) > 0 else 99999
        pixel_count = np.sum(cleaned_mask > 0)

        cluster_pixel_counts.append(pixel_count)
        cleaned_clusters.append((x, y, w, h, cleaned_mask))
        cluster_x_positions.append(x_leftmost)

    # --- Step 2: Identify smallest cluster (deleted one) ---
    min_idx = int(np.argmin(cluster_pixel_counts))
    min_pixels_count = cluster_pixel_counts[min_idx]
    print(f"[INFO] Cluster {min_idx} marked as deleted (fewest pixels: {min_pixels_count}).")

    # --- Step 3: Sort clusters left-to-right ---
    sorted_indices = np.argsort(cluster_x_positions)

    # Extract character images (excluding the noise cluster)
    char_images = []
    char_labels = []

    kept_idx = 0  # index among kept clusters (left-to-right, skipping noise)
    for i in sorted_indices:
        if i == min_idx:  # Skip noise cluster
            continue

        x, y, w, h, cleaned_mask = cleaned_clusters[i]
        roi = image[y:y + h, x:x + w]
        mask_roi = cleaned_mask[y:y + h, x:x + w]
        char_img = np.full_like(roi, 255)  # White background
        char_img[mask_roi == 255] = roi[mask_roi == 255]

        # Crop tightly to foreground
        gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        ys_fg, xs_fg = np.where(gray < 250)  # find non-white pixels
        if len(xs_fg) > 0 and len(ys_fg) > 0:
            x_min_fg, x_max_fg = np.min(xs_fg), np.max(xs_fg)
            y_min_fg, y_max_fg = np.min(ys_fg), np.max(ys_fg)
            char_img = char_img[y_min_fg:y_max_fg + 1, x_min_fg:x_max_fg + 1]

        char_images.append(char_img)

        # Determine label based on kept index (align with GT positions)
        if kept_idx < len(chars_part):
            char_labels.append(chars_part[kept_idx])
        else:
            char_labels.append('')  # out of range safeguard
        kept_idx += 1

    return char_images, char_labels

def segment_and_save_characters(image_path, min_pixels=3, top_components=5,
                                spatial_weight_x=0.5, spatial_weight_y=0.5, contour_weight=50):
    """
    Segment CAPTCHA characters using RGB + (x, y)-weighted + contour-aware KMeans.
    Keep top components, delete the smallest cluster, and save debug images.
    """
    filename = Path(image_path).stem
    chars_part = filename.split("-")[0]
    n_clusters = len(chars_part) + 1
    print(f"[INFO] Detected {len(chars_part)} characters — using {n_clusters} KMeans clusters.")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    threshold = 250
    max_threshold = 250
    step = 10
    valid_bounding_boxes = []

    # Try until enough clusters found
    while threshold <= max_threshold:
        clustered_masks, mask_foreground = get_dominant_clusters(
            image, n_clusters, threshold,
            spatial_weight_x=spatial_weight_x, spatial_weight_y=spatial_weight_y, contour_weight=contour_weight
        )
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

    cleaned_clusters = []
    cluster_x_positions = []
    cluster_pixel_counts = []

    # --- Step 1: Clean clusters and save debug image ---
    for i, (x, y, w, h, mask) in enumerate(valid_bounding_boxes):
        # --- Debug: save raw KMeans result before cropping ---
        debug_img = np.full_like(image, 255)
        debug_img[mask == 255] = image[mask == 255]
        debug_save_path = output_dir / f"debug_char_{i}.png"
        cv2.imwrite(debug_save_path, debug_img)
        print(f"[DEBUG SAVED] {debug_save_path}")

        # --- Clean cluster ---
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            sorted_idx = np.argsort(areas)[::-1]

            keep_mask = np.zeros_like(mask)
            count_kept = 0
            first_center = None
            distance_threshold = 25

            for idx2 in sorted_idx:
                if count_kept >= top_components:
                    break
                area = areas[idx2]
                if area < min_pixels:
                    continue

                label = idx2 + 1
                cx, cy = centroids[label]

                if first_center is None:
                    first_center = (cx, cy)
                    keep_mask[labels == label] = 255
                    count_kept += 1
                else:
                    dist = np.sqrt((cx - first_center[0]) ** 2 + (cy - first_center[1]) ** 2)
                    if dist < distance_threshold:
                        keep_mask[labels == label] = 255
                        count_kept += 1

            cleaned_mask = keep_mask
        else:
            cleaned_mask = mask.copy()

        ys, xs = np.where(cleaned_mask > 0)
        x_leftmost = np.min(xs) if len(xs) > 0 else 99999
        pixel_count = np.sum(cleaned_mask > 0)

        cluster_pixel_counts.append(pixel_count)
        cleaned_clusters.append((x, y, w, h, cleaned_mask))
        cluster_x_positions.append(x_leftmost)

    # --- Step 2: Identify smallest cluster (deleted one) ---
    min_idx = int(np.argmin(cluster_pixel_counts))
    min_pixels_count = cluster_pixel_counts[min_idx]
    print(f"[INFO] Cluster {min_idx} marked as deleted (fewest pixels: {min_pixels_count}).")

    # --- Step 3: Sort clusters left-to-right ---
    sorted_indices = np.argsort(cluster_x_positions)

    # --- Step 4: Save final clusters ---
    for order, i in enumerate(sorted_indices):
        x, y, w, h, cleaned_mask = cleaned_clusters[i]
        color = (0, 0, 255) if i == min_idx else (0, 255, 0)
        cv2.rectangle(output_vis, (x, y), (x + w, y + h), color, 2)

        roi = image[y:y + h, x:x + w]
        mask_roi = cleaned_mask[y:y + h, x:x + w]
        char_img = np.full_like(roi, 255)
        char_img[mask_roi == 255] = roi[mask_roi == 255]

        # Tight crop
        gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(gray < 250)
        if len(xs) > 0 and len(ys) > 0:
            x_min, x_max = np.min(xs), np.max(xs)
            y_min, y_max = np.min(ys), np.max(ys)
            char_img = char_img[y_min:y_max + 1, x_min:x_max + 1]

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
    image_path = "2uba9hkt-0.png"
    segment_and_save_characters(image_path, spatial_weight_x=4.003, spatial_weight_y=0.449, contour_weight=238.313)


if __name__ == "__main__":
    main()
