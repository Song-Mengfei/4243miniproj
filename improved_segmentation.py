"""
Improved CAPTCHA Character Segmentation
========================================
Multiple approaches for robust character segmentation:
1. Connected Components Analysis (CCA) - Best for non-touching characters
2. Projection-based segmentation - Works well for aligned text
3. Contour-based segmentation - Handles various shapes
4. Hybrid approach combining multiple methods
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class CaptchaSegmenter:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.gray.shape
        
    def preprocess(self, method='otsu'):
        """
        Preprocess image: denoise, binarize
        Methods: 'otsu', 'adaptive', 'threshold'
        """
        # Denoise
        denoised = cv2.fastNlMeansDenoising(self.gray, h=10)
        
        if method == 'otsu':
            # Otsu's binarization - automatic threshold
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            # Adaptive thresholding - good for varying lighting
            binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
        else:
            # Simple threshold
            _, binary = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY_INV)
        
        return binary

    def _foreground_mask(self, threshold=230):
        """Return binary mask of non-background pixels (255=fg)."""
        img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # Any channel darker than threshold considered foreground
        mask = (img_rgb < threshold).any(axis=2).astype(np.uint8) * 255
        return mask

    def segment_color_kmeans(self, expected_chars=None, threshold=230, subsample=0.3,
                              max_iter=20, random_seed=42, max_k=8):
        """
        Method 5: Color K-Means + Spatial Refinement
        Best for: Multi-colored characters with overlap

        Steps:
        - Foreground mask by near-white threshold
        - K-means in color space (Lab) using subsampling for speed
        - Assign all fg pixels to nearest center
        - For each cluster, split by connected components; if too wide, use intra-cluster watershed
        """
        fg_mask = self._foreground_mask(threshold)
        if fg_mask.sum() == 0:
            return [], fg_mask

        # Convert to Lab color space (better perceptual clustering)
        img_lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2Lab)

        ys, xs = np.where(fg_mask > 0)
        pixels = img_lab[ys, xs].astype(np.float32)

        # Subsample for fitting
        if len(pixels) > 0 and subsample < 1.0:
            step = max(1, int(1.0 / subsample))
            pixels_sample = pixels[::step]
        else:
            pixels_sample = pixels

        # Decide K
        if expected_chars is None or expected_chars < 2:
            k = min(max_k, max(2, int(self.width / 40)))
        else:
            k = int(expected_chars)

        # Use OpenCV kmeans for speed
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 1.0)
        flags = cv2.KMEANS_PP_CENTERS
        compactness, labels_sample, centers = cv2.kmeans(pixels_sample, k, None, criteria, 3, flags)

        centers = centers.reshape(k, 3)

        # Assign all foreground pixels to nearest center
        # Compute distances in a vectorized way
        diff = pixels[:, None, :] - centers[None, :, :]  # [N, K, 3]
        dists = np.sum(diff * diff, axis=2)  # [N, K]
        labels_full = np.argmin(dists, axis=1).astype(np.int32)

        # Create full-image label map for fg pixels
        label_map = np.full((self.height, self.width), -1, dtype=np.int32)
        label_map[ys, xs] = labels_full

        # For each cluster, extract connected components
        bounding_boxes = []
        binary_debug = fg_mask.copy()

        for ci in range(k):
            cluster_mask = (label_map == ci).astype(np.uint8) * 255
            if cluster_mask.sum() == 0:
                continue

            # Connected components in this cluster
            num, labels, stats, _ = cv2.connectedComponentsWithStats(cluster_mask, connectivity=8)
            for i in range(1, num):
                x, y, w, h, area = stats[i]
                if area < 20:
                    continue

                comp_mask = (labels == i).astype(np.uint8) * 255

                # If the component is too wide, try splitting inside this cluster via watershed
                aspect = w / (h + 1e-6)
                if aspect > 1.8 and area > 100:
                    sub_boxes = self._watershed_on_mask(comp_mask, x, y)
                    if sub_boxes:
                        bounding_boxes.extend(sub_boxes)
                        continue

                bounding_boxes.append((x, y, w, h, comp_mask, int(area)))

        return bounding_boxes, binary_debug

    def _watershed_on_mask(self, comp_mask, x_off, y_off):
        """Apply watershed inside a single component mask; return sub-boxes.
        comp_mask: full-image-sized binary mask of one component.
        x_off, y_off: top-left of component bounding rect.
        """
        ys, xs = np.where(comp_mask > 0)
        if xs.size == 0:
            return []
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        w, h = x_max - x_min + 1, y_max - y_min + 1

        roi = comp_mask[y_min:y_max + 1, x_min:x_max + 1]
        if roi.sum() == 0:
            return []

        # Distance transform and markers
        dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, cv2.THRESH_BINARY)
        sure_fg = sure_fg.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        sure_bg = cv2.dilate(roi, kernel, iterations=1)
        unknown = cv2.subtract(sure_bg, sure_fg)
        num_markers, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Build a color ROI from the original image for watershed
        color_roi = self.image[y_min:y_max + 1, x_min:x_max + 1].copy()
        cv2.watershed(color_roi, markers)

        boxes = []
        labels_ws = markers.copy()
        for label in np.unique(labels_ws):
            if label <= 1:
                continue
            mask_label = (labels_ws == label).astype(np.uint8) * 255
            if mask_label.sum() < 20:
                continue
            ys2, xs2 = np.where(mask_label > 0)
            if xs2.size == 0:
                continue
            xx_min, xx_max = xs2.min(), xs2.max()
            yy_min, yy_max = ys2.min(), ys2.max()
            ww, hh = xx_max - xx_min + 1, yy_max - yy_min + 1

            # Place into full-image coordinates
            full_mask = np.zeros_like(comp_mask)
            full_mask[y_min + yy_min:y_min + yy_max + 1, x_min + xx_min:x_min + xx_max + 1] = \
                mask_label[yy_min:yy_max + 1, xx_min:xx_max + 1]
            boxes.append((x_min + xx_min, y_min + yy_min, ww, hh, full_mask, int(full_mask.sum() // 255)))

        return boxes
    
    def segment_connected_components(self, min_area=50, max_area=5000):
        """
        Method 1: Connected Components Analysis (CCA)
        Best for: Non-overlapping characters with clear separation
        """
        binary = self.preprocess('otsu')
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        bounding_boxes = []
        for i in range(1, num_labels):  # Skip background (label 0)
            x, y, w, h, area = stats[i]
            
            # Filter by area to remove noise
            if min_area < area < max_area:
                # Extract character with mask
                mask = (labels == i).astype(np.uint8) * 255
                bounding_boxes.append((x, y, w, h, mask, area))
        
        return bounding_boxes, binary
    
    def segment_watershed(self, min_area=50, max_area=5000, dist_thresh=0.3):
        """
        Method 4: Watershed-based Segmentation
        Best for: Touching/overlapping characters that need splitting
        """
        # Start from a clean binary (foreground=255)
        binary = self.preprocess('otsu')

        # Morphology to reduce bridges and small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # Distance transform to find sure foreground
        dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, dist_thresh * dist.max(), 255, cv2.THRESH_BINARY)
        sure_fg = sure_fg.astype(np.uint8)

        # Sure background
        sure_bg = cv2.dilate(opened, kernel, iterations=2)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        num_markers, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1  # make sure background is 1 instead of 0
        markers[unknown == 255] = 0

        # Apply watershed
        img_color = self.image.copy()
        cv2.watershed(img_color, markers)

        # markers == -1 are boundaries
        labels = markers.copy()

        bounding_boxes = []
        for label in np.unique(labels):
            if label <= 1:
                continue  # skip background and boundary
            component = (labels == label).astype(np.uint8) * 255
            ys, xs = np.where(component > 0)
            if xs.size == 0:
                continue
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            w, h = x_max - x_min + 1, y_max - y_min + 1

            area = int(component.sum() // 255)
            if min_area < area < max_area:
                # Build full-image mask for this component
                mask = np.zeros_like(binary)
                mask[labels == label] = 255
                bounding_boxes.append((x_min, y_min, w, h, mask, area))

        return bounding_boxes, binary

    def segment_projection(self, threshold_ratio=0.1):
        """
        Method 2: Vertical Projection Segmentation
        Best for: Horizontally aligned text with consistent spacing
        """
        binary = self.preprocess('otsu')
        
        # Calculate vertical projection (sum of pixels in each column)
        vertical_proj = np.sum(binary, axis=0) / 255
        
        # Find valleys (gaps between characters)
        threshold = np.max(vertical_proj) * threshold_ratio
        in_char = False
        segments = []
        start = 0
        
        for i, val in enumerate(vertical_proj):
            if not in_char and val > threshold:
                start = i
                in_char = True
            elif in_char and val <= threshold:
                if i - start > 5:  # Minimum width
                    segments.append((start, i))
                in_char = False
        
        # Last segment
        if in_char:
            segments.append((start, len(vertical_proj)))
        
        bounding_boxes = []
        for x_start, x_end in segments:
            # Find vertical bounds
            col_slice = binary[:, x_start:x_end]
            ys = np.where(np.any(col_slice > 0, axis=1))[0]
            
            if len(ys) > 0:
                y_start, y_end = ys[0], ys[-1] + 1
                w, h = x_end - x_start, y_end - y_start
                
                # Create mask
                mask = np.zeros_like(binary)
                mask[y_start:y_end, x_start:x_end] = binary[y_start:y_end, x_start:x_end]
                
                bounding_boxes.append((x_start, y_start, w, h, mask, w * h))
        
        return bounding_boxes, binary
    
    def segment_contours(self, min_area=50, max_area=5000):
        """
        Method 3: Contour-based Segmentation
        Best for: Various character shapes, handles rotation
        """
        binary = self.preprocess('adaptive')
        
        # Morphological operations to connect broken characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bounding_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Create mask for this contour
                mask = np.zeros_like(binary)
                cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
                
                bounding_boxes.append((x, y, w, h, mask, area))
        
        return bounding_boxes, binary
    
    def segment_hybrid(self, expected_chars=None):
        """
        Method 4: Hybrid Approach
        Tries multiple methods and picks the best result
        """
        methods = [
            ('Connected Components', self.segment_connected_components),
            ('Projection', self.segment_projection),
            ('Contours', self.segment_contours),
            ('Color KMeans', self.segment_color_kmeans),
            ('Watershed', self.segment_watershed)
        ]
        
        results = []
        for name, method in methods:
            try:
                boxes, binary = method()
                if boxes:
                    results.append((name, boxes, binary, len(boxes)))
                    print(f"[{name}] Found {len(boxes)} segments")
            except Exception as e:
                print(f"[{name}] Failed: {e}")
        
        if not results:
            print("[ERROR] All methods failed!")
            return [], None
        
        # Choose best method
        if expected_chars:
            # Prefer method closest to expected number
            results.sort(key=lambda x: abs(x[3] - expected_chars))
        else:
            # Prefer method with most segments (likely found all chars)
            results.sort(key=lambda x: -x[3])
        
        best_method, boxes, binary, count = results[0]
        print(f"[BEST] Using {best_method} method with {count} segments")
        
        return boxes, binary
    
    def merge_overlapping_boxes(self, bounding_boxes, overlap_threshold=0.5):
        """
        Merge bounding boxes that significantly overlap
        """
        if len(bounding_boxes) <= 1:
            return bounding_boxes
        
        # Sort by x coordinate
        boxes = sorted(bounding_boxes, key=lambda b: b[0])
        merged = []
        
        current = list(boxes[0])
        
        for next_box in boxes[1:]:
            x1, y1, w1, h1 = current[0], current[1], current[2], current[3]
            x2, y2, w2, h2 = next_box[0], next_box[1], next_box[2], next_box[3]
            
            # Check horizontal overlap
            overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            min_width = min(w1, w2)
            
            if overlap > min_width * overlap_threshold:
                # Merge boxes
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                x_max = max(x1 + w1, x2 + w2)
                y_max = max(y1 + h1, y2 + h2)
                
                # Combine masks
                mask = current[4] | next_box[4]
                
                current = [x_min, y_min, x_max - x_min, y_max - y_min, mask, 
                          (x_max - x_min) * (y_max - y_min)]
            else:
                merged.append(tuple(current))
                current = list(next_box)
        
        merged.append(tuple(current))
        return merged
    
    def _split_box_by_projection(self, box, expected_pieces=None):
        """
        Split a single box using vertical projection minima within its mask.
        Returns a list of sub-box tuples like input.
        """
        x, y, w, h, mask, area = box
        roi_mask = mask[y:y + h, x:x + w]

        # Compute vertical projection within ROI
        proj = (roi_mask > 0).sum(axis=0).astype(np.float32)
        if proj.max() == 0:
            return [box]

        # Smooth projection to find valleys
        proj_sm = cv2.GaussianBlur(proj.reshape(1, -1), (1, 9), 0).flatten()

        # Candidate cut positions where projection small
        thresh = 0.35 * proj_sm.max()
        candidates = [i for i in range(2, len(proj_sm) - 2) if proj_sm[i] < thresh]

        # Deduplicate nearby candidates (min gap)
        min_gap = max(3, w // 20)
        cuts = []
        for c in candidates:
            if not cuts or c - cuts[-1] >= min_gap:
                cuts.append(c)

        # If expected pieces known, pick K-1 best valleys (lowest projection)
        if expected_pieces and expected_pieces > 1 and len(cuts) >= expected_pieces - 1:
            # Rank all positions by projection value ascending, enforce spacing
            ranked = sorted(range(len(proj_sm)), key=lambda i: proj_sm[i])
            selected = []
            for idx in ranked:
                if idx <= 2 or idx >= len(proj_sm) - 3:
                    continue
                if proj_sm[idx] >= thresh:
                    continue
                if all(abs(idx - s) >= min_gap for s in selected):
                    selected.append(idx)
                if len(selected) == expected_pieces - 1:
                    break
            cuts = sorted(selected) if selected else cuts

        if not cuts:
            return [box]

        # Build sub-boxes using cuts
        segments = []
        last = 0
        for c in cuts + [w]:
            sub_w = c - last
            if sub_w <= 3:
                last = c
                continue
            sub_mask = np.zeros_like(mask)
            sub_mask[y:y + h, x + last:x + c] = roi_mask[:, last:c]
            if np.any(sub_mask):
                # Compute tight bbox inside this sub region
                ys, xs = np.where(sub_mask > 0)
                if xs.size:
                    x_min, x_max = xs.min(), xs.max()
                    y_min, y_max = ys.min(), ys.max()
                    sub_x, sub_y = int(x_min), int(y_min)
                    sub_w2, sub_h2 = int(x_max - x_min + 1), int(y_max - y_min + 1)
                    sub_area = int(sub_mask.sum() // 255)
                    segments.append((sub_x, sub_y, sub_w2, sub_h2, sub_mask, sub_area))
            last = c

        return segments if segments else [box]

    def split_wide_boxes(self, bounding_boxes, max_aspect_ratio=1.5, expected_chars=None):
        """
        Split boxes that are too wide (likely merged characters) using
        intra-box projection; fall back to half split.
        """
        split_boxes = []
        
        for box in bounding_boxes:
            x, y, w, h, mask, area = box
            aspect_ratio = w / h if h > 0 else 0
            
            if aspect_ratio > max_aspect_ratio:
                # Try smart split by projection
                pieces_hint = None
                if expected_chars is not None and expected_chars > 0:
                    # rough guess: based on average width
                    avg_w = max(1, int(np.mean([b[2] for b in bounding_boxes])))
                    pieces_hint = min(4, max(2, w // max(8, avg_w)))

                subs = self._split_box_by_projection(box, expected_pieces=pieces_hint)
                if len(subs) > 1:
                    split_boxes.extend(subs)
                else:
                    # Fallback: split into two halves
                    mid = x + w // 2
                    mask_left = mask.copy(); mask_left[:, mid:] = 0
                    mask_right = mask.copy(); mask_right[:, :mid] = 0
                    if np.any(mask_left):
                        split_boxes.append((x, y, w // 2, h, mask_left, area // 2))
                    if np.any(mask_right):
                        split_boxes.append((mid, y, w - w // 2, h, mask_right, area // 2))
            else:
                split_boxes.append(box)
        
        return split_boxes

    def _merge_two_boxes(self, a, b):
        x1, y1, w1, h1, m1, _ = a
        x2, y2, w2, h2, m2, _ = b
        x_min = min(x1, x2); y_min = min(y1, y2)
        x_max = max(x1 + w1, x2 + w2); y_max = max(y1 + h1, y2 + h2)
        mask = m1 | m2
        area = int(mask.sum() // 255)
        return (x_min, y_min, x_max - x_min, y_max - y_min, mask, area)

    def _adjust_to_expected(self, boxes, expected_chars):
        """
        Try to adjust number of boxes to expected count by splitting/merging.
        Conservative heuristics to avoid over-fragmentation.
        """
        if expected_chars is None or expected_chars <= 0:
            return boxes

        boxes = sorted(boxes, key=lambda b: b[0])

        # If too few, split widest boxes iteratively
        attempts = 0
        while len(boxes) < expected_chars and attempts < 8:
            widths = [b[2] for b in boxes]
            idx = int(np.argmax(widths))
            subs = self._split_box_by_projection(boxes[idx], expected_pieces=2)
            if len(subs) > 1:
                boxes = boxes[:idx] + subs + boxes[idx + 1:]
                boxes = sorted(boxes, key=lambda b: b[0])
            else:
                break
            attempts += 1

        # If too many, merge closest neighbors by gap
        attempts = 0
        while len(boxes) > expected_chars and attempts < 8 and len(boxes) > 1:
            boxes = sorted(boxes, key=lambda b: b[0])
            # compute gaps between neighbors
            gaps = []
            for i in range(len(boxes) - 1):
                a = boxes[i]; b = boxes[i + 1]
                gap = max(0, b[0] - (a[0] + a[2]))
                gaps.append((gap, i))
            if not gaps:
                break
            _, i = min(gaps, key=lambda t: t[0])
            merged = self._merge_two_boxes(boxes[i], boxes[i + 1])
            boxes = boxes[:i] + [merged] + boxes[i + 2:]
            attempts += 1

        return boxes
    
    def save_segments(self, bounding_boxes, output_dir='test_char', 
                     filename_prefix=None, visualize=True):
        """
        Save segmented characters and visualize results
        """
        if not bounding_boxes:
            print("[ERROR] No segments to save!")
            return
        
        # Sort left to right
        bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if filename_prefix is None:
            filename_prefix = Path(self.image_path).stem
        
        output_vis = self.image.copy()
        saved_chars = []
        
        for i, (x, y, w, h, mask, area) in enumerate(bounding_boxes):
            # Draw bounding box
            cv2.rectangle(output_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output_vis, str(i), (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Extract character with white background
            roi = self.image[y:y + h, x:x + w]
            mask_roi = mask[y:y + h, x:x + w]
            
            char_img = np.full_like(roi, 255)  # White background
            char_img[mask_roi > 0] = roi[mask_roi > 0]
            
            # Save character
            save_path = output_path / f"{filename_prefix}_char_{i}.png"
            cv2.imwrite(str(save_path), char_img)
            saved_chars.append(str(save_path))
            print(f"[SAVED] {save_path} (size: {w}x{h}, area: {area})")
        
        # Save visualization
        vis_path = output_path / f"{filename_prefix}_segmented.png"
        cv2.imwrite(str(vis_path), output_vis)
        print(f"[SAVED] Visualization: {vis_path}")
        
        if visualize:
            self._visualize(bounding_boxes, output_vis)
        
        return saved_chars
    
    def _visualize(self, bounding_boxes, output_vis):
        """Show segmentation results"""
        fig, axes = plt.subplots(1, min(len(bounding_boxes) + 1, 8), 
                                figsize=(15, 3))
        
        if len(bounding_boxes) == 0:
            return
        
        if len(bounding_boxes) == 1:
            axes = [axes]
        
        # Show original with boxes
        if len(axes) > 0:
            axes[0].imshow(cv2.cvtColor(output_vis, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Segmented')
            axes[0].axis('off')
        
        # Show individual characters
        for i, (x, y, w, h, mask, _) in enumerate(bounding_boxes[:7]):
            if i + 1 < len(axes):
                roi = self.image[y:y + h, x:x + w]
                mask_roi = mask[y:y + h, x:x + w]
                char_img = np.full_like(roi, 255)
                char_img[mask_roi > 0] = roi[mask_roi > 0]
                
                axes[i + 1].imshow(cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB))
                axes[i + 1].set_title(f'Char {i}')
                axes[i + 1].axis('off')
        
        plt.tight_layout()
        plt.show()


def segment_captcha(image_path: str, method='hybrid', expected_chars=None,
                   output_dir='test_char', visualize=True):
    """
    Main function to segment CAPTCHA characters
    
    Parameters:
    -----------
    image_path : str
        Path to CAPTCHA image
    method : str
        'hybrid' (default), 'cca', 'projection', 'contours'
    expected_chars : int, optional
        Expected number of characters (from filename or known)
    output_dir : str
        Directory to save segmented characters
    visualize : bool
        Show matplotlib visualization
    
    Returns:
    --------
    List of saved character image paths
    """
    segmenter = CaptchaSegmenter(image_path)
    
    # Try to infer expected characters from filename
    if expected_chars is None:
        filename = Path(image_path).stem
        if '-' in filename:
            chars_part = filename.split('-')[0]
            expected_chars = len(chars_part)
            print(f"[INFO] Inferred {expected_chars} characters from filename")
    
    # Segment based on method
    if method == 'hybrid':
        boxes, binary = segmenter.segment_hybrid(expected_chars)
    elif method == 'cca':
        boxes, binary = segmenter.segment_connected_components()
    elif method == 'projection':
        boxes, binary = segmenter.segment_projection()
    elif method == 'contours':
        boxes, binary = segmenter.segment_contours()
    elif method == 'watershed':
        boxes, binary = segmenter.segment_watershed()
    elif method == 'color_kmeans':
        boxes, binary = segmenter.segment_color_kmeans(expected_chars=expected_chars)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if not boxes:
        print("[ERROR] No segments found!")
        return []
    
    # Post-processing
    print(f"[INFO] Found {len(boxes)} initial segments")
    
    # Split wide boxes (merged characters)
    boxes = segmenter.split_wide_boxes(boxes, expected_chars=expected_chars)
    print(f"[INFO] After splitting: {len(boxes)} segments")
    
    # Merge overlapping boxes
    boxes = segmenter.merge_overlapping_boxes(boxes)
    print(f"[INFO] After merging: {len(boxes)} segments")
    
    # Adjust to expected count if provided
    boxes = segmenter._adjust_to_expected(boxes, expected_chars)
    print(f"[INFO] After adjust-to-expected: {len(boxes)} segments (target={expected_chars})")
    
    # Save results
    saved_paths = segmenter.save_segments(boxes, output_dir, visualize=visualize)
    
    return saved_paths


def batch_segment(input_dir='data/train', output_dir='test_char', method='hybrid'):
    """
    Batch process multiple CAPTCHA images
    """
    input_path = Path(input_dir)
    image_files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
    
    print(f"[INFO] Found {len(image_files)} images to process")
    
    results = []
    for img_path in image_files:
        print(f"\n{'='*60}")
        print(f"Processing: {img_path.name}")
        print('='*60)
        
        try:
            saved = segment_captcha(str(img_path), method=method, 
                                   output_dir=output_dir, visualize=False)
            results.append((img_path.name, len(saved), 'Success'))
        except Exception as e:
            print(f"[ERROR] Failed: {e}")
            results.append((img_path.name, 0, str(e)))
    
    # Summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print('='*60)
    for name, count, status in results:
        print(f"{name}: {count} chars - {status}")


def main():
    """Example usage"""
    # Single image segmentation
    image_path = "1cfu8x-0.png"
    
    print("Testing different segmentation methods:\n")
    
    # Test all methods
    methods = ['hybrid', 'cca', 'projection', 'contours']
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method.upper()}")
        print('='*60)
        try:
            segment_captcha(image_path, method=method, visualize=False)
        except Exception as e:
            print(f"[ERROR] {method} failed: {e}")
    
    # Batch processing (uncomment to use)
    # batch_segment('data/train', 'test_char', method='hybrid')


if __name__ == "__main__":
    main()
