import os
import argparse

def parse_annotations(file_path, class_map=None):
    """
    Parse a YOLO-style annotation file.
    Each line should contain: <class> <x_center> <y_center> <width> <height>
    Returns a list of tuples: (class, x_center, y_center, width, height)
    """
    annotations = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            if class_map is not None:
                cls = class_map.get(cls, cls)
            try:
                x_center, y_center, width, height = map(float, parts[1:5])
                annotations.append((cls, x_center, y_center, width, height))
            except ValueError:
                print(f"Error parsing line in {file_path}: {line}")
    return annotations

def compute_iou(box1, box2):
    """
    Compute the Intersection-over-Union (IoU) of two bounding boxes.
    Each box is expected to be a tuple: (class, x_center, y_center, width, height).
    Convert center coordinates to corner coordinates before computing IoU.
    """
    # Convert box1 from center to (xmin, ymin, xmax, ymax)
    x1_min = box1[1] - box1[3] / 2
    y1_min = box1[2] - box1[4] / 2
    x1_max = box1[1] + box1[3] / 2
    y1_max = box1[2] + box1[4] / 2

    # Convert box2 from center to (xmin, ymin, xmax, ymax)
    x2_min = box2[1] - box2[3] / 2
    y2_min = box2[2] - box2[4] / 2
    x2_max = box2[1] + box2[3] / 2
    y2_max = box2[2] + box2[4] / 2

    # Calculate intersection coordinates
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def main():
    parser = argparse.ArgumentParser(
        description="Compute F1, precision, and recall for YOLO annotations"
    )
    parser.add_argument("--gt-dir", type=str, required=True,
                        help="Directory with ground truth annotation files")
    parser.add_argument("--pred-dir", type=str, required=True,
                        help="Directory with predicted annotation files")
    parser.add_argument("--iou-threshold", type=float, default=0.6,
                        help="IoU threshold to consider a detection as a match")
    args = parser.parse_args()

    class_mapping = {
        0: 0,
        20: 1,
        1: 2,
        2: 3,
        3: 4,
        4: 5,
        5: 6,
        6: 7,
        7: 8,
        8: 9,
        9: 10,
        10: 11,
        11: 12,
        12: 13,
        13: 14,
        14: 15,
        24: 16,
        15: 17,
        16: 18,
        17: 19,
        18: 20,
        19: 21,
        21: 22,
        22: 23,
        23: 24
    }


    # Dictionary to accumulate metrics for each class
    # metrics[class] = {"tp": int, "fp": int, "fn": int}
    metrics = {}

    # Process each file in the ground truth directory
    for filename in os.listdir(args.gt_dir):
        gt_file = os.path.join(args.gt_dir, filename)
        pred_file = os.path.join(args.pred_dir, filename)
        if not os.path.exists(pred_file):
            print(f"Warning: Prediction file {pred_file} not found. Skipping file {filename}.")
            continue

        gt_annotations = parse_annotations(gt_file)
        pred_annotations = parse_annotations(pred_file, class_mapping)

        # Group annotations by class for ground truth and predictions
        gt_by_class = {}
        for ann in gt_annotations:
            cls = ann[0]
            gt_by_class.setdefault(cls, []).append(ann)

        pred_by_class = {}
        for ann in pred_annotations:
            cls = ann[0]
            pred_by_class.setdefault(cls, []).append(ann)

        # Process each class that appears in either ground truth or predictions
        classes = set(list(gt_by_class.keys()) + list(pred_by_class.keys()))
        for cls in classes:
            # Initialize metrics for the class if not already done
            if cls not in metrics:
                metrics[cls] = {"tp": 0, "fp": 0, "fn": 0}

            gt_boxes = gt_by_class.get(cls, [])
            pred_boxes = pred_by_class.get(cls, [])

            # Keep track of ground truth boxes that have been matched
            matched_gt = [False] * len(gt_boxes)

            # For each predicted box, try to find a matching ground truth box (greedy matching)
            for pred_box in pred_boxes:
                best_iou = 0
                best_match_idx = -1
                for idx, gt_box in enumerate(gt_boxes):
                    if not matched_gt[idx]:
                        iou = compute_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_match_idx = idx
                if best_iou >= args.iou_threshold:
                    metrics[cls]["tp"] += 1
                    matched_gt[best_match_idx] = True
                else:
                    metrics[cls]["fp"] += 1

            # The remaining unmatched ground truth boxes are false negatives
            metrics[cls]["fn"] += matched_gt.count(False)

        # Prepare CSV rows and compute per-class metrics
    csv_rows = []
    header = ["Class", "P", "R", "F1"]
    csv_rows.append(header)

    # To compute average values
    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0
    count = 0

    # sort metrics by class
    metrics = dict(sorted(metrics.items(), key=lambda x: x[0]))

    for cls, counts in metrics.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        csv_rows.append([cls, f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])

        sum_precision += precision
        sum_recall += recall
        sum_f1 += f1
        count += 1

    # Compute averages if any classes were processed
    if count > 0:
        avg_precision = sum_precision / count
        avg_recall = sum_recall / count
        avg_f1 = sum_f1 / count
    else:
        avg_precision = avg_recall = avg_f1 = 0

    csv_rows.append(["Average", f"{avg_precision:.4f}", f"{avg_recall:.4f}", f"{avg_f1:.4f}"])

    # Write the results to a standard output and a CSV
    import csv

    writer = csv.writer(os.sys.stdout)
    writer.writerows(csv_rows)


if __name__ == "__main__":
    main()