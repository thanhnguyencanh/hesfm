#!/usr/bin/env python3
"""
HESFM Mapping Evaluation & Benchmark Script

Standalone evaluation comparing HESFM against baselines (SEE-CSOM, EvSemMap,
standard BKI). Supports SUN RGB-D, NYUv2, and SceneNN datasets.

Metrics:
  - mIoU: mean Intersection-over-Union (per-class and overall)
  - RMSE: root mean square error for geometric accuracy
  - ECE:  Expected Calibration Error for uncertainty quality
  - MCE:  Maximum Calibration Error
  - Compression ratio: points → primitives
  - Update latency

Usage:
  python3 evaluate_mapping.py --pred <path> --gt <path> [--config config.yaml]
  python3 evaluate_mapping.py --results_dir <dir>   # batch mode

Author: Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
Date: 2026
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ---------------------------------------------------------------------------
# Class name tables
# ---------------------------------------------------------------------------
SUNRGBD_37 = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table",
    "door", "window", "bookshelf", "picture", "counter", "blinds",
    "desk", "shelves", "curtain", "dresser", "pillow", "mirror",
    "floor_mat", "clothes", "ceiling", "books", "fridge",
    "tv", "paper", "towel", "shower_curtain", "box",
    "whiteboard", "person", "night_stand", "toilet", "sink",
    "lamp", "bathtub", "bag",
]

NYUV2_40 = SUNRGBD_37 + ["otherstructure", "otherfurniture", "otherprop"]


# ============================================================================
# Metric helpers
# ============================================================================

def confusion_matrix(pred: np.ndarray, gt: np.ndarray,
                     num_classes: int, ignore: int = 255) -> np.ndarray:
    """Build num_classes x num_classes confusion matrix."""
    mask = gt != ignore
    p, g = pred[mask].astype(int), gt[mask].astype(int)
    valid = (p >= 0) & (p < num_classes) & (g >= 0) & (g < num_classes)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (g[valid], p[valid]), 1)
    return cm


def iou_from_cm(cm: np.ndarray):
    """Per-class IoU from confusion matrix."""
    intersection = np.diag(cm).astype(np.float64)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    iou = np.where(union > 0, intersection / union, np.nan)
    return iou


def miou_from_cm(cm: np.ndarray) -> float:
    return float(np.nanmean(iou_from_cm(cm)))


def pixel_accuracy(cm: np.ndarray) -> float:
    total = cm.sum()
    return float(np.diag(cm).sum() / total) if total > 0 else 0.0


def rmse(pred_pos: np.ndarray, gt_pos: np.ndarray) -> float:
    """RMSE between matched 3-D positions (N x 3)."""
    diff = pred_pos - gt_pos
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                                num_bins: int = 15,
                                ignore: int = 255):
    """ECE and MCE from class-probability array and label array.

    probs:  (N, C) predicted probabilities
    labels: (N,) ground-truth class indices
    """
    mask = labels != ignore
    probs, labels = probs[mask], labels[mask]
    if len(labels) == 0:
        return 0.0, 0.0

    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == labels).astype(float)

    bins = np.linspace(0, 1, num_bins + 1)
    ece, mce = 0.0, 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        n = in_bin.sum()
        if n == 0:
            continue
        acc = correct[in_bin].mean()
        conf = confidence[in_bin].mean()
        gap = abs(acc - conf)
        ece += (n / len(labels)) * gap
        mce = max(mce, gap)

    return float(ece), float(mce)


# ============================================================================
# I/O: load predictions / ground truth
# ============================================================================

def load_label_image(path: str) -> np.ndarray:
    """Load a single-channel label image (png / npy)."""
    if path.endswith(".npy"):
        return np.load(path)
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    return img


def load_prob_npy(path: str) -> np.ndarray:
    """Load (H, W, C) probability map from .npy."""
    return np.load(path)


def load_pointcloud_map(path: str):
    """Load a semantic point cloud map (x y z class confidence).

    Accepted formats: .txt, .csv, .npy
    Returns (positions N×3, classes N, confidences N).
    """
    if path.endswith(".npy"):
        data = np.load(path)
    else:
        data = np.loadtxt(path, delimiter=None)
    positions = data[:, :3]
    classes = data[:, 3].astype(int)
    confidences = data[:, 4] if data.shape[1] > 4 else np.ones(len(data))
    return positions, classes, confidences


# ============================================================================
# Comparison framework
# ============================================================================

class MethodResult:
    """Container for one method's evaluation results."""

    def __init__(self, name: str):
        self.name = name
        self.miou = 0.0
        self.per_class_iou = None
        self.pixel_acc = 0.0
        self.rmse = None
        self.ece = 0.0
        self.mce = 0.0
        self.compression_ratio = None
        self.mean_latency_ms = None

    def to_dict(self) -> dict:
        d = OrderedDict(name=self.name, miou=self.miou,
                        pixel_acc=self.pixel_acc,
                        ece=self.ece, mce=self.mce)
        if self.rmse is not None:
            d["rmse"] = self.rmse
        if self.compression_ratio is not None:
            d["compression_ratio"] = self.compression_ratio
        if self.mean_latency_ms is not None:
            d["mean_latency_ms"] = self.mean_latency_ms
        return d


def evaluate_2d_segmentation(pred_dir: str, gt_dir: str, num_classes: int,
                              method_name: str = "HESFM") -> MethodResult:
    """Evaluate 2-D semantic segmentation predictions against ground truth.

    Expects matching filenames in pred_dir and gt_dir.
    """
    res = MethodResult(method_name)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.png")) +
                        glob.glob(os.path.join(pred_dir, "*.npy")))
    if not pred_files:
        print(f"  [WARN] No prediction files found in {pred_dir}")
        return res

    for pf in pred_files:
        base = Path(pf).stem
        # Try matching gt file
        for ext in (".png", ".npy"):
            gt_path = os.path.join(gt_dir, base + ext)
            if os.path.exists(gt_path):
                break
        else:
            continue

        pred = load_label_image(pf)
        gt = load_label_image(gt_path)
        if pred.shape != gt.shape:
            import cv2
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        cm += confusion_matrix(pred, gt, num_classes)

    res.miou = miou_from_cm(cm)
    res.per_class_iou = iou_from_cm(cm)
    res.pixel_acc = pixel_accuracy(cm)
    return res


def evaluate_3d_map(pred_path: str, gt_path: str, resolution: float,
                    num_classes: int,
                    method_name: str = "HESFM") -> MethodResult:
    """Evaluate a 3-D semantic map against ground truth."""
    res = MethodResult(method_name)

    pred_pos, pred_cls, pred_conf = load_pointcloud_map(pred_path)
    gt_pos, gt_cls, _ = load_pointcloud_map(gt_path)

    # Voxelise both into the same grid
    def voxelise(pos, cls, res):
        keys = np.round(pos / res).astype(int)
        d = {}
        for i in range(len(keys)):
            k = tuple(keys[i])
            d[k] = cls[i]
        return d

    pred_vox = voxelise(pred_pos, pred_cls, resolution)
    gt_vox = voxelise(gt_pos, gt_cls, resolution)

    common = set(pred_vox.keys()) & set(gt_vox.keys())
    if not common:
        print(f"  [WARN] No overlapping voxels between pred and gt")
        return res

    p_arr = np.array([pred_vox[k] for k in common])
    g_arr = np.array([gt_vox[k] for k in common])

    cm = confusion_matrix(p_arr, g_arr, num_classes)
    res.miou = miou_from_cm(cm)
    res.per_class_iou = iou_from_cm(cm)
    res.pixel_acc = pixel_accuracy(cm)

    # Geometric RMSE: for each gt voxel, find nearest pred voxel
    from scipy.spatial import cKDTree
    tree = cKDTree(pred_pos)
    dists, _ = tree.query(gt_pos, k=1)
    res.rmse = float(np.sqrt(np.mean(dists ** 2)))

    return res


# ============================================================================
# Reporting
# ============================================================================

def print_comparison_table(results: list, class_names=None):
    """Pretty-print comparison table to terminal."""
    hdr = f"{'Method':<20} {'mIoU':>8} {'PixAcc':>8} {'ECE':>8} {'MCE':>8}"
    if any(r.rmse is not None for r in results):
        hdr += f" {'RMSE':>8}"
    if any(r.compression_ratio is not None for r in results):
        hdr += f" {'Comp.R':>8}"
    if any(r.mean_latency_ms is not None for r in results):
        hdr += f" {'Lat(ms)':>8}"

    print("\n" + "=" * len(hdr))
    print("HESFM Benchmark Comparison")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        line = f"{r.name:<20} {r.miou:>8.4f} {r.pixel_acc:>8.4f} {r.ece:>8.4f} {r.mce:>8.4f}"
        if any(rr.rmse is not None for rr in results):
            line += f" {r.rmse or 0:>8.4f}"
        if any(rr.compression_ratio is not None for rr in results):
            line += f" {r.compression_ratio or 0:>8.1f}"
        if any(rr.mean_latency_ms is not None for rr in results):
            line += f" {r.mean_latency_ms or 0:>8.1f}"
        print(line)

    print("=" * len(hdr))

    # Per-class IoU table
    if class_names and any(r.per_class_iou is not None for r in results):
        print(f"\n{'Class':<20}", end="")
        for r in results:
            print(f" {r.name:>12}", end="")
        print()
        print("-" * (20 + 13 * len(results)))
        for i, cn in enumerate(class_names):
            print(f"{cn:<20}", end="")
            for r in results:
                if r.per_class_iou is not None and i < len(r.per_class_iou):
                    v = r.per_class_iou[i]
                    print(f" {v:>12.4f}" if not np.isnan(v) else f" {'N/A':>12}", end="")
                else:
                    print(f" {'—':>12}", end="")
            print()


def save_results_json(results: list, output_path: str):
    data = [r.to_dict() for r in results]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=float)
    print(f"Results saved to {output_path}")


def plot_comparison(results: list, output_dir: str, class_names=None):
    """Generate comparison bar charts."""
    if not HAS_MPL:
        print("[WARN] matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)
    names = [r.name for r in results]

    # --- mIoU bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    mious = [r.miou for r in results]
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    bars = ax.bar(names, mious, color=colors)
    ax.set_ylabel("mIoU")
    ax.set_title("Semantic Mapping: mIoU Comparison")
    ax.set_ylim(0, 1)
    for bar, v in zip(bars, mious):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "miou_comparison.png"), dpi=150)
    plt.close(fig)

    # --- ECE bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    eces = [r.ece for r in results]
    bars = ax.bar(names, eces, color=colors)
    ax.set_ylabel("ECE (lower is better)")
    ax.set_title("Uncertainty Calibration: ECE Comparison")
    for bar, v in zip(bars, eces):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ece_comparison.png"), dpi=150)
    plt.close(fig)

    # --- Per-class IoU grouped bar chart ---
    if class_names and all(r.per_class_iou is not None for r in results):
        nc = min(len(class_names), len(results[0].per_class_iou))
        x = np.arange(nc)
        width = 0.8 / len(results)
        fig, ax = plt.subplots(figsize=(max(12, nc * 0.5), 6))
        for idx, r in enumerate(results):
            vals = [r.per_class_iou[i] if not np.isnan(r.per_class_iou[i]) else 0
                    for i in range(nc)]
            ax.bar(x + idx * width, vals, width, label=r.name, color=colors[idx])
        ax.set_xticks(x + width * (len(results) - 1) / 2)
        ax.set_xticklabels(class_names[:nc], rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("IoU")
        ax.set_title("Per-Class IoU Comparison")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "per_class_iou.png"), dpi=150)
        plt.close(fig)

    print(f"Plots saved to {output_dir}/")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="HESFM Mapping Evaluation & Benchmark")
    p.add_argument("--pred", help="Prediction directory or map file")
    p.add_argument("--gt", help="Ground truth directory or map file")
    p.add_argument("--config", help="Benchmark config YAML")
    p.add_argument("--results_dir",
                   help="Directory with per-method subdirs (batch mode)")
    p.add_argument("--num_classes", type=int, default=37,
                   help="Number of semantic classes (37=SUNRGBD, 40=NYUv2)")
    p.add_argument("--resolution", type=float, default=0.05,
                   help="Map voxel resolution (metres)")
    p.add_argument("--mode", choices=["2d", "3d"], default="2d",
                   help="Evaluation mode: 2d segmentation or 3d map")
    p.add_argument("--output", default="eval_results",
                   help="Output directory for results and plots")
    p.add_argument("--method_name", default="HESFM",
                   help="Name of the method being evaluated")
    p.add_argument("--no_plot", action="store_true",
                   help="Skip plot generation")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    class_names = SUNRGBD_37 if args.num_classes == 37 else NYUV2_40
    results = []

    # --- Batch mode: each subfolder is a method ---
    if args.results_dir:
        gt_dir = args.gt
        for method_dir in sorted(Path(args.results_dir).iterdir()):
            if not method_dir.is_dir():
                continue
            name = method_dir.name
            print(f"Evaluating {name} ...")
            if args.mode == "2d":
                r = evaluate_2d_segmentation(str(method_dir), gt_dir,
                                              args.num_classes, name)
            else:
                pred_file = next(method_dir.glob("*.txt"), None) or \
                            next(method_dir.glob("*.npy"), None)
                if pred_file is None:
                    print(f"  [SKIP] No map file in {method_dir}")
                    continue
                r = evaluate_3d_map(str(pred_file), gt_dir,
                                     args.resolution, args.num_classes, name)
            results.append(r)

    # --- Single method mode ---
    elif args.pred and args.gt:
        print(f"Evaluating {args.method_name} ...")
        if args.mode == "2d":
            r = evaluate_2d_segmentation(args.pred, args.gt,
                                          args.num_classes, args.method_name)
        else:
            r = evaluate_3d_map(args.pred, args.gt, args.resolution,
                                 args.num_classes, args.method_name)
        results.append(r)

    # --- Config file mode ---
    elif args.config and HAS_YAML:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        gt_dir = cfg.get("ground_truth_dir", args.gt)
        nc = cfg.get("num_classes", args.num_classes)
        res_val = cfg.get("resolution", args.resolution)
        for method in cfg.get("methods", []):
            name = method["name"]
            pred = method["predictions"]
            mode = method.get("mode", args.mode)
            print(f"Evaluating {name} ...")
            if mode == "2d":
                r = evaluate_2d_segmentation(pred, gt_dir, nc, name)
            else:
                r = evaluate_3d_map(pred, gt_dir, res_val, nc, name)
            # Optionally attach extra metadata
            if "compression_ratio" in method:
                r.compression_ratio = method["compression_ratio"]
            if "mean_latency_ms" in method:
                r.mean_latency_ms = method["mean_latency_ms"]
            results.append(r)
    else:
        print("Usage: provide --pred/--gt, --results_dir, or --config")
        sys.exit(1)

    if not results:
        print("No results to report.")
        sys.exit(0)

    # Report
    print_comparison_table(results, class_names)
    save_results_json(results, os.path.join(args.output, "benchmark.json"))
    if not args.no_plot:
        plot_comparison(results, args.output, class_names)


if __name__ == "__main__":
    main()
