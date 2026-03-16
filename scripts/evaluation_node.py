#!/usr/bin/env python3
"""
HESFM Evaluation Node

Computes evaluation metrics for semantic mapping:
- mIoU (mean Intersection over Union)
- ECE (Expected Calibration Error)
- AUPRC (Area Under Precision-Recall Curve)
- Timing statistics

Author: Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
Date: 2026
"""

import rospy
import numpy as np
import cv2
import os
import json
import time
from collections import defaultdict
from threading import Lock

from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError

try:
    from sklearn.metrics import precision_recall_curve, auc
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class ConfusionMatrix:
    """Confusion matrix for semantic segmentation evaluation."""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        
    def update(self, pred, gt, ignore_index=255):
        """Update confusion matrix with predictions and ground truth."""
        mask = gt != ignore_index
        pred_valid = pred[mask]
        gt_valid = gt[mask]
        
        for p, g in zip(pred_valid.flatten(), gt_valid.flatten()):
            if 0 <= p < self.num_classes and 0 <= g < self.num_classes:
                self.matrix[g, p] += 1
                
    def get_iou(self):
        """Compute IoU per class."""
        intersection = np.diag(self.matrix)
        union = self.matrix.sum(axis=1) + self.matrix.sum(axis=0) - intersection
        iou = np.where(union > 0, intersection / union, 0)
        return iou
        
    def get_miou(self):
        """Compute mean IoU."""
        iou = self.get_iou()
        valid = self.matrix.sum(axis=1) > 0
        return np.mean(iou[valid])
        
    def get_accuracy(self):
        """Compute pixel accuracy."""
        correct = np.diag(self.matrix).sum()
        total = self.matrix.sum()
        return correct / total if total > 0 else 0
        
    def get_class_accuracy(self):
        """Compute class-wise accuracy."""
        class_total = self.matrix.sum(axis=1)
        class_correct = np.diag(self.matrix)
        return np.where(class_total > 0, class_correct / class_total, 0)
        
    def reset(self):
        """Reset confusion matrix."""
        self.matrix.fill(0)


class CalibrationEvaluator:
    """Expected Calibration Error computation."""
    
    def __init__(self, num_bins=10):
        self.num_bins = num_bins
        self.bin_boundaries = np.linspace(0, 1, num_bins + 1)
        self.reset()
        
    def update(self, probs, labels, ignore_index=255):
        """Update with probability predictions and labels."""
        mask = labels != ignore_index
        
        if not np.any(mask):
            return
            
        probs_valid = probs[mask]
        labels_valid = labels[mask]
        
        # Get predicted class and confidence
        pred_class = np.argmax(probs_valid, axis=-1)
        confidence = np.max(probs_valid, axis=-1)
        correct = (pred_class == labels_valid).astype(np.float32)
        
        # Bin by confidence
        for i in range(self.num_bins):
            low, high = self.bin_boundaries[i], self.bin_boundaries[i + 1]
            in_bin = (confidence > low) & (confidence <= high)
            
            if np.any(in_bin):
                self.bin_counts[i] += np.sum(in_bin)
                self.bin_correct[i] += np.sum(correct[in_bin])
                self.bin_confidence[i] += np.sum(confidence[in_bin])
                
    def get_ece(self):
        """Compute Expected Calibration Error."""
        total = np.sum(self.bin_counts)
        if total == 0:
            return 0.0
            
        ece = 0.0
        for i in range(self.num_bins):
            if self.bin_counts[i] > 0:
                accuracy = self.bin_correct[i] / self.bin_counts[i]
                avg_confidence = self.bin_confidence[i] / self.bin_counts[i]
                ece += (self.bin_counts[i] / total) * abs(accuracy - avg_confidence)
                
        return ece
        
    def get_mce(self):
        """Compute Maximum Calibration Error."""
        mce = 0.0
        for i in range(self.num_bins):
            if self.bin_counts[i] > 0:
                accuracy = self.bin_correct[i] / self.bin_counts[i]
                avg_confidence = self.bin_confidence[i] / self.bin_counts[i]
                mce = max(mce, abs(accuracy - avg_confidence))
        return mce
        
    def reset(self):
        """Reset calibration statistics."""
        self.bin_counts = np.zeros(self.num_bins)
        self.bin_correct = np.zeros(self.num_bins)
        self.bin_confidence = np.zeros(self.num_bins)


class PrecisionRecallEvaluator:
    """AUPRC computation per class."""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
        
    def update(self, probs, labels, ignore_index=255):
        """Update with probability predictions and labels."""
        if not HAS_SKLEARN:
            return
            
        mask = labels != ignore_index
        if not np.any(mask):
            return
            
        probs_valid = probs[mask]
        labels_valid = labels[mask]
        
        for c in range(self.num_classes):
            binary_labels = (labels_valid == c).astype(np.int32)
            if np.sum(binary_labels) > 0:
                self.all_probs[c].extend(probs_valid[:, c].tolist())
                self.all_labels[c].extend(binary_labels.tolist())
                
    def get_auprc(self):
        """Compute AUPRC per class."""
        if not HAS_SKLEARN:
            return np.zeros(self.num_classes)
            
        auprc = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            if len(self.all_labels[c]) > 0 and np.sum(self.all_labels[c]) > 0:
                precision, recall, _ = precision_recall_curve(
                    self.all_labels[c], self.all_probs[c])
                auprc[c] = auc(recall, precision)
        return auprc
        
    def get_mean_auprc(self):
        """Compute mean AUPRC."""
        auprc = self.get_auprc()
        valid = []
        for c in range(self.num_classes):
            if len(self.all_labels[c]) > 0 and np.sum(self.all_labels[c]) > 0:
                valid.append(auprc[c])
        return np.mean(valid) if valid else 0.0
        
    def reset(self):
        """Reset PR statistics."""
        self.all_probs = [[] for _ in range(self.num_classes)]
        self.all_labels = [[] for _ in range(self.num_classes)]


class TimingStatistics:
    """Timing statistics tracker."""
    
    def __init__(self):
        self.reset()
        
    def update(self, name, elapsed_time):
        """Update timing for a component."""
        self.times[name].append(elapsed_time)
        
    def get_stats(self, name):
        """Get statistics for a component."""
        if name not in self.times or len(self.times[name]) == 0:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
            
        times = np.array(self.times[name])
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'count': len(times),
            'fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0
        }
        
    def get_all_stats(self):
        """Get statistics for all components."""
        return {name: self.get_stats(name) for name in self.times}
        
    def reset(self):
        """Reset timing statistics."""
        self.times = defaultdict(list)


class EvaluationNode:
    """ROS node for HESFM evaluation."""
    
    NYU40_CLASSES = [
        "wall", "floor", "cabinet", "bed", "chair", "sofa", "table",
        "door", "window", "bookshelf", "picture", "counter", "blinds",
        "desk", "shelves", "curtain", "dresser", "pillow", "mirror",
        "floor_mat", "clothes", "ceiling", "books", "fridge",
        "television", "paper", "towel", "shower_curtain", "box",
        "whiteboard", "person", "night_stand", "toilet", "sink",
        "lamp", "bathtub", "bag", "otherstructure", "otherfurniture",
        "otherprop"
    ]
    
    def __init__(self):
        rospy.init_node('evaluation_node', anonymous=False)
        
        self.bridge = CvBridge()
        self.lock = Lock()
        
        # Parameters
        self.num_classes = rospy.get_param('~num_classes', 40)
        self.output_dir = rospy.get_param('~output_dir', '/tmp/hesfm_eval')
        self.experiment_name = rospy.get_param('~experiment_name', 'hesfm_eval')
        self.ece_num_bins = rospy.get_param('~ece_num_bins', 10)
        
        self.evaluate_segmentation = rospy.get_param('~evaluate_segmentation', True)
        self.evaluate_mapping = rospy.get_param('~evaluate_mapping', True)
        self.evaluate_uncertainty = rospy.get_param('~evaluate_uncertainty', True)
        self.evaluate_timing = rospy.get_param('~evaluate_timing', True)
        self.save_visualizations = rospy.get_param('~save_visualizations', False)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize evaluators
        self.confusion_matrix = ConfusionMatrix(self.num_classes)
        self.calibration = CalibrationEvaluator(self.ece_num_bins)
        self.pr_evaluator = PrecisionRecallEvaluator(self.num_classes)
        self.timing = TimingStatistics()
        
        # Cached data
        self.pred_semantic = None
        self.gt_semantic = None
        self.pred_probs = None
        self.frame_count = 0
        
        # Subscribers
        self.pred_sub = rospy.Subscriber('predicted_semantic', Image,
                                          self.pred_callback, queue_size=1)
        self.gt_sub = rospy.Subscriber('gt_semantic', Image,
                                        self.gt_callback, queue_size=1)
        
        # Publishers
        self.vis_pub = rospy.Publisher('~visualization', Image, queue_size=1)
        
        # Timer for periodic reporting
        self.report_timer = rospy.Timer(rospy.Duration(10.0), self.report_callback)
        
        rospy.loginfo(f"Evaluation node initialized. Output: {self.output_dir}")
        
    def pred_callback(self, msg):
        """Predicted semantic image callback."""
        try:
            start_time = time.time()
            self.pred_semantic = self.bridge.imgmsg_to_cv2(msg, 'mono8')
            self.timing.update('prediction', time.time() - start_time)
            self.evaluate()
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
            
    def gt_callback(self, msg):
        """Ground truth semantic image callback."""
        try:
            self.gt_semantic = self.bridge.imgmsg_to_cv2(msg, 'mono8')
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
            
    def evaluate(self):
        """Evaluate current frame."""
        if self.pred_semantic is None or self.gt_semantic is None:
            return
            
        with self.lock:
            start_time = time.time()
            
            # Ensure same size
            if self.pred_semantic.shape != self.gt_semantic.shape:
                self.pred_semantic = cv2.resize(
                    self.pred_semantic, 
                    (self.gt_semantic.shape[1], self.gt_semantic.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
                    
            # Update confusion matrix
            if self.evaluate_segmentation:
                self.confusion_matrix.update(self.pred_semantic, self.gt_semantic)
                
            self.frame_count += 1
            self.timing.update('evaluation', time.time() - start_time)
            
    def report_callback(self, event):
        """Periodic reporting callback."""
        if self.frame_count == 0:
            return
            
        with self.lock:
            rospy.loginfo("=" * 50)
            rospy.loginfo(f"Evaluation Report (Frame {self.frame_count})")
            rospy.loginfo("=" * 50)
            
            if self.evaluate_segmentation:
                miou = self.confusion_matrix.get_miou()
                acc = self.confusion_matrix.get_accuracy()
                rospy.loginfo(f"mIoU: {miou:.4f}")
                rospy.loginfo(f"Pixel Accuracy: {acc:.4f}")
                
                # Per-class IoU
                iou = self.confusion_matrix.get_iou()
                rospy.loginfo("Per-class IoU:")
                for c in range(min(self.num_classes, len(self.NYU40_CLASSES))):
                    if iou[c] > 0:
                        rospy.loginfo(f"  {self.NYU40_CLASSES[c]}: {iou[c]:.4f}")
                        
            if self.evaluate_uncertainty:
                ece = self.calibration.get_ece()
                mce = self.calibration.get_mce()
                rospy.loginfo(f"ECE: {ece:.4f}")
                rospy.loginfo(f"MCE: {mce:.4f}")
                
                if HAS_SKLEARN:
                    mean_auprc = self.pr_evaluator.get_mean_auprc()
                    rospy.loginfo(f"Mean AUPRC: {mean_auprc:.4f}")
                    
            if self.evaluate_timing:
                timing_stats = self.timing.get_all_stats()
                rospy.loginfo("Timing Statistics:")
                for name, stats in timing_stats.items():
                    rospy.loginfo(f"  {name}: {stats['mean']*1000:.2f}ms "
                                  f"(FPS: {stats['fps']:.1f})")
                                  
    def save_results(self):
        """Save evaluation results to file."""
        results = {
            'experiment': self.experiment_name,
            'num_frames': self.frame_count,
            'timestamp': rospy.Time.now().to_sec(),
        }
        
        if self.evaluate_segmentation:
            results['miou'] = float(self.confusion_matrix.get_miou())
            results['pixel_accuracy'] = float(self.confusion_matrix.get_accuracy())
            results['per_class_iou'] = {
                self.NYU40_CLASSES[c]: float(iou)
                for c, iou in enumerate(self.confusion_matrix.get_iou())
                if c < len(self.NYU40_CLASSES)
            }
            
        if self.evaluate_uncertainty:
            results['ece'] = float(self.calibration.get_ece())
            results['mce'] = float(self.calibration.get_mce())
            if HAS_SKLEARN:
                results['mean_auprc'] = float(self.pr_evaluator.get_mean_auprc())
                
        if self.evaluate_timing:
            results['timing'] = self.timing.get_all_stats()
            
        # Save JSON
        output_path = os.path.join(self.output_dir, f'{self.experiment_name}_results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x))
            
        rospy.loginfo(f"Results saved to {output_path}")
        
        # Save confusion matrix as CSV
        cm_path = os.path.join(self.output_dir, f'{self.experiment_name}_confusion_matrix.csv')
        np.savetxt(cm_path, self.confusion_matrix.matrix, delimiter=',', fmt='%d')
        
    def run(self):
        """Run the node."""
        rospy.on_shutdown(self.save_results)
        rospy.spin()


def main():
    try:
        node = EvaluationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()