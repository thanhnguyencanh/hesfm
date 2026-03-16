#!/usr/bin/env python3
"""
Uncertainty Visualizer Node for HESFM

Provides real-time visualization of uncertainty decomposition and
uncertainty-weighted confidence maps.

Subscriptions:
    - semantic/image (sensor_msgs/Image): Semantic labels
    - semantic/uncertainty (sensor_msgs/Image): Per-pixel uncertainty
    - color/image_raw (sensor_msgs/Image): Original RGB image

Publications:
    - visualization/uncertainty_overlay (sensor_msgs/Image): Overlay visualization
    - visualization/uncertainty_heatmap (sensor_msgs/Image): Uncertainty heatmap
    - visualization/confidence_map (sensor_msgs/Image): Confidence visualization

Author: Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
Date: 2026
"""

import rospy
import numpy as np
import cv2
from typing import Optional, Tuple
import threading

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber


# Colormaps
def create_uncertainty_colormap() -> np.ndarray:
    """Create colormap: blue (low) -> yellow -> red (high)"""
    colormap = np.zeros((256, 3), dtype=np.uint8)
    
    for i in range(256):
        t = i / 255.0
        
        if t < 0.5:
            # Blue to yellow
            r = int(255 * (2 * t))
            g = int(255 * (2 * t))
            b = int(255 * (1 - 2 * t))
        else:
            # Yellow to red
            r = 255
            g = int(255 * (2 - 2 * t))
            b = 0
        
        colormap[i] = [b, g, r]  # BGR
    
    return colormap


def create_confidence_colormap() -> np.ndarray:
    """Create colormap: red (low) -> green (high)"""
    colormap = np.zeros((256, 3), dtype=np.uint8)
    
    for i in range(256):
        t = i / 255.0
        
        r = int(255 * (1 - t))
        g = int(255 * t)
        b = 0
        
        colormap[i] = [b, g, r]  # BGR
    
    return colormap


class UncertaintyVisualizer:
    """Real-time uncertainty visualization"""
    
    def __init__(self):
        rospy.init_node('uncertainty_visualizer', anonymous=False)
        
        self.bridge = CvBridge()
        
        # Parameters
        self.overlay_alpha = rospy.get_param('~overlay_alpha', 0.5)
        self.threshold_low = rospy.get_param('~threshold_low', 0.3)
        self.threshold_high = rospy.get_param('~threshold_high', 0.7)
        
        # Colormaps
        self.uncertainty_cmap = create_uncertainty_colormap()
        self.confidence_cmap = create_confidence_colormap()
        
        # State
        self.latest_rgb = None
        self.latest_labels = None
        self.lock = threading.Lock()
        
        # Setup ROS
        self._setup_subscribers()
        self._setup_publishers()
        
        rospy.loginfo("Uncertainty Visualizer initialized")
    
    def _setup_subscribers(self):
        """Setup synchronized subscribers"""
        self.rgb_sub = Subscriber('color/image_raw', Image)
        self.uncertainty_sub = Subscriber('semantic/uncertainty', Image)
        self.labels_sub = Subscriber('semantic/image', Image)
        
        # Sync RGB and uncertainty
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.uncertainty_sub, self.labels_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self._sync_callback)
    
    def _setup_publishers(self):
        """Setup publishers"""
        self.overlay_pub = rospy.Publisher(
            'visualization/uncertainty_overlay', Image, queue_size=1)
        self.heatmap_pub = rospy.Publisher(
            'visualization/uncertainty_heatmap', Image, queue_size=1)
        self.confidence_pub = rospy.Publisher(
            'visualization/confidence_map', Image, queue_size=1)
        self.decomposition_pub = rospy.Publisher(
            'visualization/uncertainty_decomposition', Image, queue_size=1)
    
    def _sync_callback(self, rgb_msg: Image, uncertainty_msg: Image, labels_msg: Image):
        """Process synchronized images"""
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            uncertainty = self.bridge.imgmsg_to_cv2(uncertainty_msg, '32FC1')
            labels = self.bridge.imgmsg_to_cv2(labels_msg, 'mono8')
            
            with self.lock:
                self.latest_rgb = rgb
                self.latest_labels = labels
            
            # Generate visualizations
            self._publish_overlay(rgb_msg.header, rgb, uncertainty)
            self._publish_heatmap(rgb_msg.header, uncertainty)
            self._publish_confidence(rgb_msg.header, uncertainty)
            
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
    
    def _publish_overlay(self, header, rgb: np.ndarray, uncertainty: np.ndarray):
        """Publish uncertainty overlay on RGB"""
        if self.overlay_pub.get_num_connections() == 0:
            return
        
        # Resize uncertainty to match RGB if needed
        if uncertainty.shape[:2] != rgb.shape[:2]:
            uncertainty = cv2.resize(uncertainty, (rgb.shape[1], rgb.shape[0]))
        
        # Convert uncertainty to colormap
        uncertainty_uint8 = (np.clip(uncertainty, 0, 1) * 255).astype(np.uint8)
        uncertainty_color = self.uncertainty_cmap[uncertainty_uint8]
        
        # Blend with RGB
        overlay = cv2.addWeighted(
            rgb, 1 - self.overlay_alpha,
            uncertainty_color, self.overlay_alpha,
            0
        )
        
        # Add legend
        overlay = self._add_legend(overlay, "Uncertainty")
        
        # Publish
        try:
            msg = self.bridge.cv2_to_imgmsg(overlay, 'bgr8')
            msg.header = header
            self.overlay_pub.publish(msg)
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
    
    def _publish_heatmap(self, header, uncertainty: np.ndarray):
        """Publish standalone uncertainty heatmap"""
        if self.heatmap_pub.get_num_connections() == 0:
            return
        
        # Convert to colormap
        uncertainty_uint8 = (np.clip(uncertainty, 0, 1) * 255).astype(np.uint8)
        heatmap = self.uncertainty_cmap[uncertainty_uint8]
        
        # Add contours at thresholds
        mask_low = (uncertainty > self.threshold_low).astype(np.uint8) * 255
        mask_high = (uncertainty > self.threshold_high).astype(np.uint8) * 255
        
        contours_low, _ = cv2.findContours(mask_low, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_high, _ = cv2.findContours(mask_high, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.drawContours(heatmap, contours_low, -1, (255, 255, 0), 1)  # Cyan
        cv2.drawContours(heatmap, contours_high, -1, (0, 0, 255), 2)   # Red
        
        # Add statistics
        mean_unc = np.mean(uncertainty)
        max_unc = np.max(uncertainty)
        high_ratio = np.mean(uncertainty > self.threshold_high) * 100
        
        cv2.putText(heatmap, f"Mean: {mean_unc:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(heatmap, f"Max: {max_unc:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(heatmap, f"High: {high_ratio:.1f}%", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        try:
            msg = self.bridge.cv2_to_imgmsg(heatmap, 'bgr8')
            msg.header = header
            self.heatmap_pub.publish(msg)
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
    
    def _publish_confidence(self, header, uncertainty: np.ndarray):
        """Publish confidence map (1 - uncertainty)"""
        if self.confidence_pub.get_num_connections() == 0:
            return
        
        confidence = 1.0 - uncertainty
        confidence_uint8 = (np.clip(confidence, 0, 1) * 255).astype(np.uint8)
        confidence_color = self.confidence_cmap[confidence_uint8]
        
        # Add legend
        confidence_color = self._add_legend(confidence_color, "Confidence")
        
        try:
            msg = self.bridge.cv2_to_imgmsg(confidence_color, 'bgr8')
            msg.header = header
            self.confidence_pub.publish(msg)
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
    
    def _add_legend(self, img: np.ndarray, title: str) -> np.ndarray:
        """Add colorbar legend to image"""
        h, w = img.shape[:2]
        legend_width = 30
        legend_margin = 10
        
        # Create legend bar
        legend = np.zeros((h - 2*legend_margin, legend_width, 3), dtype=np.uint8)
        
        if title == "Uncertainty":
            cmap = self.uncertainty_cmap
        else:
            cmap = self.confidence_cmap
        
        for i in range(legend.shape[0]):
            val = int(255 * (1 - i / legend.shape[0]))
            legend[i, :] = cmap[val]
        
        # Add border
        cv2.rectangle(legend, (0, 0), (legend_width-1, legend.shape[0]-1), (255, 255, 255), 1)
        
        # Place legend
        result = img.copy()
        x_offset = w - legend_width - legend_margin
        result[legend_margin:legend_margin+legend.shape[0], x_offset:x_offset+legend_width] = legend
        
        # Add labels
        cv2.putText(result, "1.0", (x_offset - 30, legend_margin + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(result, "0.0", (x_offset - 30, h - legend_margin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(result, title, (x_offset - 60, legend_margin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result
    
    def run(self):
        """Main loop"""
        rospy.spin()


def main():
    try:
        node = UncertaintyVisualizer()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
