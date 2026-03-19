#!/usr/bin/env python3
"""
Segmentation Visualizer Node

Provides side-by-side visualization of RGB, depth, and semantic segmentation.
Useful for debugging and demonstration purposes.

Author: Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
Date: 2026
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from threading import Lock
import message_filters

# NYUv2 40-class color palette (RGB)
NYUV2_COLORS = np.array([
    [128, 128, 128],  # 0  wall
    [139, 119, 101],  # 1  floor
    [244, 164, 96],   # 2  cabinet
    [255, 182, 193],  # 3  bed
    [255, 215, 0],    # 4  chair
    [220, 20, 60],    # 5  sofa
    [255, 140, 0],    # 6  table
    [139, 69, 19],    # 7  door
    [135, 206, 235],  # 8  window
    [160, 82, 45],    # 9  bookshelf
    [255, 105, 180],  # 10 picture
    [0, 128, 128],    # 11 counter
    [210, 180, 140],  # 12 blinds
    [70, 130, 180],   # 13 desk
    [188, 143, 143],  # 14 shelves
    [147, 112, 219],  # 15 curtain
    [222, 184, 135],  # 16 dresser
    [255, 228, 225],  # 17 pillow
    [192, 192, 192],  # 18 mirror
    [139, 119, 101],  # 19 floor_mat
    [128, 0, 128],    # 20 clothes
    [245, 245, 245],  # 21 ceiling
    [139, 90, 43],    # 22 books
    [173, 216, 230],  # 23 fridge
    [0, 0, 139],      # 24 television
    [255, 255, 224],  # 25 paper
    [240, 255, 255],  # 26 towel
    [176, 224, 230],  # 27 shower_curtain
    [210, 105, 30],   # 28 box
    [255, 255, 255],  # 29 whiteboard
    [255, 0, 0],      # 30 person
    [85, 107, 47],    # 31 night_stand
    [255, 255, 240],  # 32 toilet
    [176, 196, 222],  # 33 sink
    [255, 250, 205],  # 34 lamp
    [224, 255, 255],  # 35 bathtub
    [75, 0, 130],     # 36 bag
    [169, 169, 169],  # 37 otherstructure
    [105, 105, 105],  # 38 otherfurniture
    [128, 128, 0],    # 39 otherprop
], dtype=np.uint8)

NYUV2_CLASSES = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
    'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves',
    'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat', 'clothes', 'ceiling',
    'books', 'fridge', 'television', 'paper', 'towel', 'shower_curtain', 'box',
    'whiteboard', 'person', 'night_stand', 'toilet', 'sink', 'lamp', 'bathtub',
    'bag', 'otherstructure', 'otherfurniture', 'otherprop'
]


class SegmentationVisualizer:
    """Visualize RGB, depth, and semantic segmentation side-by-side."""
    
    def __init__(self):
        rospy.init_node('segmentation_visualizer', anonymous=False)
        
        self.bridge = CvBridge()
        self.lock = Lock()
        
        # Parameters
        self.layout = rospy.get_param('~layout', 'horizontal')  # horizontal, vertical, grid
        self.show_legend = rospy.get_param('~show_legend', True)
        self.show_labels = rospy.get_param('~show_labels', True)
        self.overlay_alpha = rospy.get_param('~overlay_alpha', 0.5)
        self.output_width = rospy.get_param('~output_width', 1920)
        self.window_name = rospy.get_param('~window_name', 'Segmentation Visualizer')
        self.save_frames = rospy.get_param('~save_frames', False)
        self.save_path = rospy.get_param('~save_path', '/tmp/segmentation_frames')
        
        # Cached images
        self.rgb_image = None
        self.depth_image = None
        self.semantic_image = None
        self.semantic_color = None
        self.uncertainty_image = None
        
        # Frame counter
        self.frame_count = 0
        
        # Publishers
        self.viz_pub = rospy.Publisher('~visualization', Image, queue_size=1)
        self.overlay_pub = rospy.Publisher('~overlay', Image, queue_size=1)
        
        # Use message_filters for synchronized subscription
        self.use_sync = rospy.get_param('~use_sync', False)
        
        if self.use_sync:
            # Synchronized subscribers
            rgb_sub = message_filters.Subscriber('color', Image)
            semantic_sub = message_filters.Subscriber('semantic', Image)
            
            # Approximate time synchronizer
            self.sync = message_filters.ApproximateTimeSynchronizer(
                [rgb_sub, semantic_sub], queue_size=10, slop=0.1
            )
            self.sync.registerCallback(self.sync_callback)
        else:
            # Independent subscribers
            self.rgb_sub = rospy.Subscriber('color', Image, self.rgb_callback, queue_size=1)
            self.depth_sub = rospy.Subscriber('depth', Image, self.depth_callback, queue_size=1)
            self.semantic_sub = rospy.Subscriber('semantic', Image, self.semantic_callback, queue_size=1)
            self.uncertainty_sub = rospy.Subscriber('uncertainty', Image, self.uncertainty_callback, queue_size=1)
        
        # Timer for visualization update
        self.viz_timer = rospy.Timer(rospy.Duration(0.033), self.update_visualization)  # 30 Hz
        
        # Create save directory if needed
        if self.save_frames:
            import os
            os.makedirs(self.save_path, exist_ok=True)
        
        rospy.loginfo(f"Segmentation Visualizer initialized (layout: {self.layout})")
    
    def rgb_callback(self, msg):
        """RGB image callback."""
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error (RGB): {e}")
    
    def depth_callback(self, msg):
        """Depth image callback."""
        try:
            if msg.encoding == '16UC1':
                depth = self.bridge.imgmsg_to_cv2(msg, '16UC1').astype(np.float32) * 0.001
            elif msg.encoding == '32FC1':
                depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            else:
                depth = self.bridge.imgmsg_to_cv2(msg)
            
            # Normalize for visualization
            depth_valid = depth[depth > 0]
            if len(depth_valid) > 0:
                min_d, max_d = np.percentile(depth_valid, [5, 95])
                depth_norm = np.clip((depth - min_d) / (max_d - min_d + 1e-6), 0, 1)
                self.depth_image = (plt_colormap(depth_norm) * 255).astype(np.uint8)
            else:
                self.depth_image = np.zeros((*depth.shape, 3), dtype=np.uint8)
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error (depth): {e}")
    
    def semantic_callback(self, msg):
        """Semantic segmentation callback."""
        try:
            if msg.encoding == 'mono8' or msg.encoding == '8UC1':
                labels = self.bridge.imgmsg_to_cv2(msg, 'mono8')
                self.semantic_image = labels
                # Colorize
                self.semantic_color = colorize_labels(labels)
            elif msg.encoding == 'bgr8' or msg.encoding == 'rgb8':
                # Already colorized
                self.semantic_color = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                self.semantic_image = None
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error (semantic): {e}")
    
    def uncertainty_callback(self, msg):
        """Uncertainty image callback."""
        try:
            if msg.encoding == '32FC1':
                uncertainty = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            else:
                uncertainty = self.bridge.imgmsg_to_cv2(msg).astype(np.float32) / 255.0
            
            # Colorize uncertainty (red = high uncertainty)
            uncertainty_norm = np.clip(uncertainty, 0, 1)
            self.uncertainty_image = (plt_colormap(uncertainty_norm, 'hot') * 255).astype(np.uint8)
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error (uncertainty): {e}")
    
    def sync_callback(self, rgb_msg, semantic_msg):
        """Synchronized callback for RGB and semantic."""
        self.rgb_callback(rgb_msg)
        self.semantic_callback(semantic_msg)
    
    def update_visualization(self, event):
        """Update and publish visualization."""
        with self.lock:
            if self.rgb_image is None and self.semantic_color is None:
                return
            
            # Create visualization
            viz = self.create_visualization()
            if viz is None:
                return
            
            # Publish
            try:
                viz_msg = self.bridge.cv2_to_imgmsg(viz, 'bgr8')
                self.viz_pub.publish(viz_msg)
            except CvBridgeError as e:
                rospy.logerr(f"CV Bridge error (viz): {e}")
            
            # Create and publish overlay
            if self.rgb_image is not None and self.semantic_color is not None:
                overlay = self.create_overlay()
                if overlay is not None:
                    try:
                        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, 'bgr8')
                        self.overlay_pub.publish(overlay_msg)
                    except CvBridgeError as e:
                        rospy.logerr(f"CV Bridge error (overlay): {e}")
            
            # Save frame
            if self.save_frames:
                self.save_frame(viz)
            
            self.frame_count += 1
    
    def create_visualization(self):
        """Create side-by-side or grid visualization."""
        panels = []
        labels = []
        
        # Collect available images
        if self.rgb_image is not None:
            panels.append(self.rgb_image.copy())
            labels.append('RGB')
        
        if self.depth_image is not None:
            panels.append(self.depth_image.copy())
            labels.append('Depth')
        
        if self.semantic_color is not None:
            panels.append(self.semantic_color.copy())
            labels.append('Semantic')
        
        if self.uncertainty_image is not None:
            panels.append(self.uncertainty_image.copy())
            labels.append('Uncertainty')
        
        if len(panels) == 0:
            return None
        
        # Resize all panels to same size
        target_h, target_w = panels[0].shape[:2]
        for i in range(len(panels)):
            if panels[i].shape[:2] != (target_h, target_w):
                panels[i] = cv2.resize(panels[i], (target_w, target_h))
            # Ensure 3 channels
            if len(panels[i].shape) == 2:
                panels[i] = cv2.cvtColor(panels[i], cv2.COLOR_GRAY2BGR)
        
        # Add labels
        if self.show_labels:
            for i, (panel, label) in enumerate(zip(panels, labels)):
                cv2.putText(panel, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(panel, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Combine panels
        if self.layout == 'horizontal':
            viz = np.hstack(panels)
        elif self.layout == 'vertical':
            viz = np.vstack(panels)
        elif self.layout == 'grid':
            # 2x2 grid
            if len(panels) == 1:
                viz = panels[0]
            elif len(panels) == 2:
                viz = np.hstack(panels)
            elif len(panels) == 3:
                # Add blank panel
                blank = np.zeros_like(panels[0])
                top = np.hstack([panels[0], panels[1]])
                bottom = np.hstack([panels[2], blank])
                viz = np.vstack([top, bottom])
            else:
                top = np.hstack([panels[0], panels[1]])
                bottom = np.hstack([panels[2], panels[3]])
                viz = np.vstack([top, bottom])
        else:
            viz = np.hstack(panels)
        
        # Resize to output width
        h, w = viz.shape[:2]
        if w != self.output_width:
            scale = self.output_width / w
            new_h = int(h * scale)
            viz = cv2.resize(viz, (self.output_width, new_h))
        
        # Add legend
        if self.show_legend and self.semantic_image is not None:
            viz = self.add_legend(viz)
        
        return viz
    
    def create_overlay(self):
        """Create RGB with semantic overlay."""
        if self.rgb_image is None or self.semantic_color is None:
            return None
        
        rgb = self.rgb_image.copy()
        semantic = self.semantic_color.copy()
        
        # Resize if needed
        if rgb.shape[:2] != semantic.shape[:2]:
            semantic = cv2.resize(semantic, (rgb.shape[1], rgb.shape[0]))
        
        # Blend
        overlay = cv2.addWeighted(rgb, 1 - self.overlay_alpha, semantic, self.overlay_alpha, 0)
        
        return overlay
    
    def add_legend(self, viz):
        """Add class color legend to visualization."""
        if self.semantic_image is None:
            return viz
        
        # Find unique classes in current frame
        unique_classes = np.unique(self.semantic_image)
        unique_classes = unique_classes[unique_classes < len(NYUV2_CLASSES)]
        
        if len(unique_classes) == 0:
            return viz
        
        # Create legend
        legend_h = min(len(unique_classes) * 25 + 20, 400)
        legend_w = 180
        legend = np.zeros((legend_h, legend_w, 3), dtype=np.uint8)
        legend[:] = (40, 40, 40)  # Dark background
        
        y = 15
        for cls_id in unique_classes[:15]:  # Limit to 15 classes
            color = NYUV2_COLORS[cls_id].tolist()
            name = NYUV2_CLASSES[cls_id]
            
            # Color box
            cv2.rectangle(legend, (10, y - 10), (30, y + 10), color[::-1], -1)
            cv2.rectangle(legend, (10, y - 10), (30, y + 10), (255, 255, 255), 1)
            
            # Class name
            cv2.putText(legend, f"{cls_id}: {name}", (40, y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            y += 25
        
        # Overlay legend on visualization
        h, w = viz.shape[:2]
        x_offset = w - legend_w - 10
        y_offset = 10
        
        # Ensure legend fits
        if y_offset + legend_h > h:
            legend_h = h - y_offset - 10
            legend = legend[:legend_h]
        
        if x_offset > 0 and legend_h > 0:
            # Semi-transparent overlay
            roi = viz[y_offset:y_offset+legend_h, x_offset:x_offset+legend_w]
            blended = cv2.addWeighted(roi, 0.3, legend, 0.7, 0)
            viz[y_offset:y_offset+legend_h, x_offset:x_offset+legend_w] = blended
        
        return viz
    
    def save_frame(self, viz):
        """Save frame to disk."""
        filename = f"{self.save_path}/frame_{self.frame_count:06d}.png"
        cv2.imwrite(filename, viz)


def colorize_labels(labels):
    """Convert label map to RGB color image using NYUv2 palette."""
    # Clip to valid range
    labels_clipped = np.clip(labels, 0, len(NYUV2_COLORS) - 1)
    # Index into color map (RGB)
    colored = NYUV2_COLORS[labels_clipped]
    # Convert RGB to BGR for OpenCV
    return colored[:, :, ::-1]


def plt_colormap(data, cmap='viridis'):
    """Apply matplotlib-style colormap without importing matplotlib."""
    # Simple colormaps
    if cmap == 'viridis':
        # Viridis approximation
        r = np.clip(0.267004 + data * (0.993248 - 0.267004), 0, 1)
        g = np.clip(0.004874 + data * (0.906157 - 0.004874), 0, 1)
        b = np.clip(0.329415 + data * (0.143936 - 0.329415), 0, 1)
    elif cmap == 'hot':
        # Hot colormap
        r = np.clip(data * 3, 0, 1)
        g = np.clip(data * 3 - 1, 0, 1)
        b = np.clip(data * 3 - 2, 0, 1)
    elif cmap == 'jet':
        # Jet approximation
        r = np.clip(1.5 - np.abs(data - 0.75) * 4, 0, 1)
        g = np.clip(1.5 - np.abs(data - 0.5) * 4, 0, 1)
        b = np.clip(1.5 - np.abs(data - 0.25) * 4, 0, 1)
    else:
        # Grayscale fallback
        r = g = b = data
    
    return np.stack([b, g, r], axis=-1)  # BGR for OpenCV


def main():
    try:
        node = SegmentationVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()