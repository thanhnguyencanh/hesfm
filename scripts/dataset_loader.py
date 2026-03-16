#!/usr/bin/env python3
"""
Dataset Loader for HESFM Evaluation

Loads NYUv2 and SUN RGB-D datasets and publishes as ROS messages.

Author: Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
Date: 2026
"""

import rospy
import numpy as np
import cv2
import os
import glob
from threading import Thread

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge


class NYUv2Loader:
    """NYUv2 dataset loader."""
    
    # NYUv2 camera intrinsics
    FX = 518.8579
    FY = 519.4696
    CX = 325.5824
    CY = 253.7362
    
    def __init__(self, data_path, split='test'):
        self.data_path = data_path
        self.split = split
        self.samples = self._load_samples()
        rospy.loginfo(f"Loaded {len(self.samples)} samples from NYUv2 {split}")
        
    def _load_samples(self):
        """Load sample list."""
        samples = []
        
        # Standard NYUv2 structure
        rgb_dir = os.path.join(self.data_path, 'rgb')
        depth_dir = os.path.join(self.data_path, 'depth')
        label_dir = os.path.join(self.data_path, 'label40')
        
        if os.path.exists(rgb_dir):
            rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
            for rgb_path in rgb_files:
                name = os.path.basename(rgb_path).replace('.png', '')
                depth_path = os.path.join(depth_dir, f'{name}.png')
                label_path = os.path.join(label_dir, f'{name}.png')
                
                if os.path.exists(depth_path):
                    samples.append({
                        'rgb': rgb_path,
                        'depth': depth_path,
                        'label': label_path if os.path.exists(label_path) else None,
                        'name': name
                    })
                    
        return samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        rgb = cv2.imread(sample['rgb'])
        depth = cv2.imread(sample['depth'], cv2.IMREAD_UNCHANGED)
        
        # Convert depth to meters
        if depth is not None:
            depth = depth.astype(np.float32) / 1000.0
            
        label = None
        if sample['label'] and os.path.exists(sample['label']):
            label = cv2.imread(sample['label'], cv2.IMREAD_UNCHANGED)
            
        return {
            'rgb': rgb,
            'depth': depth,
            'label': label,
            'name': sample['name']
        }
        
    def get_camera_info(self):
        """Get camera intrinsics."""
        info = CameraInfo()
        info.width = 640
        info.height = 480
        info.K = [self.FX, 0, self.CX, 0, self.FY, self.CY, 0, 0, 1]
        info.D = [0, 0, 0, 0, 0]
        info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        info.P = [self.FX, 0, self.CX, 0, 0, self.FY, self.CY, 0, 0, 0, 1, 0]
        return info


class SUNRGBDLoader:
    """SUN RGB-D dataset loader."""
    
    def __init__(self, data_path, split='test'):
        self.data_path = data_path
        self.split = split
        self.samples = self._load_samples()
        rospy.loginfo(f"Loaded {len(self.samples)} samples from SUN RGB-D {split}")
        
    def _load_samples(self):
        """Load sample list."""
        samples = []
        
        # SUN RGB-D structure varies, try common patterns
        image_dir = os.path.join(self.data_path, 'image')
        depth_dir = os.path.join(self.data_path, 'depth')
        label_dir = os.path.join(self.data_path, 'label')
        
        if os.path.exists(image_dir):
            image_files = sorted(glob.glob(os.path.join(image_dir, '*.*')))
            for img_path in image_files:
                name = os.path.splitext(os.path.basename(img_path))[0]
                
                # Find matching depth
                depth_path = None
                for ext in ['.png', '.jpg', '.pgm']:
                    p = os.path.join(depth_dir, name + ext)
                    if os.path.exists(p):
                        depth_path = p
                        break
                        
                # Find matching label
                label_path = None
                for ext in ['.png', '.mat']:
                    p = os.path.join(label_dir, name + ext)
                    if os.path.exists(p):
                        label_path = p
                        break
                        
                if depth_path:
                    samples.append({
                        'rgb': img_path,
                        'depth': depth_path,
                        'label': label_path,
                        'name': name
                    })
                    
        return samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        rgb = cv2.imread(sample['rgb'])
        depth = cv2.imread(sample['depth'], cv2.IMREAD_UNCHANGED)
        
        if depth is not None:
            # Handle different depth formats
            if depth.max() > 100:
                depth = depth.astype(np.float32) / 10000.0  # Shift to meters
            else:
                depth = depth.astype(np.float32)
                
        label = None
        if sample['label'] and os.path.exists(sample['label']):
            if sample['label'].endswith('.mat'):
                try:
                    import scipy.io
                    mat = scipy.io.loadmat(sample['label'])
                    label = mat.get('seglabel', mat.get('labels', None))
                except:
                    pass
            else:
                label = cv2.imread(sample['label'], cv2.IMREAD_UNCHANGED)
                
        return {
            'rgb': rgb,
            'depth': depth,
            'label': label,
            'name': sample['name']
        }
        
    def get_camera_info(self):
        """Get camera intrinsics (approximate)."""
        info = CameraInfo()
        info.width = 640
        info.height = 480
        info.K = [570.0, 0, 320.0, 0, 570.0, 240.0, 0, 0, 1]
        info.D = [0, 0, 0, 0, 0]
        info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        info.P = [570.0, 0, 320.0, 0, 0, 570.0, 240.0, 0, 0, 0, 1, 0]
        return info


class DatasetLoaderNode:
    """ROS node for dataset loading."""
    
    def __init__(self):
        rospy.init_node('dataset_loader', anonymous=False)
        
        self.bridge = CvBridge()
        
        # Parameters
        self.dataset = rospy.get_param('~dataset', 'nyuv2')
        self.data_path = rospy.get_param('~data_path', '')
        self.split = rospy.get_param('~split', 'test')
        self.rate = rospy.get_param('~rate', 1.0)
        self.publish_gt = rospy.get_param('~publish_gt', True)
        self.loop = rospy.get_param('~loop', False)
        
        # Initialize dataset
        if self.dataset == 'nyuv2':
            self.loader = NYUv2Loader(self.data_path, self.split)
        elif self.dataset == 'sunrgbd':
            self.loader = SUNRGBDLoader(self.data_path, self.split)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
            
        # Publishers
        self.rgb_pub = rospy.Publisher('color/image_raw', Image, queue_size=1)
        self.depth_pub = rospy.Publisher('depth/image_raw', Image, queue_size=1)
        self.camera_info_pub = rospy.Publisher('camera_info', CameraInfo, queue_size=1)
        
        if self.publish_gt:
            self.gt_pub = rospy.Publisher('semantic_gt', Image, queue_size=1)
            
        self.current_idx = 0
        self.camera_info = self.loader.get_camera_info()
        
        rospy.loginfo(f"Dataset loader initialized: {self.dataset}")
        
    def publish_sample(self, idx):
        """Publish a single sample."""
        if idx >= len(self.loader):
            if self.loop:
                idx = idx % len(self.loader)
            else:
                rospy.loginfo("Dataset complete")
                return False
                
        sample = self.loader[idx]
        
        stamp = rospy.Time.now()
        header = Header()
        header.stamp = stamp
        header.frame_id = "camera_color_optical_frame"
        
        # Publish RGB
        if sample['rgb'] is not None:
            rgb_msg = self.bridge.cv2_to_imgmsg(sample['rgb'], encoding='bgr8')
            rgb_msg.header = header
            self.rgb_pub.publish(rgb_msg)
            
        # Publish depth
        if sample['depth'] is not None:
            depth_msg = self.bridge.cv2_to_imgmsg(sample['depth'], encoding='32FC1')
            depth_msg.header = header
            self.depth_pub.publish(depth_msg)
            
        # Publish camera info
        self.camera_info.header = header
        self.camera_info_pub.publish(self.camera_info)
        
        # Publish ground truth
        if self.publish_gt and sample['label'] is not None:
            gt_msg = self.bridge.cv2_to_imgmsg(sample['label'].astype(np.uint8), encoding='mono8')
            gt_msg.header = header
            self.gt_pub.publish(gt_msg)
            
        rospy.loginfo(f"Published sample {idx + 1}/{len(self.loader)}: {sample['name']}")
        return True
        
    def run(self):
        """Run the node."""
        rate = rospy.Rate(self.rate)
        
        while not rospy.is_shutdown():
            if not self.publish_sample(self.current_idx):
                break
            self.current_idx += 1
            rate.sleep()


def main():
    try:
        node = DatasetLoaderNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
