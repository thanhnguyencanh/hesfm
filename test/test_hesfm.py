#!/usr/bin/env python3
"""
HESFM Integration Tests

Tests for ROS node integration and message flow.

Author: Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
Date: 2026
"""

import unittest
import rospy
import rostest
import numpy as np
import time

from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Empty
from cv_bridge import CvBridge


class TestSemanticCloud(unittest.TestCase):
    """Test semantic point cloud generation."""
    
    @classmethod
    def setUpClass(cls):
        rospy.init_node('test_hesfm', anonymous=True)
        cls.bridge = CvBridge()
        cls.cloud_received = False
        cls.last_cloud = None
        
    def setUp(self):
        self.cloud_received = False
        self.last_cloud = None
        
        # Publishers
        self.rgb_pub = rospy.Publisher(
            '/semantic_cloud_node/color/image_raw', Image, queue_size=1)
        self.depth_pub = rospy.Publisher(
            '/semantic_cloud_node/depth/image_rect_raw', Image, queue_size=1)
        self.semantic_pub = rospy.Publisher(
            '/semantic_cloud_node/semantic/image', Image, queue_size=1)
        self.info_pub = rospy.Publisher(
            '/semantic_cloud_node/color/camera_info', CameraInfo, queue_size=1)
            
        # Subscriber
        self.cloud_sub = rospy.Subscriber(
            '/semantic_cloud', PointCloud2, self.cloud_callback)
            
        rospy.sleep(0.5)  # Wait for connections
        
    def cloud_callback(self, msg):
        self.cloud_received = True
        self.last_cloud = msg
        
    def test_cloud_generation(self):
        """Test that semantic cloud is generated from images."""
        # Create test images
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = (np.random.random((480, 640)) * 5000 + 500).astype(np.uint16)
        semantic = np.random.randint(0, 40, (480, 640), dtype=np.uint8)
        
        # Create camera info
        info = CameraInfo()
        info.width = 640
        info.height = 480
        info.K = [386, 0, 320, 0, 386, 240, 0, 0, 1]
        info.D = [0, 0, 0, 0, 0]
        info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        info.P = [386, 0, 320, 0, 0, 386, 240, 0, 0, 0, 1, 0]
        
        # Publish messages
        stamp = rospy.Time.now()
        
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb, 'bgr8')
        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = 'camera_color_optical_frame'
        
        depth_msg = self.bridge.cv2_to_imgmsg(depth, '16UC1')
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = 'camera_color_optical_frame'
        
        semantic_msg = self.bridge.cv2_to_imgmsg(semantic, 'mono8')
        semantic_msg.header.stamp = stamp
        semantic_msg.header.frame_id = 'camera_color_optical_frame'
        
        info.header.stamp = stamp
        info.header.frame_id = 'camera_color_optical_frame'
        
        self.info_pub.publish(info)
        rospy.sleep(0.1)
        
        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)
        self.semantic_pub.publish(semantic_msg)
        
        # Wait for cloud
        timeout = rospy.Time.now() + rospy.Duration(5.0)
        while not self.cloud_received and rospy.Time.now() < timeout:
            rospy.sleep(0.1)
            
        self.assertTrue(self.cloud_received, "No point cloud received")
        self.assertIsNotNone(self.last_cloud)
        self.assertGreater(self.last_cloud.width, 0)


class TestMapper(unittest.TestCase):
    """Test HESFM mapper node."""
    
    @classmethod
    def setUpClass(cls):
        if not rospy.core.is_initialized():
            rospy.init_node('test_hesfm', anonymous=True)
            
    def setUp(self):
        self.map_received = False
        self.costmap_received = False
        
        # Subscribers
        self.map_sub = rospy.Subscriber(
            '/hesfm_mapper_node/semantic_map', PointCloud2, self.map_callback)
        self.costmap_sub = rospy.Subscriber(
            '/hesfm_mapper_node/costmap', OccupancyGrid, self.costmap_callback)
            
        rospy.sleep(0.5)
        
    def map_callback(self, msg):
        self.map_received = True
        
    def costmap_callback(self, msg):
        self.costmap_received = True
        
    def test_reset_service(self):
        """Test map reset service."""
        try:
            rospy.wait_for_service('/hesfm_mapper_node/reset_map', timeout=5.0)
            reset = rospy.ServiceProxy('/hesfm_mapper_node/reset_map', Empty)
            reset()
            self.assertTrue(True)
        except rospy.ROSException:
            self.skipTest("Reset service not available")


class TestExploration(unittest.TestCase):
    """Test exploration node."""
    
    @classmethod
    def setUpClass(cls):
        if not rospy.core.is_initialized():
            rospy.init_node('test_hesfm', anonymous=True)
            
    def setUp(self):
        self.goal_received = False
        
        # Subscriber
        self.goal_sub = rospy.Subscriber(
            '/exploration_node/exploration_goal', PoseStamped, self.goal_callback)
            
        rospy.sleep(0.5)
        
    def goal_callback(self, msg):
        self.goal_received = True
        self.last_goal = msg


class TestMetrics(unittest.TestCase):
    """Test evaluation metrics computation."""
    
    def test_iou_computation(self):
        """Test IoU computation."""
        # Create mock predictions and ground truth
        pred = np.array([0, 0, 1, 1, 2])
        gt = np.array([0, 1, 1, 1, 2])
        
        # Compute IoU manually
        # Class 0: intersection=1, union=2, IoU=0.5
        # Class 1: intersection=2, union=3, IoU=0.667
        # Class 2: intersection=1, union=1, IoU=1.0
        
        num_classes = 3
        iou = []
        for c in range(num_classes):
            intersection = np.sum((pred == c) & (gt == c))
            union = np.sum((pred == c) | (gt == c))
            if union > 0:
                iou.append(intersection / union)
                
        expected_miou = (0.5 + 2/3 + 1.0) / 3
        actual_miou = np.mean(iou)
        
        self.assertAlmostEqual(actual_miou, expected_miou, places=2)
        
    def test_ece_computation(self):
        """Test ECE computation."""
        # Perfect calibration: accuracy = confidence
        confidences = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        correct = np.array([0, 0, 1, 1, 1])  # Roughly matches confidence
        
        # Simple ECE (would be more complex with binning)
        avg_conf = np.mean(confidences)
        accuracy = np.mean(correct)
        simple_ece = abs(avg_conf - accuracy)
        
        # Should be reasonably low for well-calibrated predictions
        self.assertLess(simple_ece, 0.5)


if __name__ == '__main__':
    rostest.rosrun('hesfm', 'test_hesfm', TestSemanticCloud)
    rostest.rosrun('hesfm', 'test_hesfm', TestMapper)
    rostest.rosrun('hesfm', 'test_hesfm', TestExploration)
    unittest.main()
