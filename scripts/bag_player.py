#!/usr/bin/env python3
"""
Bag Player Utility for HESFM

Provides controlled playback of rosbag files with optional
frame-by-frame stepping and topic remapping.

Usage:
    rosrun hesfm bag_player.py _bag_file:=/path/to/bag.bag
    
    # With remapping
    rosrun hesfm bag_player.py _bag_file:=/path/to/bag.bag \
        _remap_color:=/camera/color/image_raw \
        _remap_depth:=/camera/depth/image_rect_raw

Author: Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
Date: 2026
"""

import rospy
import rosbag
import numpy as np
from typing import Dict, List, Optional
import threading
import time

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_srvs.srv import Trigger, TriggerResponse, SetBool, SetBoolResponse
from std_msgs.msg import Bool, Float32


class BagPlayerNode:
    """Controlled rosbag playback"""
    
    def __init__(self):
        rospy.init_node('bag_player', anonymous=False)
        
        # Parameters
        self.bag_file = rospy.get_param('~bag_file', '')
        self.loop = rospy.get_param('~loop', False)
        self.rate_multiplier = rospy.get_param('~rate', 1.0)
        self.start_time = rospy.get_param('~start_time', 0.0)
        self.end_time = rospy.get_param('~end_time', -1.0)
        self.paused = rospy.get_param('~start_paused', False)
        
        # Topic remapping
        self.remap_color = rospy.get_param('~remap_color', '/color/image_raw')
        self.remap_depth = rospy.get_param('~remap_depth', '/depth/image_rect_raw')
        self.remap_camera_info = rospy.get_param('~remap_camera_info', '/color/camera_info')
        
        # State
        self.bag = None
        self.publishers = {}
        self.current_time = 0.0
        self.total_duration = 0.0
        self.frame_count = 0
        self.lock = threading.Lock()
        self.step_mode = False
        self.step_requested = False
        
        # Load bag
        if not self._load_bag():
            rospy.logerr("Failed to load bag file")
            return
        
        # Setup ROS
        self._setup_publishers()
        self._setup_services()
        self._setup_status_publishers()
        
        rospy.loginfo("Bag Player initialized")
        rospy.loginfo(f"  Bag file: {self.bag_file}")
        rospy.loginfo(f"  Duration: {self.total_duration:.2f}s")
        rospy.loginfo(f"  Topics: {len(self.publishers)}")
    
    def _load_bag(self) -> bool:
        """Load rosbag file"""
        if not self.bag_file:
            rospy.logerr("No bag file specified")
            return False
        
        try:
            self.bag = rosbag.Bag(self.bag_file, 'r')
            
            # Get bag info
            info = self.bag.get_type_and_topic_info()
            self.topics = info.topics
            
            # Get duration
            start_time = self.bag.get_start_time()
            end_time = self.bag.get_end_time()
            self.total_duration = end_time - start_time
            self.bag_start_time = start_time
            
            # Apply time limits
            if self.end_time < 0:
                self.end_time = self.total_duration
            
            rospy.loginfo(f"Loaded bag: {self.bag_file}")
            for topic, info in self.topics.items():
                rospy.loginfo(f"  {topic}: {info.msg_type} ({info.message_count} msgs)")
            
            return True
            
        except Exception as e:
            rospy.logerr(f"Failed to load bag: {e}")
            return False
    
    def _setup_publishers(self):
        """Setup publishers for bag topics"""
        type_map = {
            'sensor_msgs/Image': Image,
            'sensor_msgs/CameraInfo': CameraInfo,
            'sensor_msgs/PointCloud2': PointCloud2,
        }
        
        for topic, info in self.topics.items():
            if info.msg_type in type_map:
                # Apply remapping
                output_topic = topic
                if 'color' in topic.lower() and 'image' in topic.lower():
                    output_topic = self.remap_color
                elif 'depth' in topic.lower() and 'image' in topic.lower():
                    output_topic = self.remap_depth
                elif 'camera_info' in topic.lower():
                    output_topic = self.remap_camera_info
                
                msg_type = type_map[info.msg_type]
                self.publishers[topic] = rospy.Publisher(
                    output_topic, msg_type, queue_size=10)
                
                rospy.loginfo(f"  {topic} -> {output_topic}")
    
    def _setup_services(self):
        """Setup control services"""
        self.pause_srv = rospy.Service('~pause', SetBool, self._pause_callback)
        self.step_srv = rospy.Service('~step', Trigger, self._step_callback)
        self.restart_srv = rospy.Service('~restart', Trigger, self._restart_callback)
        self.set_rate_srv = rospy.Service('~set_rate', SetBool, self._set_rate_callback)
    
    def _setup_status_publishers(self):
        """Setup status publishers"""
        self.progress_pub = rospy.Publisher('~progress', Float32, queue_size=1)
        self.playing_pub = rospy.Publisher('~playing', Bool, queue_size=1)
    
    def _pause_callback(self, req) -> SetBoolResponse:
        """Pause/resume playback"""
        with self.lock:
            self.paused = req.data
        
        state = "paused" if self.paused else "playing"
        return SetBoolResponse(success=True, message=f"Playback {state}")
    
    def _step_callback(self, req) -> TriggerResponse:
        """Step one frame forward"""
        with self.lock:
            self.step_mode = True
            self.step_requested = True
        
        return TriggerResponse(success=True, message="Stepping one frame")
    
    def _restart_callback(self, req) -> TriggerResponse:
        """Restart playback from beginning"""
        with self.lock:
            self.current_time = self.start_time
            self.frame_count = 0
        
        return TriggerResponse(success=True, message="Playback restarted")
    
    def _set_rate_callback(self, req) -> SetBoolResponse:
        """Set playback rate (data field reused as rate)"""
        # Note: This is a hack - normally would use a custom service
        with self.lock:
            self.rate_multiplier = 1.0 if req.data else 0.5
        
        return SetBoolResponse(success=True, 
                               message=f"Rate set to {self.rate_multiplier}x")
    
    def play(self):
        """Main playback loop"""
        rospy.loginfo("Starting playback...")
        
        rate = rospy.Rate(100)  # 100 Hz update rate
        last_msg_time = None
        
        for topic, msg, t in self.bag.read_messages():
            if rospy.is_shutdown():
                break
            
            # Convert timestamp
            msg_time = t.to_sec() - self.bag_start_time
            
            # Skip if before start time
            if msg_time < self.start_time:
                continue
            
            # Stop if after end time
            if msg_time > self.end_time:
                if self.loop:
                    rospy.loginfo("Looping playback...")
                    self.bag.close()
                    self.bag = rosbag.Bag(self.bag_file, 'r')
                    return self.play()
                else:
                    break
            
            # Handle pause
            while self.paused and not rospy.is_shutdown():
                if self.step_mode and self.step_requested:
                    with self.lock:
                        self.step_requested = False
                    break
                rate.sleep()
                self._publish_status(msg_time)
            
            # Timing
            if last_msg_time is not None:
                dt = (msg_time - last_msg_time) / self.rate_multiplier
                if dt > 0:
                    rospy.sleep(dt)
            
            last_msg_time = msg_time
            
            # Publish
            if topic in self.publishers:
                try:
                    # Update timestamp to current time
                    if hasattr(msg, 'header'):
                        msg.header.stamp = rospy.Time.now()
                    
                    self.publishers[topic].publish(msg)
                    self.frame_count += 1
                    
                except Exception as e:
                    rospy.logwarn(f"Failed to publish {topic}: {e}")
            
            # Update state
            with self.lock:
                self.current_time = msg_time
            
            self._publish_status(msg_time)
        
        rospy.loginfo(f"Playback complete. {self.frame_count} messages published.")
    
    def _publish_status(self, current_time: float):
        """Publish playback status"""
        # Progress (0-1)
        progress = current_time / self.total_duration if self.total_duration > 0 else 0
        self.progress_pub.publish(Float32(progress))
        
        # Playing state
        self.playing_pub.publish(Bool(not self.paused))
    
    def run(self):
        """Run node"""
        if self.bag is None:
            return
        
        try:
            self.play()
        finally:
            if self.bag:
                self.bag.close()


def main():
    try:
        node = BagPlayerNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
