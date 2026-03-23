#!/usr/bin/env python3
"""
FPS and Performance Monitor for HESFM

Monitors message rates and system performance.

Author: Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
Date: 2026
"""

import os
import rospy
import numpy as np
import time
from collections import deque
from threading import Lock

from std_msgs.msg import Float32
from sensor_msgs.msg import Image, PointCloud2


class TopicMonitor:
    """Monitor message rate for a topic."""
    
    def __init__(self, topic, window_size=30):
        self.topic = topic
        self.window_size = window_size
        self.timestamps = deque(maxlen=window_size)
        self.lock = Lock()
        
        # Auto-detect message type
        self.subscriber = None
        self._setup_subscriber()
        
    def _setup_subscriber(self):
        """Setup subscriber with auto-detection."""
        # Try common message types
        for msg_type in [Image, PointCloud2]:
            try:
                self.subscriber = rospy.Subscriber(
                    self.topic, msg_type, self._callback, queue_size=1)
                rospy.loginfo(f"Monitoring {self.topic}")
                return
            except:
                continue
                
        rospy.logwarn(f"Could not subscribe to {self.topic}")
        
    def _callback(self, msg):
        """Message callback."""
        with self.lock:
            self.timestamps.append(time.time())
            
    def get_fps(self):
        """Compute current FPS."""
        with self.lock:
            if len(self.timestamps) < 2:
                return 0.0
                
            duration = self.timestamps[-1] - self.timestamps[0]
            if duration <= 0:
                return 0.0
                
            return (len(self.timestamps) - 1) / duration
            
    def get_latency_stats(self):
        """Compute latency statistics."""
        with self.lock:
            if len(self.timestamps) < 2:
                return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
                
            intervals = np.diff(list(self.timestamps))
            return {
                'mean': np.mean(intervals) * 1000,  # ms
                'std': np.std(intervals) * 1000,
                'min': np.min(intervals) * 1000,
                'max': np.max(intervals) * 1000
            }


class SystemMonitor:
    """Monitor system resources (Jetson-compatible)."""

    # Jetson sysfs paths for GPU load (Xavier AGX / Orin)
    _GPU_LOAD_PATHS = [
        '/sys/devices/gpu.0/load',           # Xavier AGX (reports 0-1000)
        '/sys/class/devfreq/17000000.gv11b/device/load',
    ]
    # Jetson thermal zones for GPU temperature
    _GPU_TEMP_PATHS = [
        '/sys/devices/virtual/thermal/thermal_zone1/temp',  # GPU zone on Xavier
        '/sys/class/thermal/thermal_zone1/temp',
    ]

    def __init__(self):
        self.has_psutil = False

        try:
            import psutil
            self.psutil = psutil
            self.has_psutil = True
        except ImportError:
            pass

        # Detect which sysfs path works for GPU load
        self._gpu_load_path = None
        for p in self._GPU_LOAD_PATHS:
            if os.path.exists(p):
                self._gpu_load_path = p
                break

        self._gpu_temp_path = None
        for p in self._GPU_TEMP_PATHS:
            if os.path.exists(p):
                self._gpu_temp_path = p
                break

    def get_cpu_usage(self):
        if self.has_psutil:
            return self.psutil.cpu_percent()
        return 0

    def get_memory_usage(self):
        if self.has_psutil:
            return self.psutil.virtual_memory().percent
        return 0

    def get_gpu_usage(self):
        """Read GPU load from Jetson sysfs (0-100%)."""
        if self._gpu_load_path:
            try:
                with open(self._gpu_load_path, 'r') as f:
                    val = int(f.read().strip())
                # Xavier reports 0-1000 (per-mille); clamp to 0-100
                return min(val / 10.0, 100.0)
            except Exception:
                pass
        return -1  # -1 means unavailable

    def get_gpu_temp(self):
        """Read GPU temperature from Jetson thermal sysfs (°C)."""
        if self._gpu_temp_path:
            try:
                with open(self._gpu_temp_path, 'r') as f:
                    millideg = int(f.read().strip())
                return millideg / 1000.0
            except Exception:
                pass
        return -1


class FPSMonitorNode:
    """ROS node for FPS monitoring."""
    
    def __init__(self):
        rospy.init_node('fps_monitor', anonymous=True)
        
        # Parameters
        self.window_size = rospy.get_param('~window_size', 30)
        self.print_interval = rospy.get_param('~print_interval', 5.0)
        topics = rospy.get_param('~topics', [])
        
        # Topic monitors
        self.monitors = {}
        for topic in topics:
            self.monitors[topic] = TopicMonitor(topic, self.window_size)
            
        # System monitor
        self.system = SystemMonitor()
        
        # Publishers
        self.fps_pub = rospy.Publisher('~fps', Float32, queue_size=1)
        
        # Timer
        self.print_timer = rospy.Timer(
            rospy.Duration(self.print_interval), self.print_callback)
            
        rospy.loginfo(f"FPS Monitor initialized, monitoring {len(self.monitors)} topics")
        
    def print_callback(self, event):
        """Print statistics."""
        rospy.loginfo("=" * 60)
        rospy.loginfo("Performance Statistics")
        rospy.loginfo("=" * 60)
        
        # Topic FPS
        for topic, monitor in self.monitors.items():
            fps = monitor.get_fps()
            latency = monitor.get_latency_stats()
            rospy.loginfo(f"{topic}:")
            rospy.loginfo(f"  FPS: {fps:.1f}")
            rospy.loginfo(f"  Latency: {latency['mean']:.1f} ± {latency['std']:.1f} ms")
            
        # System stats
        rospy.loginfo("-" * 60)
        rospy.loginfo(f"CPU Usage: {self.system.get_cpu_usage():.1f}%")
        rospy.loginfo(f"Memory Usage: {self.system.get_memory_usage():.1f}%")

        gpu_usage = self.system.get_gpu_usage()
        if gpu_usage >= 0:
            rospy.loginfo(f"GPU Usage: {gpu_usage:.1f}%")
        else:
            rospy.loginfo("GPU Usage: N/A (sysfs not found)")

        gpu_temp = self.system.get_gpu_temp()
        if gpu_temp >= 0:
            rospy.loginfo(f"GPU Temp: {gpu_temp:.1f}°C")
            
    def run(self):
        """Run the node."""
        rospy.spin()


def main():
    try:
        node = FPSMonitorNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
