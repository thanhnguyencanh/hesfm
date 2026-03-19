#!/usr/bin/env python3
"""
Compute depth_mean and depth_std from a live RealSense depth stream.

Usage:
    rosrun hesfm compute_depth_stats.py [_topic:=/camera/depth/image_rect_raw] [_n_frames:=200]

Decodes the depth image directly from ROS message bytes — no cv_bridge required.
"""

import rospy
import numpy as np
from sensor_msgs.msg import Image


class DepthStatsCollector:
    def __init__(self):
        rospy.init_node('compute_depth_stats', anonymous=False)

        self.n_frames = rospy.get_param('~n_frames', 200)
        topic = rospy.get_param('~topic', '/camera/depth/image_rect_raw')

        self.pixel_sum = 0.0
        self.pixel_sum_sq = 0.0
        self.pixel_count = 0
        self.frame_count = 0

        rospy.loginfo(f"Collecting {self.n_frames} frames from '{topic}' ...")
        self.sub = rospy.Subscriber(topic, Image, self.callback, queue_size=1)

    def _decode(self, msg):
        """Decode a sensor_msgs/Image to a float32 numpy array in mm."""
        raw = np.frombuffer(msg.data, dtype=np.uint8)

        if msg.encoding == '16UC1':
            # 2 bytes per pixel, little-endian unsigned short → already in mm
            depth = raw.view(np.uint16).reshape(msg.height, msg.width).astype(np.float32)
        elif msg.encoding == '32FC1':
            # 4 bytes per pixel, float32 in metres → convert to mm
            depth = raw.view(np.float32).reshape(msg.height, msg.width) * 1000.0
        else:
            rospy.logerr_once(f"Unsupported encoding: {msg.encoding}")
            return None

        return depth

    def callback(self, msg):
        if self.frame_count >= self.n_frames:
            return

        depth = self._decode(msg)
        if depth is None:
            return

        valid = depth[depth > 0].astype(np.float64)
        if valid.size == 0:
            return

        self.pixel_sum    += valid.sum()
        self.pixel_sum_sq += (valid ** 2).sum()
        self.pixel_count  += valid.size
        self.frame_count  += 1

        if self.frame_count % 20 == 0:
            rospy.loginfo(
                f"  Frame {self.frame_count}/{self.n_frames} — "
                f"this frame: mean={valid.mean():.1f} mm, std={valid.std():.1f} mm, "
                f"range=[{valid.min():.0f}, {valid.max():.0f}] mm"
            )

        if self.frame_count >= self.n_frames:
            self._report()

    def _report(self):
        self.sub.unregister()

        mean     = self.pixel_sum / self.pixel_count
        variance = (self.pixel_sum_sq / self.pixel_count) - mean ** 2
        std      = np.sqrt(max(variance, 0.0))

        rospy.loginfo("=" * 60)
        rospy.loginfo("RESULTS  (valid pixels only, values in mm)")
        rospy.loginfo(f"  Frames collected : {self.frame_count}")
        rospy.loginfo(f"  Valid pixels     : {self.pixel_count:,}")
        rospy.loginfo(f"  depth_mean       : {mean:.5f}")
        rospy.loginfo(f"  depth_std        : {std:.5f}")
        rospy.loginfo("")
        rospy.loginfo("Reference values used by ESANet training:")
        rospy.loginfo("  NYUv2   depth_mean=2841.94941  depth_std=1417.25943")
        rospy.loginfo("  SUNRGBD depth_mean=19025.14930 depth_std=9880.91607")
        rospy.loginfo("=" * 60)
        rospy.signal_shutdown("Done")


def main():
    collector = DepthStatsCollector()
    rospy.spin()


if __name__ == '__main__':
    main()
