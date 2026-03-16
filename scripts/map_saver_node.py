#!/usr/bin/env python3
"""
Map Saver Node for HESFM

Provides services for saving and loading semantic maps in various formats.

Services:
    - save_map (hesfm/SaveMap): Save map to file
    - load_map (hesfm/LoadMap): Load map from file

Supported formats:
    - PCD: Point Cloud Data (semantic labels as field)
    - PLY: Polygon File Format
    - YAML: Metadata + binary data
    - NPZ: NumPy compressed archive

Author: Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
Date: 2026
"""

import rospy
import numpy as np
import struct
import os
import yaml
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
import threading

from sensor_msgs.msg import PointCloud2, PointField
from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import Pose

import sensor_msgs.point_cloud2 as pc2


@dataclass
class MapPoint:
    """Single map point with semantic information"""
    x: float
    y: float
    z: float
    label: int
    confidence: float
    r: int = 128
    g: int = 128
    b: int = 128


class MapSaverNode:
    """Map saving and loading utilities"""
    
    def __init__(self):
        rospy.init_node('map_saver_node', anonymous=False)
        
        # Parameters
        self.output_dir = rospy.get_param('~output_dir', '/tmp/hesfm_maps')
        self.auto_save = rospy.get_param('~auto_save', False)
        self.auto_save_interval = rospy.get_param('~auto_save_interval', 60.0)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # State
        self.latest_map = None
        self.map_header = None
        self.lock = threading.Lock()
        
        # Setup ROS
        self._setup_subscribers()
        self._setup_services()
        
        if self.auto_save:
            self._setup_auto_save()
        
        rospy.loginfo("Map Saver Node initialized")
        rospy.loginfo(f"  Output dir: {self.output_dir}")
    
    def _setup_subscribers(self):
        """Setup subscribers"""
        self.map_sub = rospy.Subscriber(
            'semantic_map', PointCloud2, self._map_callback, queue_size=1)
    
    def _setup_services(self):
        """Setup services"""
        self.save_pcd_srv = rospy.Service(
            '~save_pcd', Trigger, self._save_pcd_callback)
        self.save_ply_srv = rospy.Service(
            '~save_ply', Trigger, self._save_ply_callback)
        self.save_npz_srv = rospy.Service(
            '~save_npz', Trigger, self._save_npz_callback)
        self.save_yaml_srv = rospy.Service(
            '~save_yaml', Trigger, self._save_yaml_callback)
    
    def _setup_auto_save(self):
        """Setup auto-save timer"""
        self.auto_save_timer = rospy.Timer(
            rospy.Duration(self.auto_save_interval),
            self._auto_save_callback)
        rospy.loginfo(f"Auto-save enabled every {self.auto_save_interval}s")
    
    def _map_callback(self, msg: PointCloud2):
        """Store latest map"""
        with self.lock:
            self.latest_map = msg
            self.map_header = msg.header
    
    def _parse_pointcloud(self, msg: PointCloud2) -> List[MapPoint]:
        """Parse PointCloud2 to list of MapPoints"""
        points = []
        
        # Get field indices
        field_names = [f.name for f in msg.fields]
        
        for point in pc2.read_points(msg, skip_nans=True):
            mp = MapPoint(
                x=point[field_names.index('x')] if 'x' in field_names else 0,
                y=point[field_names.index('y')] if 'y' in field_names else 0,
                z=point[field_names.index('z')] if 'z' in field_names else 0,
                label=int(point[field_names.index('label')]) if 'label' in field_names else 0,
                confidence=float(point[field_names.index('confidence')]) if 'confidence' in field_names else 0.8,
            )
            
            if 'rgb' in field_names:
                rgb = point[field_names.index('rgb')]
                if isinstance(rgb, float):
                    # Packed RGB
                    rgb_int = struct.unpack('I', struct.pack('f', rgb))[0]
                    mp.r = (rgb_int >> 16) & 0xFF
                    mp.g = (rgb_int >> 8) & 0xFF
                    mp.b = rgb_int & 0xFF
            
            points.append(mp)
        
        return points
    
    def _get_timestamp_filename(self, extension: str) -> str:
        """Generate filename with timestamp"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.output_dir, f"hesfm_map_{timestamp}.{extension}")
    
    def _save_pcd_callback(self, req) -> TriggerResponse:
        """Save map as PCD file"""
        with self.lock:
            if self.latest_map is None:
                return TriggerResponse(success=False, message="No map available")
            
            points = self._parse_pointcloud(self.latest_map)
        
        if not points:
            return TriggerResponse(success=False, message="Empty map")
        
        filepath = self._get_timestamp_filename("pcd")
        
        try:
            self._write_pcd(filepath, points)
            rospy.loginfo(f"Saved PCD map to {filepath}")
            return TriggerResponse(success=True, message=f"Saved to {filepath}")
        except Exception as e:
            return TriggerResponse(success=False, message=str(e))
    
    def _write_pcd(self, filepath: str, points: List[MapPoint]):
        """Write points to PCD file"""
        with open(filepath, 'w') as f:
            # Header
            f.write("# .PCD v0.7 - HESFM Semantic Map\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z rgb label confidence\n")
            f.write("SIZE 4 4 4 4 4 4\n")
            f.write("TYPE F F F U U F\n")
            f.write("COUNT 1 1 1 1 1 1\n")
            f.write(f"WIDTH {len(points)}\n")
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {len(points)}\n")
            f.write("DATA ascii\n")
            
            # Data
            for p in points:
                rgb_packed = (p.r << 16) | (p.g << 8) | p.b
                f.write(f"{p.x:.6f} {p.y:.6f} {p.z:.6f} {rgb_packed} {p.label} {p.confidence:.4f}\n")
    
    def _save_ply_callback(self, req) -> TriggerResponse:
        """Save map as PLY file"""
        with self.lock:
            if self.latest_map is None:
                return TriggerResponse(success=False, message="No map available")
            
            points = self._parse_pointcloud(self.latest_map)
        
        if not points:
            return TriggerResponse(success=False, message="Empty map")
        
        filepath = self._get_timestamp_filename("ply")
        
        try:
            self._write_ply(filepath, points)
            rospy.loginfo(f"Saved PLY map to {filepath}")
            return TriggerResponse(success=True, message=f"Saved to {filepath}")
        except Exception as e:
            return TriggerResponse(success=False, message=str(e))
    
    def _write_ply(self, filepath: str, points: List[MapPoint]):
        """Write points to PLY file"""
        with open(filepath, 'w') as f:
            # Header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("property int label\n")
            f.write("property float confidence\n")
            f.write("end_header\n")
            
            # Data
            for p in points:
                f.write(f"{p.x:.6f} {p.y:.6f} {p.z:.6f} {p.r} {p.g} {p.b} {p.label} {p.confidence:.4f}\n")
    
    def _save_npz_callback(self, req) -> TriggerResponse:
        """Save map as NPZ file"""
        with self.lock:
            if self.latest_map is None:
                return TriggerResponse(success=False, message="No map available")
            
            points = self._parse_pointcloud(self.latest_map)
        
        if not points:
            return TriggerResponse(success=False, message="Empty map")
        
        filepath = self._get_timestamp_filename("npz")
        
        try:
            # Convert to numpy arrays
            positions = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)
            labels = np.array([p.label for p in points], dtype=np.int32)
            confidences = np.array([p.confidence for p in points], dtype=np.float32)
            colors = np.array([[p.r, p.g, p.b] for p in points], dtype=np.uint8)
            
            np.savez_compressed(
                filepath,
                positions=positions,
                labels=labels,
                confidences=confidences,
                colors=colors,
                frame_id=self.map_header.frame_id if self.map_header else "map"
            )
            
            rospy.loginfo(f"Saved NPZ map to {filepath}")
            return TriggerResponse(success=True, message=f"Saved to {filepath}")
        except Exception as e:
            return TriggerResponse(success=False, message=str(e))
    
    def _save_yaml_callback(self, req) -> TriggerResponse:
        """Save map metadata as YAML with binary data"""
        with self.lock:
            if self.latest_map is None:
                return TriggerResponse(success=False, message="No map available")
            
            points = self._parse_pointcloud(self.latest_map)
        
        if not points:
            return TriggerResponse(success=False, message="Empty map")
        
        filepath = self._get_timestamp_filename("yaml")
        
        try:
            # Metadata
            metadata = {
                'format': 'hesfm_semantic_map',
                'version': '1.0',
                'frame_id': self.map_header.frame_id if self.map_header else "map",
                'num_points': len(points),
                'timestamp': rospy.Time.now().to_sec(),
                'data_file': os.path.basename(filepath).replace('.yaml', '.bin')
            }
            
            # Compute statistics
            labels = [p.label for p in points]
            unique_labels, counts = np.unique(labels, return_counts=True)
            metadata['class_distribution'] = {int(l): int(c) for l, c in zip(unique_labels, counts)}
            metadata['mean_confidence'] = float(np.mean([p.confidence for p in points]))
            
            # Save YAML
            with open(filepath, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
            
            # Save binary data
            bin_filepath = filepath.replace('.yaml', '.bin')
            self._write_binary(bin_filepath, points)
            
            rospy.loginfo(f"Saved YAML map to {filepath}")
            return TriggerResponse(success=True, message=f"Saved to {filepath}")
        except Exception as e:
            return TriggerResponse(success=False, message=str(e))
    
    def _write_binary(self, filepath: str, points: List[MapPoint]):
        """Write points to binary file"""
        with open(filepath, 'wb') as f:
            for p in points:
                # Pack: x, y, z (float32), label (int32), confidence (float32), r, g, b (uint8)
                data = struct.pack('fffifBBB', 
                                   p.x, p.y, p.z, p.label, p.confidence,
                                   p.r, p.g, p.b)
                f.write(data)
    
    def _auto_save_callback(self, event):
        """Auto-save callback"""
        if self.latest_map is not None:
            self._save_npz_callback(None)
    
    def run(self):
        """Main loop"""
        rospy.spin()


def main():
    try:
        node = MapSaverNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
