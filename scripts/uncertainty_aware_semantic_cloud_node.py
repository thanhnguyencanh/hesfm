#!/usr/bin/env python3
"""
Builds an uncertainty-aware semantic point cloud from RGB-D + evidential
segmentation outputs.

Each point in the published PointCloud2 carries:
    x, y, z              (float32) - 3D position in camera frame
    rgb                  (float32, packed) - RGB for visualization
    label                (uint16)  - argmax class id
    confidence           (float32) - max class probability  in [0, 1]
    uncertainty          (float32) - Dirichlet vacuity      in [0, 1]
    probs                (float32 x C) - full class distribution (optional)

Compared to the original `semantic_cloud_node.py`, this version:
  1. Subscribes to /hesfm/semantic_uncertainty additionally.
  2. Drops points where uncertainty exceeds U_thr (Eq. 15 in Kim et al.).
  3. Carries the per-pixel probability vector through to the cloud, enabling
     the Continuous Categorical BKI update (Eq. 14) downstream.

Author: HESFM @ JAIST
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import rospy
import message_filters

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header


class UncertaintyAwareSemanticCloudNode:

    def __init__(self):
        rospy.init_node("uncertainty_aware_semantic_cloud_node")

        self.num_classes      = rospy.get_param("~num_classes", 40)
        self.depth_min_m      = rospy.get_param("~depth_min_m", 0.3)
        self.depth_max_m      = rospy.get_param("~depth_max_m", 6.0)
        self.subsample        = rospy.get_param("~subsample", 2)
        self.unc_threshold    = rospy.get_param("~uncertainty_threshold", 0.7)
        self.adaptive_unc     = rospy.get_param("~adaptive_threshold", True)
        self.unc_top_percent  = rospy.get_param("~uncertainty_top_percent", 0.10)
        self.publish_full_probs = rospy.get_param("~publish_full_probs", False)
        self.queue_size       = rospy.get_param("~queue_size", 5)
        self.slop             = rospy.get_param("~approx_sync_slop", 0.05)

        self.bridge = CvBridge()
        self.K: Optional[np.ndarray] = None

        depth_topic = rospy.get_param("~depth_topic",
                                      "/camera/aligned_depth_to_color/image_raw")
        info_topic  = rospy.get_param("~camera_info_topic",
                                      "/camera/color/camera_info")
        label_topic = "/hesfm/semantic_image"
        unc_topic   = "/hesfm/semantic_uncertainty"
        prob_topic  = "/hesfm/semantic_probs"

        self.info_sub = rospy.Subscriber(info_topic, CameraInfo,
                                         self._info_cb, queue_size=1)

        depth_sub = message_filters.Subscriber(depth_topic, Image)
        label_sub = message_filters.Subscriber(label_topic, Image)
        unc_sub   = message_filters.Subscriber(unc_topic,   Image)
        prob_sub  = message_filters.Subscriber(prob_topic,  Image)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [depth_sub, label_sub, unc_sub, prob_sub],
            queue_size=self.queue_size, slop=self.slop,
        )
        self.sync.registerCallback(self._cb)

        self.pub = rospy.Publisher("/hesfm/semantic_cloud", PointCloud2,
                                   queue_size=2)

        rospy.loginfo("[uncertainty_aware_semantic_cloud_node] ready "
                      f"(U_thr={self.unc_threshold}, adaptive={self.adaptive_unc})")

    def _info_cb(self, msg: CameraInfo):
        self.K = np.array(msg.K, dtype=np.float32).reshape(3, 3)
        self.info_sub.unregister()
        rospy.loginfo("[uncertainty_aware_semantic_cloud_node] camera intrinsics OK")

    def _cb(self, depth_msg, label_msg, unc_msg, prob_msg):
        if self.K is None:
            return

        depth = self.bridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding)
        label = self.bridge.imgmsg_to_cv2(label_msg, "mono8")
        unc   = self.bridge.imgmsg_to_cv2(unc_msg,   "32FC1")
        probs = self.bridge.imgmsg_to_cv2(prob_msg)

        if depth.dtype == np.uint16:
            depth_m = depth.astype(np.float32) / 1000.0
        else:
            depth_m = depth.astype(np.float32)

        H, W = depth_m.shape
        ys, xs = np.meshgrid(np.arange(0, H, self.subsample),
                             np.arange(0, W, self.subsample),
                             indexing="ij")
        ys = ys.ravel(); xs = xs.ravel()

        z = depth_m[ys, xs]
        valid = (z > self.depth_min_m) & (z < self.depth_max_m) & np.isfinite(z)

        u_flat = unc[ys, xs]
        if self.adaptive_unc:
            if valid.any():
                # Adaptive: drop the top `unc_top_percent` most uncertain
                # pixels of the frame *and* honour the fixed cap. The
                # effective threshold is therefore the stricter (smaller)
                # of the two — using max() would let MORE uncertain pixels
                # through and defeat the "drop top X%" intent.
                cutoff = np.quantile(u_flat[valid], 1.0 - self.unc_top_percent)
                u_thr = min(self.unc_threshold, cutoff)
            else:
                u_thr = self.unc_threshold
        else:
            u_thr = self.unc_threshold
        valid &= (u_flat <= u_thr)

        ys = ys[valid]; xs = xs[valid]; z = z[valid]; u_flat = u_flat[valid]

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy

        lab = label[ys, xs].astype(np.uint16)
        prob_pix = probs[ys, xs]
        if prob_pix.ndim == 1:
            prob_pix = prob_pix[:, None]
        conf = prob_pix.max(axis=1).astype(np.float32)

        rgb_u32 = (((lab.astype(np.uint32) * 2654435761) & 0xFFFFFF))
        rgb_f = rgb_u32.view(np.float32)

        cloud = self._build_cloud(
            header=Header(stamp=depth_msg.header.stamp,
                          frame_id=depth_msg.header.frame_id),
            x=x.astype(np.float32),
            y=y.astype(np.float32),
            z=z.astype(np.float32),
            rgb=rgb_f,
            label=lab,
            confidence=conf,
            uncertainty=u_flat.astype(np.float32),
            probs=prob_pix.astype(np.float32) if self.publish_full_probs else None,
        )
        self.pub.publish(cloud)

    def _build_cloud(self, header, x, y, z, rgb, label, confidence,
                     uncertainty, probs=None) -> PointCloud2:
        N = x.shape[0]
        dtype_fields = [
            ("x",            "<f4"),
            ("y",            "<f4"),
            ("z",            "<f4"),
            ("rgb",          "<f4"),
            ("label",        "<u2"),
            ("_pad0",        "<u2"),
            ("confidence",   "<f4"),
            ("uncertainty",  "<f4"),
        ]
        if probs is not None:
            C = probs.shape[1]
            dtype_fields.append(("probs", "<f4", (C,)))
        record = np.zeros(N, dtype=np.dtype(dtype_fields))
        record["x"] = x; record["y"] = y; record["z"] = z; record["rgb"] = rgb
        record["label"] = label
        record["confidence"]  = confidence
        record["uncertainty"] = uncertainty
        if probs is not None:
            record["probs"] = probs

        offset = 0
        new_fields = []
        for entry in dtype_fields:
            name = entry[0]
            fmt  = entry[1]
            if name.startswith("_"):
                offset += np.dtype(fmt).itemsize
                continue
            count = 1
            if len(entry) == 3:
                shape = entry[2]
                count = int(np.prod(shape))
                npdtype = fmt
            else:
                npdtype = fmt
            datatype = {
                "<f4": PointField.FLOAT32,
                "<u2": PointField.UINT16,
                "<u4": PointField.UINT32,
            }[npdtype]
            new_fields.append(PointField(name=name, offset=offset,
                                         datatype=datatype, count=count))
            offset += np.dtype(npdtype).itemsize * count

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width  = N
        msg.fields = new_fields
        msg.is_bigendian = False
        msg.point_step   = record.dtype.itemsize
        msg.row_step     = msg.point_step * N
        msg.is_dense     = True
        msg.data         = record.tobytes()
        return msg


def main():
    UncertaintyAwareSemanticCloudNode()
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
