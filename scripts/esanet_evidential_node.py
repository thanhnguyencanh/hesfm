#!/usr/bin/env python3
"""
ROS Noetic node: evidential semantic segmentation with ESANet + F-EDL.

Subscribes to RGB+depth, publishes:
    /hesfm/semantic_image          [sensor_msgs/Image, mono8]   argmax label
    /hesfm/semantic_color          [sensor_msgs/Image, bgr8]    coloured viz
    /hesfm/semantic_probs          [sensor_msgs/Image, 32FC?]   per-class mu
    /hesfm/semantic_uncertainty    [sensor_msgs/Image, 32FC1]   F-EDL vacuity
    /hesfm/semantic_alpha          [sensor_msgs/Image, 32FC?]   Dirichlet alpha
                                                                (optional, large)

`semantic_probs` carries the F-EDL predictive mean mu = (alpha + tau*p)/(alpha0+tau),
which is what the downstream Continuous Categorical BKI update consumes.
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
import rospy
import message_filters
import torch
import cv2

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from evidential import EvidentialESANet, predictive_outputs

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    _HAS_TRT = True
except ImportError:
    _HAS_TRT = False


class _TorchBackend:

    def __init__(self, ckpt: str, num_classes: int, flexible: bool, device: str,
                 depth_mean: float, depth_std: float):
        try:
            from src.models.model import ESANet
        except ImportError:
            from esanet.models.model import ESANet

        self.device = torch.device(device)
        self.depth_mean = float(depth_mean)
        self.depth_std  = float(depth_std)
        backbone = ESANet(
            num_classes=num_classes,
            encoder="resnet34",
            encoder_block="NonBottleneck1D",
            modality="rgbd",
            pretrained_on_imagenet=False,
        )
        self.model = EvidentialESANet(
            backbone, num_classes=num_classes, flexible=flexible
        ).to(self.device).eval()

        state = torch.load(ckpt, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            rospy.logwarn(f"[esanet_evidential] missing keys: {len(missing)}; "
                          f"first few: {missing[:5]}")
        rospy.loginfo(f"[esanet_evidential] PyTorch backend on {self.device}")

    @torch.inference_mode()
    def __call__(self, rgb: np.ndarray, depth: np.ndarray):
        H, W = rgb.shape[:2]
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        rgb_t = (rgb_t - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) \
                / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        depth_t = torch.from_numpy(depth).float().unsqueeze(0)
        if depth_t.max() < 50:
            depth_t = depth_t * 1000.0
        depth_t = (depth_t - self.depth_mean) / self.depth_std

        rgb_t   = rgb_t.unsqueeze(0).to(self.device)
        depth_t = depth_t.unsqueeze(0).to(self.device)

        out = self.model(rgb_t, depth_t, target_size=(H, W))
        label, mu, unc = predictive_outputs(out)
        return {
            "label": label[0].cpu().numpy().astype(np.uint8),
            "prob":  mu[0].cpu().numpy(),
            "unc":   unc[0].cpu().numpy().astype(np.float32),
            "alpha": out["alpha"][0].cpu().numpy() if rospy.get_param(
                "~publish_alpha", False) else None,
        }


class _TRTBackend:
    """TensorRT-accelerated path. Uses the TRT 10 IO-tensor API."""

    def __init__(self, engine_path: str, num_classes: int,
                 depth_mean: float, depth_std: float):
        if not _HAS_TRT:
            raise RuntimeError("tensorrt not installed in this environment")
        self.num_classes = num_classes
        self.depth_mean = float(depth_mean)
        self.depth_std  = float(depth_std)

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self._allocate_buffers()
        rospy.loginfo(f"[esanet_evidential] TensorRT engine loaded: {engine_path}")

    def _allocate_buffers(self):
        self.tensor_names = []
        self.tensor_modes = {}
        self.shapes  = {}
        self.dtypes  = {}
        self.host    = {}
        self.dev     = {}
        self.stream  = cuda.Stream()

        n = self.engine.num_io_tensors
        for i in range(n):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(np.prod(shape))
            host = cuda.pagelocked_empty(size, dtype)
            dev  = cuda.mem_alloc(host.nbytes)

            self.tensor_names.append(name)
            self.tensor_modes[name] = mode
            self.shapes[name] = shape
            self.dtypes[name] = dtype
            self.host[name]   = host
            self.dev[name]    = dev
            self.context.set_tensor_address(name, int(dev))

        self.input_names = [n for n in self.tensor_names
                            if self.tensor_modes[n] == trt.TensorIOMode.INPUT]
        self.output_names = [n for n in self.tensor_names
                             if self.tensor_modes[n] == trt.TensorIOMode.OUTPUT]

        for required in ("rgb", "depth"):
            if required not in self.input_names:
                raise RuntimeError(
                    f"TRT engine is missing required input '{required}'. "
                    f"Got inputs: {self.input_names}")
        for required in ("prob", "uncertainty"):
            if required not in self.output_names:
                raise RuntimeError(
                    f"TRT engine is missing required output '{required}'. "
                    f"Got outputs: {self.output_names}")

    def __call__(self, rgb: np.ndarray, depth: np.ndarray):
        H, W = rgb.shape[:2]
        rgb_t = (rgb.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) \
                / np.array([0.229, 0.224, 0.225])
        rgb_t = rgb_t.transpose(2, 0, 1)[None]
        d = depth.astype(np.float32)
        if d.max() < 50: d *= 1000.0
        d = (d - self.depth_mean) / self.depth_std
        d = d[None, None]

        sources = {"rgb": rgb_t, "depth": d}
        for name in self.input_names:
            src = sources[name].astype(self.dtypes[name], copy=False)
            np.copyto(self.host[name], src.ravel())
            cuda.memcpy_htod_async(self.dev[name], self.host[name], self.stream)

        self.context.execute_async_v3(self.stream.handle)

        for name in self.output_names:
            cuda.memcpy_dtoh_async(self.host[name], self.dev[name], self.stream)
        self.stream.synchronize()

        prob = self.host["prob"].reshape(self.shapes["prob"])[0]
        unc  = self.host["uncertainty"].reshape(self.shapes["uncertainty"])[0]
        if prob.shape[-2:] != (H, W):
            prob = np.stack([
                cv2.resize(prob[c], (W, H), interpolation=cv2.INTER_LINEAR)
                for c in range(prob.shape[0])
            ])
            unc = cv2.resize(unc, (W, H), interpolation=cv2.INTER_LINEAR)
        label = prob.argmax(0).astype(np.uint8)
        return {"label": label, "prob": prob.astype(np.float32),
                "unc":   unc.astype(np.float32), "alpha": None}


class EvidentialSegmentationNode:

    NYUv2_40_PALETTE = np.array([
        [174, 199, 232], [152, 223, 138], [ 31, 119, 180], [255, 187, 120],
        [188, 189,  34], [140,  86,  75], [255, 152, 150], [214,  39,  40],
        [197, 176, 213], [148, 103, 189], [196, 156, 148], [ 23, 190, 207],
        [247, 182, 210], [219, 219, 141], [255, 127,  14], [158, 218, 229],
        [ 44, 160,  44], [112, 128, 144], [227, 119, 194], [ 82,  84, 163],
        [ 96, 165, 218], [183, 209, 255], [192, 192, 192], [102, 102, 102],
        [142, 124,  68], [255, 255, 153], [186, 176, 172], [254, 178, 105],
        [128, 128,   0], [  0, 128, 128], [128,   0, 128], [  0,   0, 128],
        [255, 215,   0], [123, 104, 238], [ 60, 179, 113], [221, 160, 221],
        [255, 105, 180], [188, 143, 143], [ 47,  79,  79], [255,  69,   0],
    ], dtype=np.uint8)

    def __init__(self):
        rospy.init_node("esanet_evidential_node")

        self.num_classes = rospy.get_param("~num_classes", 40)
        self.flexible    = rospy.get_param("~flexible", True)
        backend_kind     = rospy.get_param("~backend", "torch")
        self.publish_probs = rospy.get_param("~publish_probs", True)
        self.publish_alpha = rospy.get_param("~publish_alpha", False)
        self.publish_color = rospy.get_param("~publish_color", True)
        self.queue_size  = rospy.get_param("~queue_size", 5)
        self.slop        = rospy.get_param("~approx_sync_slop", 0.05)

        # Depth normalization stats (millimetres). Defaults are NYUv2-40 train.
        depth_mean = rospy.get_param("~depth_norm_mean", 2841.94941272766)
        depth_std  = rospy.get_param("~depth_norm_std",  1417.2594281672277)

        if backend_kind == "trt":
            engine = rospy.get_param("~engine_path")
            self.backend = _TRTBackend(engine, self.num_classes,
                                       depth_mean, depth_std)
        else:
            ckpt = rospy.get_param("~checkpoint")
            device = rospy.get_param("~device", "cuda:0")
            self.backend = _TorchBackend(
                ckpt, self.num_classes, self.flexible, device,
                depth_mean, depth_std,
            )
        self.bridge = CvBridge()

        rgb_topic   = rospy.get_param("~rgb_topic",
                                      "/camera/color/image_raw")
        depth_topic = rospy.get_param("~depth_topic",
                                      "/camera/aligned_depth_to_color/image_raw")
        rgb_sub   = message_filters.Subscriber(rgb_topic,   Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=self.queue_size, slop=self.slop,
        )
        self.sync.registerCallback(self._cb)

        self.pub_label  = rospy.Publisher("/hesfm/semantic_image", Image,
                                          queue_size=2)
        self.pub_color  = rospy.Publisher("/hesfm/semantic_color", Image,
                                          queue_size=2) if self.publish_color else None
        self.pub_unc    = rospy.Publisher("/hesfm/semantic_uncertainty", Image,
                                          queue_size=2)
        self.pub_probs  = rospy.Publisher("/hesfm/semantic_probs", Image,
                                          queue_size=2) if self.publish_probs else None
        self.pub_alpha  = rospy.Publisher("/hesfm/semantic_alpha", Image,
                                          queue_size=2) if self.publish_alpha else None

        self._t_last = time.time()
        self._frames = 0

        rospy.loginfo("[esanet_evidential_node] ready "
                      f"(C={self.num_classes}, flexible={self.flexible}, "
                      f"backend={backend_kind})")

    def _cb(self, rgb_msg: Image, depth_msg: Image):
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding)
        except Exception as e:
            rospy.logerr(f"cv_bridge: {e}")
            return

        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0
        else:
            depth = depth.astype(np.float32)

        t0 = time.time()
        out = self.backend(rgb, depth)
        dt = time.time() - t0

        h = Header(stamp=rgb_msg.header.stamp, frame_id=rgb_msg.header.frame_id)

        msg_label = self.bridge.cv2_to_imgmsg(out["label"], encoding="mono8")
        msg_label.header = h
        self.pub_label.publish(msg_label)

        msg_unc = self.bridge.cv2_to_imgmsg(out["unc"], encoding="32FC1")
        msg_unc.header = h
        self.pub_unc.publish(msg_unc)

        if self.pub_color is not None:
            color = self.NYUv2_40_PALETTE[out["label"] %
                                          len(self.NYUv2_40_PALETTE)]
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            msg_c = self.bridge.cv2_to_imgmsg(color, encoding="bgr8")
            msg_c.header = h
            self.pub_color.publish(msg_c)

        if self.pub_probs is not None:
            prob = np.transpose(out["prob"], (1, 2, 0)).astype(np.float32)
            msg_p = self.bridge.cv2_to_imgmsg(prob, encoding=f"32FC{prob.shape[2]}")
            msg_p.header = h
            self.pub_probs.publish(msg_p)

        if self.pub_alpha is not None and out["alpha"] is not None:
            alpha = np.transpose(out["alpha"], (1, 2, 0)).astype(np.float32)
            msg_a = self.bridge.cv2_to_imgmsg(alpha,
                                              encoding=f"32FC{alpha.shape[2]}")
            msg_a.header = h
            self.pub_alpha.publish(msg_a)

        self._frames += 1
        if time.time() - self._t_last > 5.0:
            fps = self._frames / (time.time() - self._t_last)
            rospy.loginfo(f"[esanet_evidential_node] {fps:5.1f} fps "
                          f"(latest forward {dt*1000:.1f} ms, "
                          f"u_mean={out['unc'].mean():.3f})")
            self._frames = 0
            self._t_last = time.time()


def main():
    EvidentialSegmentationNode()
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
