#!/usr/bin/env python3
"""
Semantic Segmentation Node for HESFM

Unified semantic segmentation node supporting multiple backends:
- DFormerv2-Large (highest accuracy, RTX 4090/4080)
- ESANet-R34-NBt1D PyTorch (moderate GPU)
- ESANet-R34-NBt1D TensorRT FP16 (Jetson Orin)

Author: Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
Date: 2026
"""

import sys
import os
import time
import threading

import rospy
import numpy as np
import cv2

from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    # Patch deprecated numpy aliases before importing tensorrt/pycuda
    if not hasattr(np, 'bool'):   np.bool   = bool
    if not hasattr(np, 'int'):    np.int    = int
    if not hasattr(np, 'float'):  np.float  = float
    import tensorrt as trt
    import pycuda.driver as cuda
    cuda.init()
    HAS_TRT = True
except (ImportError, AttributeError):
    HAS_TRT = False


# Explicit color for compact remap fallback class: "other"
# Keep this distinct from wall gray for clear visualization.
OTHER_CLASS_COLOR = np.array([255, 165, 0], dtype=np.uint8)  # yellow-orange


class SUNRGBDColorPalette:
    """SUN RGB-D 37-class colour palette (matches ESANet CLASS_COLORS, void excluded)."""

    # Index 0 in CLASS_COLORS is void — skip it; classes 1-37 map to indices 0-36
    COLORS = np.array([
        [119, 119, 119],  # wall (gray)
        [244, 243, 131],  # floor (yellow)
        [137,  28, 157],  # cabinet (purple)
        [150, 255, 255],  # bed
        [ 54, 114, 113],  # chair (teal)
        [  0,   0, 176],  # sofa
        [255,  69,   0],  # table (orange)
        [ 87, 112, 255],  # door (blue)
        [  0, 163,  33],  # window (green)
        [255, 150, 255],  # bookshelf
        [255, 180,  10],  # picture
        [101,  70,  86],  # counter
        [ 38, 230,   0],  # blinds
        [255, 120,  70],  # desk (salmon)
        [117,  41, 121],  # shelves
        [150, 255,   0],  # curtain
        [132,   0, 255],  # dresser
        [ 24, 209, 255],  # pillow
        [191, 130,  35],  # mirror
        [219, 200, 109],  # floor_mat
        [154,  62,  86],  # clothes
        [255, 190, 190],  # ceiling (pink)
        [255,   0, 255],  # books
        [152, 163,  55],  # fridge
        [192,  79, 212],  # television (violet)
        [230, 230, 230],  # paper (gray)
        [ 53, 130,  64],  # towel
        [155, 249, 152],  # shower_curtain
        [ 87,  64,  34],  # box
        [214, 209, 175],  # whiteboard
        [170,   0,  59],  # person (red)
        [255,   0,   0],  # night_stand
        [193, 195, 234],  # toilet
        [ 70,  72, 115],  # sink
        [255, 255,   0],  # lamp
        [ 52,  57, 131],  # bathtub
        [ 12,  83,  45],  # bag
    ], dtype=np.uint8)

    @classmethod
    def colorize(cls, labels):
        """Convert label image to RGB using vectorized numpy indexing."""
        return cls.COLORS[np.clip(labels, 0, len(cls.COLORS) - 1)]


class BaseSegmentationBackend:
    """Base class for segmentation backends."""

    def __init__(self, config):
        self.num_classes = config.get('num_classes', 40)
        self.input_height = config.get('input_height', 480)
        self.input_width = config.get('input_width', 640)


class DFormerBackend(BaseSegmentationBackend):
    """DFormerv2-Large backend using SUN RGB-D checkpoint."""

    # Normalization constants matching DFormer dataloader
    RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    RGB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    DEP_MEAN = np.array([0.48,  0.48,  0.48 ], dtype=np.float32)
    DEP_STD  = np.array([0.28,  0.28,  0.28 ], dtype=np.float32)

    def __init__(self, config):
        super().__init__(config)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fp16   = config.get('fp16', False) and self.device.type == 'cuda'

        # DFormerv2 on SUN RGB-D uses square 480×480 input
        self.input_height = 480
        self.input_width  = 480

        model_path = config.get('model_path', '')
        self.model = self._load_model(model_path)
        rospy.loginfo(f"DFormer backend initialized on {self.device} (fp16={self.fp16})")

    def _load_model(self, model_path):
        """Build DFormer-Large and load SUN RGB-D checkpoint."""
        dformer_dir = os.path.join(os.path.dirname(__file__), '..', 'DFormer')
        dformer_dir = os.path.abspath(dformer_dir)
        stubs_dir   = os.path.join(dformer_dir, 'mmstubs')

        # Inject stub packages first so mmcv/mmengine/mmseg resolve without installation
        for d in (stubs_dir, dformer_dir):
            if d not in sys.path:
                sys.path.insert(0, d)

        try:
            from easydict import EasyDict as edict
            from models.builder import EncoderDecoder
        except ImportError as e:
            raise RuntimeError(f"Failed to import DFormer modules: {e}")

        cfg = edict()
        cfg.backbone          = 'DFormer-Large'
        cfg.decoder           = 'ham'
        cfg.decoder_embed_dim = 512
        cfg.drop_path_rate    = 0.2
        cfg.num_classes       = self.num_classes
        cfg.pretrained_model  = None
        cfg.bn_eps            = 1e-3
        cfg.bn_momentum       = 0.1
        cfg.aux_rate          = 0.0

        # criterion=None skips pretrained backbone loading in EncoderDecoder
        model = EncoderDecoder(cfg, criterion=None)

        if not model_path or not os.path.isfile(model_path):
            rospy.logwarn(f"[DFormer] checkpoint not found: {model_path}")
            return model.to(self.device).eval()

        rospy.loginfo(f"[DFormer] loading checkpoint: {model_path}")
        ckpt = torch.load(model_path, map_location='cpu')
        # Handle various checkpoint formats
        state = ckpt.get('state_dict', ckpt.get('model', ckpt))
        # Strip 'module.' prefix from DDP-saved checkpoints
        state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        rospy.loginfo("[DFormer] checkpoint loaded")

        model = model.to(self.device).eval()
        if self.fp16:
            model = model.half()
        return model

    def preprocess(self, rgb, depth=None):
        """Preprocess RGB (BGR uint8) and depth (float32 metres) for DFormer-Large."""
        # DFormer-Large + SUNRGBD uses BGR order (only DFormerv2 uses RGB)
        rgb_resized = cv2.resize(rgb, (self.input_width, self.input_height),
                                  interpolation=cv2.INTER_LINEAR)
        rgb_norm = (rgb_resized.astype(np.float32) / 255.0 - self.RGB_MEAN) / self.RGB_STD
        rgb_tensor = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        dep_tensor = None
        if depth is not None:
            # Map depth metres → uint8 [0-255] matching SUN RGB-D depth PNG convention
            # SUN RGB-D stores depth as uint16 PNG; cv2.IMREAD_GRAYSCALE scales to uint8 (/256)
            # Equivalent: clip to 0-10m, scale to 0-255
            dep_u8 = np.clip(depth / 10.0 * 255.0, 0, 255).astype(np.uint8)
            dep_u8 = cv2.resize(dep_u8, (self.input_width, self.input_height),
                                 interpolation=cv2.INTER_NEAREST)
            dep_3ch = cv2.merge([dep_u8, dep_u8, dep_u8])
            dep_norm = (dep_3ch.astype(np.float32) / 255.0 - self.DEP_MEAN) / self.DEP_STD
            dep_tensor = torch.from_numpy(dep_norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        if self.fp16:
            rgb_tensor = rgb_tensor.half()
            if dep_tensor is not None:
                dep_tensor = dep_tensor.half()

        return rgb_tensor, dep_tensor

    def infer(self, rgb, depth=None, need_uncertainty=False):
        """Run DFormer inference and return (labels, uncertainty)."""
        original_size = rgb.shape[:2]
        rgb_tensor, dep_tensor = self.preprocess(rgb, depth)

        with torch.no_grad():
            output = self.model(rgb_tensor, dep_tensor)  # (1, C, H, W) logits

        probs = F.softmax(output.float(), dim=1)
        labels = torch.argmax(probs, dim=1).squeeze().cpu().numpy().astype(np.uint8)

        h, w = original_size
        labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)

        if need_uncertainty:
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            uncertainty = (entropy / np.log(self.num_classes)).squeeze().cpu().numpy()
            uncertainty = cv2.resize(uncertainty, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            uncertainty = None

        return labels, uncertainty


class ESANetPyTorchBackend(BaseSegmentationBackend):
    """ESANet-R34-NBt1D PyTorch backend."""
    
    def __init__(self, config):
        super().__init__(config)
                  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fp16 = config.get('fp16', True) and self.device.type == 'cuda'
        
        # Normalization — pick stats matching the checkpoint dataset
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        dataset = config.get('dataset', 'nyuv2')
        if dataset == 'sunrgbd':
            # SUN RGB-D — includes RealSense sensor in training data
            self.depth_mean = 19025.14930492213
            self.depth_std  = 9880.916071806689
        else:
            # NYUv2 — Kinect v1 only
            self.depth_mean = 2841.94941272766
            self.depth_std  = 1417.2594281672277
        
        model_path = config.get('model_path', '')
        rospy.loginfo(f"Loading ESANet model from {model_path}")
        self.model = self._load_model(model_path)
        
        rospy.loginfo(f"ESANet PyTorch backend initialized on {self.device}")
        
    def _load_model(self, model_path):
        """Load ESANet-R34-NBt1D model from a checkpoint file."""
        import sys
        import argparse

        # Add ESANet directory to path so its src package is importable
        esanet_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ESANet'
        )
        if esanet_dir not in sys.path:
            sys.path.insert(0, esanet_dir)

        # Patch numpy for old pandas compatibility (np.bool removed in NumPy 1.20).
        # Unconditional assignment avoids triggering the FutureWarning that
        # hasattr(np, 'bool') raises when the deprecated attribute still exists.
        np.bool = bool

        from src.build_model import build_model

        if not os.path.exists(model_path):
            rospy.logerr(f"ESANet checkpoint not found: {model_path}")
            return None

        # Reconstruct the args namespace with ESANet-R34-NBt1D defaults.
        # pretrained_on_imagenet=False because we load a full checkpoint.
        args = argparse.Namespace(
            pretrained_on_imagenet=False,
            last_ckpt='',
            pretrained_dir=os.path.join(esanet_dir, 'trained_models', 'imagenet'),
            pretrained_scenenet='',
            finetune=None,
            he_init=False,
            height=self.input_height,
            width=self.input_width,
            modality='rgbd',
            encoder='resnet34',
            encoder_depth=None,           # will be set to 'resnet34' inside build_model
            encoder_block='NonBottleneck1D',
            activation='relu',
            encoder_decoder_fusion='add',
            context_module='ppm',
            channels_decoder=128,
            decoder_channels_mode='decreasing',
            nr_decoder_blocks=[3],
            fuse_depth_in_rgb_encoder='SE-add',
            upsampling='learned-3x3-zeropad',
        )

        model, _ = build_model(args, n_classes=self.num_classes)

        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        rospy.loginfo(f"Loaded ESANet checkpoint from {model_path}")

        model.eval()
        model.to(self.device)

        if self.fp16:
            model.half()

        return model
        
    def preprocess(self, rgb, depth=None):
        """Preprocess for ESANet."""
        # cv_bridge delivers BGR; ESANet expects RGB
        rgb_resized = cv2.resize(rgb, (self.input_width, self.input_height))
        rgb_resized = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2RGB)
        rgb_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).float() / 255.0
        rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)
        rgb_tensor = (rgb_tensor - self.rgb_mean) / self.rgb_std
        
        depth_tensor = None
        if depth is not None:
            depth_resized = cv2.resize(depth, (self.input_width, self.input_height),
                                        interpolation=cv2.INTER_NEAREST)
            # depth_callback stores depth in metres; ESANet normalization uses mm
            depth_mm = depth_resized * 1000.0
            depth_tensor = torch.from_numpy(depth_mm).float().unsqueeze(0).unsqueeze(0)
            depth_tensor = depth_tensor.to(self.device)
            depth_tensor = (depth_tensor - self.depth_mean) / self.depth_std
            
        if self.fp16:
            rgb_tensor = rgb_tensor.half()
            if depth_tensor is not None:
                depth_tensor = depth_tensor.half()
                
        return rgb_tensor, depth_tensor
        
    def infer(self, rgb, depth=None, need_uncertainty=False):
        """Run ESANet inference and return (labels, uncertainty)."""
        if self.model is None:
            h, w = rgb.shape[:2]
            return np.zeros((h, w), dtype=np.uint8), None

        rgb_tensor, depth_tensor = self.preprocess(rgb, depth)

        # ESANet RGBD requires a depth tensor; use zeros if depth is unavailable
        if depth_tensor is None:
            dtype = torch.float16 if self.fp16 else torch.float32
            depth_tensor = torch.zeros(
                (1, 1, self.input_height, self.input_width),
                dtype=dtype, device=self.device
            )

        with torch.no_grad():
            output = self.model(rgb_tensor, depth_tensor)

        # Cast to float32 before softmax to avoid fp16 overflow
        probs = F.softmax(output.float(), dim=1)
        labels = torch.argmax(probs, dim=1).squeeze().cpu().numpy().astype(np.uint8)

        h, w = rgb.shape[:2]
        labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)

        if need_uncertainty:
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            uncertainty = (entropy / np.log(self.num_classes)).squeeze().cpu().float().numpy()
            uncertainty = cv2.resize(uncertainty, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            uncertainty = None

        return labels, uncertainty


class ESANetTensorRTBackend(BaseSegmentationBackend):
    """ESANet TensorRT FP16 backend for Jetson Xavier AGX / Orin.

    Buffer layout follows ESANet's alloc_buf convention:
      bindings = [in_gpu_0, in_gpu_1, ..., out_gpu]  (ordered pointer list)
    Supports both TRT7 (binding-index API) and TRT8+ (tensor-name API).
    """

    def __init__(self, config):
        super().__init__(config)

        self.engine_path = config.get('engine_path', '')

        # RGB normalization (ImageNet)
        self.rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.rgb_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Depth normalization — must match training dataset (mm, valid pixels)
        dataset = config.get('dataset', 'nyuv2')
        if dataset == 'sunrgbd':
            self.depth_mean = 19025.14930492213
            self.depth_std  = 9880.916071806689
        else:  # nyuv2
            self.depth_mean = 2841.94941272766
            self.depth_std  = 1417.2594281672277

        # TRT inference state — following ESANet's alloc_buf pattern
        self.engine   = None
        self.context  = None
        self.stream   = None
        self.in_cpu   = []    # pagelocked host input buffers  [rgb, depth]
        self.in_gpu   = []    # device input buffers           [rgb, depth]
        self.out_cpu  = None  # pagelocked host output buffer
        self.out_gpu  = None  # device output buffer
        self.bindings = []    # ordered int pointers for execute_async_v2

        # Retain and push the primary CUDA context once — keep it active for lifetime
        self.cuda_ctx = cuda.Device(0).retain_primary_context()
        self.cuda_ctx.push()

        if os.path.exists(self.engine_path):
            self._load_engine()
            self._setup_buffers()
        else:
            rospy.logwarn(f"TensorRT engine not found: {self.engine_path}")

        rospy.loginfo("ESANet TensorRT backend initialized")

    def _load_engine(self):
        """Deserialize engine and allocate I/O buffers (TRT7 + TRT8+ compatible)."""
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream  = cuda.Stream()

        # TRT8+ uses tensor-name API; TRT7 uses binding-index API
        use_trt8_api = hasattr(self.engine, 'get_tensor_name')

        def _np_to_torch_dtype(np_dtype):
            return torch.float16 if np_dtype == np.float16 else torch.float32

        first_input_shape = None   # will be (N, C, H, W) from engine

        if use_trt8_api:
            n = self.engine.num_io_tensors
            for i in range(n):
                name       = self.engine.get_tensor_name(i)
                shape_full = self.engine.get_tensor_shape(name)   # e.g. (1,3,480,640)
                shape      = abs(trt.volume(shape_full))
                dtype      = trt.nptype(self.engine.get_tensor_dtype(name))
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    if first_input_shape is None:
                        first_input_shape = tuple(shape_full)
                    host_buf   = cuda.pagelocked_empty(shape, dtype)
                    device_buf = cuda.mem_alloc(host_buf.nbytes)
                    self.in_cpu.append(host_buf)
                    self.in_gpu.append(device_buf)
                    self.bindings.append(int(device_buf))
                else:
                    self.out_gpu_torch = torch.empty(
                        shape, dtype=_np_to_torch_dtype(dtype), device='cuda')
                    self.bindings.append(int(self.out_gpu_torch.data_ptr()))
        else:
            n_inputs = self.engine.num_bindings - 1
            for i in range(n_inputs):
                shape_full = self.engine.get_binding_shape(i)
                if first_input_shape is None:
                    first_input_shape = tuple(shape_full)
                shape      = abs(trt.volume(shape_full))
                dtype      = trt.nptype(self.engine.get_binding_dtype(i))
                host_buf   = cuda.pagelocked_empty(shape, dtype)
                device_buf = cuda.mem_alloc(host_buf.nbytes)
                self.in_cpu.append(host_buf)
                self.in_gpu.append(device_buf)
                self.bindings.append(int(device_buf))
            out_idx = self.engine.num_bindings - 1
            shape   = abs(trt.volume(self.engine.get_binding_shape(out_idx)))
            dtype   = trt.nptype(self.engine.get_binding_dtype(out_idx))
            self.out_gpu_torch = torch.empty(
                shape, dtype=_np_to_torch_dtype(dtype), device='cuda')
            self.bindings.append(int(self.out_gpu_torch.data_ptr()))

        # Derive H/W from the engine's actual first-input binding (N,C,H,W)
        if first_input_shape is not None and len(first_input_shape) == 4:
            self.input_height = int(first_input_shape[2])
            self.input_width  = int(first_input_shape[3])

        api_label = "TRT8+" if use_trt8_api else "TRT7"
        rospy.loginfo(f"TRT engine loaded [{api_label}]: "
                      f"{len(self.in_cpu)} input(s), "
                      f"input shape {self.input_height}x{self.input_width}")

    def _setup_buffers(self):
        """Pre-allocate reusable buffers to avoid per-frame numpy allocations."""
        H, W = self.input_height, self.input_width

        # Shaped views into pagelocked input buffers — write directly, skip np.copyto
        self._rgb_pl = self.in_cpu[0].reshape(3, H, W)
        if len(self.in_cpu) > 1:
            self._dep_pl = self.in_cpu[1].reshape(1, H, W)

        # Reusable float32 scratch for normalization (avoids 4-6 temp arrays per frame)
        self._rgb_f32 = np.empty((H, W, 3), dtype=np.float32)
        self._dep_f32 = np.empty((H, W),    dtype=np.float32)

        # Fused normalization: (x/255 - mean) / std  =  x * scale + offset
        self._rgb_scale  = (1.0 / (255.0 * self.rgb_std)).astype(np.float32)   # shape (3,)
        self._rgb_offset = (-self.rgb_mean / self.rgb_std).astype(np.float32)  # shape (3,)

        # Depth: (depth_m * 1000 - mean) / std  =  depth_m * k + b
        self._dep_k = np.float32(1000.0 / self.depth_std)
        self._dep_b = np.float32(-self.depth_mean / self.depth_std)

        # Pre-computed constant for entropy normalization
        self._log_C = np.float32(np.log(self.num_classes))

        rospy.loginfo(f"TRT buffers: rgb_f32 {self._rgb_f32.shape}, "
                      f"dep_f32 {self._dep_f32.shape}")

    def _preprocess_into_buffers(self, rgb, depth):
        """Normalize and write directly into pagelocked input buffers (zero extra copies)."""
        H, W = self.input_height, self.input_width

        # --- RGB ---
        if rgb.shape[0] != H or rgb.shape[1] != W:
            rgb = cv2.resize(rgb, (W, H))
        rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Fused normalize into pre-allocated float32 scratch
        self._rgb_f32[:] = rgb_rgb                 # uint8 → float32
        self._rgb_f32 *= self._rgb_scale           # in-place
        self._rgb_f32 += self._rgb_offset          # in-place

        # HWC→CHW + float32→float16 directly into pagelocked buffer
        self._rgb_pl[0] = self._rgb_f32[:, :, 0]
        self._rgb_pl[1] = self._rgb_f32[:, :, 1]
        self._rgb_pl[2] = self._rgb_f32[:, :, 2]

        # --- Depth ---
        if len(self.in_cpu) > 1:
            if depth is not None:
                if depth.shape[0] != H or depth.shape[1] != W:
                    depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
                np.multiply(depth, self._dep_k, out=self._dep_f32)
                self._dep_f32 += self._dep_b
                self._dep_pl[0] = self._dep_f32    # float32 → float16
            else:
                self._dep_pl[:] = 0

    def infer(self, rgb, depth=None, need_uncertainty=False):
        """Run TRT inference: preprocess → H2D → execute → postprocess."""
        if self.engine is None:
            h, w = rgb.shape[:2]
            return np.zeros((h, w), dtype=np.uint8), None

        # Preprocess writes directly into pagelocked buffers — no intermediate allocs
        self._preprocess_into_buffers(rgb, depth)

        # DMA to GPU
        for h_buf, d_buf in zip(self.in_cpu, self.in_gpu):
            cuda.memcpy_htod_async(d_buf, h_buf, self.stream)

        self.context.execute_async_v2(bindings=self.bindings,
                                      stream_handle=self.stream.handle)
        self.stream.synchronize()

        # Postprocess on GPU — out_gpu_torch already contains TRT output
        with torch.no_grad():
            logits = self.out_gpu_torch.reshape(
                self.num_classes, self.input_height, self.input_width)

            if need_uncertainty:
                # Need fp32 for log_softmax numerical precision
                log_probs = torch.log_softmax(logits.float(), dim=0)
                probs = log_probs.exp()                                     # compute once
                labels = probs.argmax(dim=0).byte().cpu().numpy()
                entropy = -(probs * log_probs).sum(dim=0).cpu().numpy()
                uncertainty = np.clip(
                    entropy / self._log_C, 0.0, 1.0
                ).astype(np.float32)
            else:
                # argmax works fine on fp16 — skip .float() conversion
                labels = logits.argmax(dim=0).byte().cpu().numpy()
                uncertainty = None

        # Resize output only if camera resolution differs from engine resolution
        h, w = rgb.shape[:2]
        if h != self.input_height or w != self.input_width:
            labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
            if uncertainty is not None:
                uncertainty = cv2.resize(uncertainty, (w, h), interpolation=cv2.INTER_LINEAR)

        return labels, uncertainty


class SemanticSegmentationNode:
    """ROS node for semantic segmentation."""
    
    def __init__(self):
        rospy.init_node('semantic_segmentation_node', anonymous=False)
        
        self.bridge = CvBridge()
        
        # Load parameters
        self.backend_name = rospy.get_param('~backend', 'dformer')
        # network_num_classes: the number of classes the TRT/PyTorch model outputs.
        # This must match the compiled engine — never the remapped compact count.
        esanet_dataset_tmp = rospy.get_param('~esanet_dataset', 'sunrgbd')
        default_net_classes = 37 if esanet_dataset_tmp == 'sunrgbd' else 40
        self.num_classes = rospy.get_param('~network_num_classes', default_net_classes)
        self.input_height = rospy.get_param('~input_height', 480)
        self.input_width = rospy.get_param('~input_width', 640)
        self.publish_color = rospy.get_param('~publish_color', False)
        self.publish_uncertainty = rospy.get_param('~publish_uncertainty', False)
        
        # Class remapping: relevant classes → compact indices, rest → "other".
        # Read from the hesfm_mapper_node namespace where hesfm_params.yaml is loaded.
        relevant_list = rospy.get_param('/hesfm_mapper_node/navigation/relevant_classes', [])
        if relevant_list:
            sorted_relevant = sorted(relevant_list)
            other_idx = len(sorted_relevant)  # last index = "other"
            self.remap_lut = np.full(256, other_idx, dtype=np.uint8)
            for new_idx, old_idx in enumerate(sorted_relevant):
                self.remap_lut[old_idx] = new_idx
            # Build compact color palette from original SUNRGBD palette + "other"
            compact_colors = [SUNRGBDColorPalette.COLORS[i] for i in sorted_relevant]
            compact_colors.append(OTHER_CLASS_COLOR)
            self.compact_palette = np.array(compact_colors, dtype=np.uint8)
            rospy.loginfo(f"Class remap: {len(sorted_relevant)} relevant + 1 other = {other_idx + 1} classes")
        else:
            self.remap_lut = None
            self.compact_palette = None

        # ESANet dataset & model paths
        self.esanet_dataset = rospy.get_param('~esanet_dataset', 'sunrgbd')
        self.dformer_model_path = rospy.get_param('~dformer_model_path', '')
        self.esanet_model_path = rospy.get_param('~esanet_model_path', '')
        self.esanet_trt_engine = rospy.get_param('~esanet_trt_engine', '')
        
        # Initialize backend
        self.backend = self._create_backend()
        
        # Cached images — written by callbacks, read by inference thread
        self.rgb_image = None
        self.depth_image = None
        self.rgb_stamp = None
        self._frame_lock = threading.Lock()
        self._new_frame  = threading.Event()

        # Pre-allocated depth float32 buffer (avoid per-frame astype allocation)
        self._depth_f32 = np.zeros((self.input_height, self.input_width), dtype=np.float32)

        # Publishers
        self.semantic_pub = rospy.Publisher('~semantic_image', Image, queue_size=1)
        self.semantic_color_pub = rospy.Publisher('~semantic_color', Image, queue_size=1)
        self.uncertainty_pub = rospy.Publisher('~uncertainty', Image, queue_size=1)

        # Subscribers
        self.rgb_sub   = rospy.Subscriber('color/image_raw',        Image, self.rgb_callback,   queue_size=1)
        self.depth_sub = rospy.Subscriber('depth/image_rect_raw',   Image, self.depth_callback, queue_size=1)

        # Stats
        # self.frame_count = 0
        # self.total_time = 0.0

        # Inference runs in a dedicated thread — callbacks never block
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._infer_thread.start()

        rospy.loginfo(f"Semantic Segmentation Node initialized with {self.backend_name} backend")
        
    def _create_backend(self):
        """Create segmentation backend."""
        config = {
            'num_classes': self.num_classes,
            'input_height': self.input_height,
            'input_width': self.input_width,
            'fp16': True,
        }
        
        if self.backend_name == 'dformer':
            config['model_path'] = self.dformer_model_path
            return DFormerBackend(config)
        elif self.backend_name == 'esanet_pytorch':
            config['model_path'] = self.esanet_model_path
            config['dataset'] = self.esanet_dataset
            return ESANetPyTorchBackend(config)
        elif self.backend_name == 'esanet_trt':
            config['engine_path'] = self.esanet_trt_engine
            config['dataset'] = self.esanet_dataset
            return ESANetTensorRTBackend(config)
        else:
            raise ValueError(f"Unknown backend: {self.backend_name}")
            
    def rgb_callback(self, msg):
        """Store latest RGB frame and signal the inference thread."""
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self._frame_lock:
                self.rgb_image = img
                self.rgb_stamp = msg.header.stamp
            self._new_frame.set()
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")

    def depth_callback(self, msg):
        """Store latest depth frame (float32, metres)."""
        try:
            if msg.encoding == '16UC1':
                raw = self.bridge.imgmsg_to_cv2(msg, '16UC1')
                # Write into pre-allocated buffer — avoid extra allocation
                np.multiply(raw, np.float32(0.001), out=self._depth_f32
                            if raw.shape == self._depth_f32.shape
                            else np.empty(raw.shape, dtype=np.float32))
                with self._frame_lock:
                    self.depth_image = self._depth_f32
            elif msg.encoding == '32FC1':
                with self._frame_lock:
                    self.depth_image = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            else:
                with self._frame_lock:
                    self.depth_image = self.bridge.imgmsg_to_cv2(msg)
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")

    def _infer_loop(self):
        """Inference thread: wait for a new RGB frame, then run the full pipeline."""
        while not rospy.is_shutdown():
            if not self._new_frame.wait(timeout=0.5):
                continue
            self._new_frame.clear()

            with self._frame_lock:
                rgb   = self.rgb_image
                depth = self.depth_image
                stamp = self.rgb_stamp

            if rgb is None:
                continue

            self.process(rgb, depth, stamp)

    def process(self, rgb, depth, stamp):
        """Process one frame (called from inference thread)."""
        need_unc = self.publish_uncertainty and self.uncertainty_pub.get_num_connections() > 0

        start_time = time.time()
        labels, uncertainty = self.backend.infer(rgb, depth, need_uncertainty=need_unc)
        elapsed = time.time() - start_time

        # Remap to compact class set if configured
        if self.remap_lut is not None:
            labels = self.remap_lut[labels]

        # self.total_time += elapsed
        # self.frame_count += 1
        # if self.frame_count % 90 == 0:
        #     rospy.loginfo(f"Segmentation FPS: {self.frame_count / self.total_time:.1f}")

        # Create header
        header = Header()
        header.stamp = stamp
        header.frame_id = "camera_color_optical_frame"

        # Publish semantic labels (remapped)
        semantic_msg = self.bridge.cv2_to_imgmsg(labels, encoding='mono8')
        semantic_msg.header = header
        self.semantic_pub.publish(semantic_msg)

        # Publish colored visualization only when needed.
        # Colorization allocates a full RGB image, so skip it if nobody subscribes.
        if self.publish_color and self.semantic_color_pub.get_num_connections() > 0:
            if self.compact_palette is not None:
                colored = self.compact_palette[np.clip(labels, 0, len(self.compact_palette) - 1)]
            else:
                colored = SUNRGBDColorPalette.colorize(labels)
            colored_msg = self.bridge.cv2_to_imgmsg(colored, encoding='rgb8')
            colored_msg.header = header
            self.semantic_color_pub.publish(colored_msg)

        # Publish uncertainty only when needed.
        if need_unc and uncertainty is not None:
            uncertainty_uint8 = (np.clip(uncertainty, 0.0, 1.0) * 255).astype(np.uint8)
            uncertainty_msg = self.bridge.cv2_to_imgmsg(uncertainty_uint8, encoding='mono8')
            uncertainty_msg.header = header
            self.uncertainty_pub.publish(uncertainty_msg)
                
    def run(self):
        """Run the node."""
        rospy.spin()


def main():
    try:
        node = SemanticSegmentationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
