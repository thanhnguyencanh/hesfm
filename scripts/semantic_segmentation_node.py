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
from threading import Lock

import rospy
import numpy as np
import cv2

from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server

try:
    from hesfm.cfg import SemanticSegmentationConfig
    HAS_DYNRECONF = True
except ImportError:
    HAS_DYNRECONF = False

# Try importing deep learning frameworks
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
    cuda.init()  # explicit init — do NOT use pycuda.autoinit (creates non-primary context)
    HAS_TRT = True
except (ImportError, AttributeError):
    HAS_TRT = False


class NYUv2ColorPalette:
    """NYUv2 40-class color palette for visualization."""
    
    COLORS = np.array([
        [128, 128, 128],  # wall
        [139, 119, 101],  # floor
        [244, 164, 96],   # cabinet
        [255, 182, 193],  # bed
        [255, 215, 0],    # chair
        [220, 20, 60],    # sofa
        [255, 140, 0],    # table
        [139, 69, 19],    # door
        [135, 206, 235],  # window
        [160, 82, 45],    # bookshelf
        [255, 105, 180],  # picture
        [0, 128, 128],    # counter
        [210, 180, 140],  # blinds
        [70, 130, 180],   # desk
        [188, 143, 143],  # shelves
        [147, 112, 219],  # curtain
        [222, 184, 135],  # dresser
        [255, 228, 225],  # pillow
        [192, 192, 192],  # mirror
        [139, 119, 101],  # floor_mat
        [128, 0, 128],    # clothes
        [245, 245, 245],  # ceiling
        [139, 90, 43],    # books
        [173, 216, 230],  # fridge
        [0, 0, 139],      # television
        [255, 255, 224],  # paper
        [240, 255, 255],  # towel
        [176, 224, 230],  # shower_curtain
        [210, 105, 30],   # box
        [255, 255, 255],  # whiteboard
        [255, 0, 0],      # person
        [85, 107, 47],    # night_stand
        [255, 255, 240],  # toilet
        [176, 196, 222],  # sink
        [255, 250, 205],  # lamp
        [224, 255, 255],  # bathtub
        [75, 0, 130],     # bag
        [169, 169, 169],  # otherstructure
        [105, 105, 105],  # otherfurniture
        [128, 128, 0],    # otherprop
    ], dtype=np.uint8)
    
    @classmethod
    def colorize(cls, labels):
        """Convert label image to RGB using vectorized numpy indexing."""
        return cls.COLORS[np.clip(labels, 0, len(cls.COLORS) - 1)]


class SUNRGBDColorPalette:
    """SUN RGB-D 37-class colour palette (matches ESANet CLASS_COLORS, void excluded)."""

    # Index 0 in CLASS_COLORS is void — skip it; classes 1-37 map to indices 0-36
    COLORS = np.array([
        [119, 119, 119],  # wall
        [244, 243, 131],  # floor
        [137,  28, 157],  # cabinet
        [150, 255, 255],  # bed
        [ 54, 114, 113],  # chair
        [  0,   0, 176],  # sofa
        [255,  69,   0],  # table
        [ 87, 112, 255],  # door
        [  0, 163,  33],  # window
        [255, 150, 255],  # bookshelf
        [255, 180,  10],  # picture
        [101,  70,  86],  # counter
        [ 38, 230,   0],  # blinds
        [255, 120,  70],  # desk
        [117,  41, 121],  # shelves
        [150, 255,   0],  # curtain
        [132,   0, 255],  # dresser
        [ 24, 209, 255],  # pillow
        [191, 130,  35],  # mirror
        [219, 200, 109],  # floor_mat
        [154,  62,  86],  # clothes
        [255, 190, 190],  # ceiling
        [255,   0, 255],  # books
        [152, 163,  55],  # fridge
        [192,  79, 212],  # television
        [230, 230, 230],  # paper
        [ 53, 130,  64],  # towel
        [155, 249, 152],  # shower_curtain
        [ 87,  64,  34],  # box
        [214, 209, 175],  # whiteboard
        [170,   0,  59],  # person
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
        self.config = config
        self.num_classes = config.get('num_classes', 40)
        self.input_height = config.get('input_height', 480)
        self.input_width = config.get('input_width', 640)
        self.publish_probabilities = config.get('publish_probabilities', False)
        
    def preprocess(self, *args, **kwargs):
        raise NotImplementedError

    def infer(self, *args, **kwargs):
        raise NotImplementedError

    def postprocess(self, *args, **kwargs):
        raise NotImplementedError


class DFormerBackend(BaseSegmentationBackend):
    """DFormerv2-Large backend using SUN RGB-D checkpoint."""

    # Normalization constants matching DFormer dataloader
    RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    RGB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    DEP_MEAN = np.array([0.48,  0.48,  0.48 ], dtype=np.float32)
    DEP_STD  = np.array([0.28,  0.28,  0.28 ], dtype=np.float32)

    def __init__(self, config):
        super().__init__(config)

        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for DFormer backend")

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

    def infer(self, rgb, depth=None):
        """Run DFormer inference and return (labels, probs, uncertainty)."""
        original_size = rgb.shape[:2]
        rgb_tensor, dep_tensor = self.preprocess(rgb, depth)

        with torch.no_grad():
            output = self.model(rgb_tensor, dep_tensor)  # (1, C, H, W) logits

        return self.postprocess(output, original_size)

    def postprocess(self, output, original_size):
        """Convert DFormer logits to labels, probs, uncertainty."""
        probs = F.softmax(output.float(), dim=1)
        labels = torch.argmax(probs, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        uncertainty = (entropy / np.log(self.num_classes)).squeeze().cpu().numpy()

        labels_np = labels.squeeze().cpu().numpy().astype(np.uint8)
        h, w = original_size
        labels_np  = cv2.resize(labels_np,  (w, h), interpolation=cv2.INTER_NEAREST)
        uncertainty = cv2.resize(uncertainty, (w, h), interpolation=cv2.INTER_LINEAR)

        probs_np = None
        if self.publish_probabilities:
            probs_np = probs.squeeze().permute(1, 2, 0).cpu().numpy()
            probs_np = cv2.resize(probs_np, (w, h), interpolation=cv2.INTER_LINEAR)

        return labels_np, probs_np, uncertainty


class ESANetPyTorchBackend(BaseSegmentationBackend):
    """ESANet-R34-NBt1D PyTorch backend."""
    
    def __init__(self, config):
        super().__init__(config)
        
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for ESANet backend")
            
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
        
    def infer(self, rgb, depth=None):
        """Run ESANet inference."""
        if self.model is None:
            h, w = rgb.shape[:2]
            labels = np.zeros((h, w), dtype=np.uint8)
            probs = np.ones((h, w, self.num_classes), dtype=np.float32) / self.num_classes
            uncertainty = np.ones((h, w), dtype=np.float32) * 0.5
            return labels, probs, uncertainty

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
        return self.postprocess(output, rgb.shape[:2])
        
    def postprocess(self, output, original_size):
        """Postprocess ESANet output."""
        # Cast to float32 before softmax to avoid fp16 overflow
        probs = F.softmax(output.float(), dim=1)
        labels = torch.argmax(probs, dim=1)

        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        max_entropy = np.log(self.num_classes)
        uncertainty = entropy / max_entropy
        
        labels = labels.squeeze().cpu().numpy().astype(np.uint8)
        uncertainty = uncertainty.squeeze().cpu().float().numpy()
        if self.publish_probabilities:
            probs = probs.squeeze().permute(1, 2, 0).cpu().float().numpy()
        else:
            probs = None

        h, w = original_size
        labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
        uncertainty = cv2.resize(uncertainty, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return labels, probs, uncertainty


class ESANetTensorRTBackend(BaseSegmentationBackend):
    """ESANet TensorRT FP16 backend for Jetson Xavier AGX / Orin.

    Buffer layout follows ESANet's alloc_buf convention:
      bindings = [in_gpu_0, in_gpu_1, ..., out_gpu]  (ordered pointer list)
    Supports both TRT7 (binding-index API) and TRT8+ (tensor-name API).
    """

    def __init__(self, config):
        super().__init__(config)

        if not HAS_TRT:
            raise RuntimeError("TensorRT required for ESANet TRT backend")

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

        if use_trt8_api:
            n = self.engine.num_io_tensors
            for i in range(n):
                name  = self.engine.get_tensor_name(i)
                shape = abs(trt.volume(self.engine.get_tensor_shape(name)))
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    host_buf   = cuda.pagelocked_empty(shape, dtype)
                    device_buf = cuda.mem_alloc(host_buf.nbytes)
                    self.in_cpu.append(host_buf)
                    self.in_gpu.append(device_buf)
                    self.bindings.append(int(device_buf))
                else:
                    # Output: allocate as PyTorch CUDA tensor — TRT writes directly,
                    # no GPU→CPU→GPU round-trip needed in postprocess
                    self.out_gpu_torch = torch.empty(
                        shape, dtype=_np_to_torch_dtype(dtype), device='cuda')
                    self.bindings.append(int(self.out_gpu_torch.data_ptr()))
        else:
            n_inputs = self.engine.num_bindings - 1
            for i in range(n_inputs):
                shape = abs(trt.volume(self.engine.get_binding_shape(i)))
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                host_buf   = cuda.pagelocked_empty(shape, dtype)
                device_buf = cuda.mem_alloc(host_buf.nbytes)
                self.in_cpu.append(host_buf)
                self.in_gpu.append(device_buf)
                self.bindings.append(int(device_buf))
            out_idx = self.engine.num_bindings - 1
            shape = abs(trt.volume(self.engine.get_binding_shape(out_idx)))
            dtype = trt.nptype(self.engine.get_binding_dtype(out_idx))
            self.out_gpu_torch = torch.empty(
                shape, dtype=_np_to_torch_dtype(dtype), device='cuda')
            self.bindings.append(int(self.out_gpu_torch.data_ptr()))

        api_label = "TRT8+" if use_trt8_api else "TRT7"
        rospy.loginfo(f"TRT engine loaded [{api_label}]: "
                      f"{len(self.in_cpu)} input(s)")

    def preprocess(self, rgb, depth=None):
        """Preprocess BGR image and depth (metres) -> FP16 CHW arrays for TRT."""
        # BGR->RGB — cv_bridge delivers BGR, ESANet was trained on RGB
        rgb_resized = cv2.resize(rgb, (self.input_width, self.input_height))
        rgb_rgb  = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2RGB)
        rgb_norm = (rgb_rgb.astype(np.float32) / 255.0 - self.rgb_mean) / self.rgb_std
        rgb_chw  = np.ascontiguousarray(rgb_norm.transpose(2, 0, 1), dtype=np.float16)

        # Depth: depth_callback stores metres -> convert to mm.
        # Fall back to zeros when depth is unavailable.
        if depth is not None:
            depth_resized = cv2.resize(depth, (self.input_width, self.input_height),
                                       interpolation=cv2.INTER_NEAREST)
            depth_mm   = depth_resized * 1000.0
            depth_norm = (depth_mm.astype(np.float32) - self.depth_mean) / self.depth_std
            depth_chw  = np.ascontiguousarray(depth_norm[np.newaxis], dtype=np.float16)
        else:
            depth_chw = np.zeros((1, self.input_height, self.input_width), dtype=np.float16)

        return rgb_chw, depth_chw

    def infer(self, rgb, depth=None):
        """Run TRT inference: async H2D -> execute -> async D2H -> sync."""
        if self.engine is None:
            h, w = rgb.shape[:2]
            labels      = np.zeros((h, w), dtype=np.uint8)
            probs       = np.ones((h, w, self.num_classes), dtype=np.float32) / self.num_classes
            uncertainty = np.ones((h, w), dtype=np.float32) * 0.5
            return labels, probs, uncertainty

        rgb_chw, depth_chw = self.preprocess(rgb, depth)

        # Copy inputs into page-locked host buffers then DMA to GPU
        np.copyto(self.in_cpu[0], rgb_chw.ravel())
        if len(self.in_cpu) > 1:
            np.copyto(self.in_cpu[1], depth_chw.ravel())

        for h_buf, d_buf in zip(self.in_cpu, self.in_gpu):
            cuda.memcpy_htod_async(d_buf, h_buf, self.stream)

        self.context.execute_async_v2(bindings=self.bindings,
                                      stream_handle=self.stream.handle)

        # No GPU→CPU copy — out_gpu_torch is a PyTorch CUDA tensor TRT writes into
        self.stream.synchronize()

        return self.postprocess(self.out_gpu_torch, rgb.shape[:2])

    def postprocess(self, output, original_size):
        """Reshape flat TRT output -> labels, probs, uncertainty.

        output is a PyTorch CUDA tensor (TRT writes directly into it).
        No GPU→CPU→GPU round-trip needed.
        """
        with torch.no_grad():
            logits_t  = output.reshape(
                self.num_classes, self.input_height, self.input_width
            ).float()                                         # fp16 → fp32 on GPU
            log_probs = torch.log_softmax(logits_t, dim=0)   # (C, H, W)
            probs_t   = torch.exp(log_probs)                  # (C, H, W)
            labels    = probs_t.argmax(dim=0).byte().cpu().numpy()        # (H, W)
            entropy   = -(probs_t * log_probs).sum(dim=0).cpu().numpy()   # (H, W)
            
            if self.publish_probabilities:
                probs = probs_t.cpu().numpy().transpose(1, 2, 0)          # (H, W, C)
            else:
                probs = None

        uncertainty = np.clip(
            entropy / np.log(self.num_classes), 0.0, 1.0
        ).astype(np.float32)

        h, w = original_size
        labels      = cv2.resize(labels,      (w, h), interpolation=cv2.INTER_NEAREST)
        uncertainty = cv2.resize(uncertainty, (w, h), interpolation=cv2.INTER_LINEAR)

        return labels, probs, uncertainty


class SemanticSegmentationNode:
    """ROS node for semantic segmentation."""
    
    def __init__(self):
        rospy.init_node('semantic_segmentation_node', anonymous=False)
        
        self.bridge = CvBridge()
        self.lock = Lock()
        
        # Load parameters
        self.backend_name = rospy.get_param('~backend', 'dformer')
        self.num_classes = rospy.get_param('~num_classes', 40)
        self.input_height = rospy.get_param('~input_height', 480)
        self.input_width = rospy.get_param('~input_width', 640)
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.3)
        self.publish_color = rospy.get_param('~publish_color', True)
        self.publish_probabilities = rospy.get_param('~publish_probabilities', False)
        self.publish_uncertainty = rospy.get_param('~publish_uncertainty', True)
        
        # Dataset used to train the ESANet checkpoint ('nyuv2' or 'sunrgbd')
        self.esanet_dataset = rospy.get_param('~esanet_dataset', 'nyuv2')

        # Model paths
        self.dformer_model_path = rospy.get_param('~dformer_model_path', '')
        self.esanet_model_path = rospy.get_param('~esanet_model_path', '')
        self.esanet_trt_engine = rospy.get_param('~esanet_trt_engine', '')
        
        # Initialize backend
        self.backend = self._create_backend()
        
        # Cached images
        self.rgb_image = None
        self.depth_image = None
        self.rgb_stamp = None
        
        # Publishers
        self.semantic_pub = rospy.Publisher('~semantic_image', Image, queue_size=1)
        self.semantic_color_pub = rospy.Publisher('~semantic_color', Image, queue_size=1)
        self.uncertainty_pub = rospy.Publisher('~uncertainty', Image, queue_size=1)
        
        # Subscribers
        self.rgb_sub = rospy.Subscriber('color/image_raw', Image, self.rgb_callback, queue_size=1)

        self.depth_sub = rospy.Subscriber('depth/image_rect_raw', Image, self.depth_callback, queue_size=1)
        
        # Dynamic reconfigure
        if HAS_DYNRECONF:
            self.dyn_srv = Server(SemanticSegmentationConfig, self.dyn_callback)
            
        # Stats
        self.frame_count = 0
        self.total_time = 0.0
        
        rospy.loginfo(f"Semantic Segmentation Node initialized with {self.backend_name} backend")
        
    def _create_backend(self):
        """Create segmentation backend."""
        config = {
            'num_classes': self.num_classes,
            'input_height': self.input_height,
            'input_width': self.input_width,
            'fp16': True,
            'publish_probabilities': self.publish_probabilities,
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
        elif self.backend_name == 'auto':
            # Auto-select based on hardware
            if HAS_TRT and os.path.exists(self.esanet_trt_engine):
                config['engine_path'] = self.esanet_trt_engine
                return ESANetTensorRTBackend(config)
            elif HAS_TORCH:
                config['model_path'] = self.dformer_model_path
                return DFormerBackend(config)
            else:
                raise RuntimeError("No suitable backend available")
        else:
            raise ValueError(f"Unknown backend: {self.backend_name}")
            
    def dyn_callback(self, config, _level):
        """Dynamic reconfigure callback."""
        with self.lock:
            self.confidence_threshold = config.confidence_threshold
            self.publish_color = config.publish_color
            self.publish_uncertainty = config.publish_uncertainty
        return config
        
    def rgb_callback(self, msg):
        """RGB image callback."""
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.rgb_stamp = msg.header.stamp
            self.process()
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
            
    def depth_callback(self, msg):
        """Depth image callback."""
        try:
            if msg.encoding == '16UC1':
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, '16UC1').astype(np.float32) * 0.001
            elif msg.encoding == '32FC1':
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            else:
                self.depth_image = self.bridge.imgmsg_to_cv2(msg)
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
            
    def process(self):
        """Process current frame."""
        if self.rgb_image is None:
            return

        # Snapshot shared images under lock; run GPU inference outside it
        with self.lock:
            rgb   = self.rgb_image
            depth = self.depth_image
            stamp = self.rgb_stamp

        start_time = time.time()
        labels, _, uncertainty = self.backend.infer(rgb, depth)
        elapsed = time.time() - start_time

        self.total_time += elapsed
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            rospy.loginfo(f"Segmentation FPS: {self.frame_count / self.total_time:.1f}")

        # Create header
        header = Header()
        header.stamp = stamp
        header.frame_id = "camera_color_optical_frame"

        # Publish semantic labels
        semantic_msg = self.bridge.cv2_to_imgmsg(labels, encoding='mono8')
        semantic_msg.header = header
        self.semantic_pub.publish(semantic_msg)

        # Publish colored visualization
        if self.publish_color:
            palette = (SUNRGBDColorPalette if self.esanet_dataset == 'sunrgbd'
                       else NYUv2ColorPalette)
            colored_msg = self.bridge.cv2_to_imgmsg(palette.colorize(labels), encoding='rgb8')
            colored_msg.header = header
            self.semantic_color_pub.publish(colored_msg)

        # Publish uncertainty
        if self.publish_uncertainty:
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
