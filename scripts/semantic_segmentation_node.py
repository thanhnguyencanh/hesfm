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
# if sys.version_info < (3, 12):
#     os.execv('/usr/local/bin/python3.12', ['/usr/local/bin/python3.12', __file__] + sys.argv[1:])

import rospy
import numpy as np
import cv2
import os
import time
from threading import Lock

from sensor_msgs.msg import Image, CameraInfo
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
    rospy.logwarn("PyTorch is available: %s", torch.__version__)
except ImportError:
    HAS_TORCH = False
    rospy.logwarn("PyTorch not available")

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_TRT = True
except ImportError:
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
        """Convert label image to RGB."""
        h, w = labels.shape[:2]
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(len(cls.COLORS)):
            colored[labels == c] = cls.COLORS[c]
        return colored


class BaseSegmentationBackend:
    """Base class for segmentation backends."""
    
    def __init__(self, config):
        self.config = config
        self.num_classes = config.get('num_classes', 40)
        self.input_height = config.get('input_height', 480)
        self.input_width = config.get('input_width', 640)
        
    def preprocess(self, rgb, depth=None):
        """Preprocess images for inference."""
        raise NotImplementedError
        
    def infer(self, rgb, depth=None):
        """Run inference, return (labels, probabilities, uncertainty)."""
        raise NotImplementedError
        
    def postprocess(self, output):
        """Postprocess network output."""
        raise NotImplementedError


class DFormerBackend(BaseSegmentationBackend):
    """DFormerv2-Large backend for RTX 4090/4080."""
    
    def __init__(self, config):
        super().__init__(config)
        
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for DFormer backend")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fp16 = config.get('fp16', True) and self.device.type == 'cuda'
        
        # Normalization
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        self.depth_mean = 2.8
        self.depth_std = 1.4
        
        # Load model
        model_path = config.get('model_path', '')
        self.model = self._load_model(model_path)
        
        rospy.loginfo(f"DFormer backend initialized on {self.device}")
        
    def _load_model(self, model_path):
        """Load DFormer model."""
        # Placeholder - actual implementation depends on DFormer code
        rospy.logwarn("DFormer model loading - implement with actual DFormer code")
        return None
        
    def preprocess(self, rgb, depth=None):
        """Preprocess for DFormer."""
        # Resize
        rgb_resized = cv2.resize(rgb, (self.input_width, self.input_height))
        
        # To tensor and normalize
        rgb_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).float() / 255.0
        rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)
        rgb_tensor = (rgb_tensor - self.rgb_mean) / self.rgb_std
        
        depth_tensor = None
        if depth is not None:
            depth_resized = cv2.resize(depth, (self.input_width, self.input_height),
                                        interpolation=cv2.INTER_NEAREST)
            depth_tensor = torch.from_numpy(depth_resized).float().unsqueeze(0).unsqueeze(0)
            depth_tensor = depth_tensor.to(self.device)
            depth_tensor = (depth_tensor - self.depth_mean) / self.depth_std
            
        if self.fp16:
            rgb_tensor = rgb_tensor.half()
            if depth_tensor is not None:
                depth_tensor = depth_tensor.half()
                
        return rgb_tensor, depth_tensor
        
    def infer(self, rgb, depth=None):
        """Run DFormer inference."""
        if self.model is None:
            # Return dummy output for testing
            h, w = rgb.shape[:2]
            labels = np.zeros((h, w), dtype=np.uint8)
            probs = np.ones((h, w, self.num_classes), dtype=np.float32) / self.num_classes
            uncertainty = np.ones((h, w), dtype=np.float32) * 0.5
            return labels, probs, uncertainty
            
        rgb_tensor, depth_tensor = self.preprocess(rgb, depth)
        
        with torch.no_grad():
            if self.fp16:
                with torch.cuda.amp.autocast():
                    output = self.model(rgb_tensor, depth_tensor)
            else:
                output = self.model(rgb_tensor, depth_tensor)
                
        return self.postprocess(output, rgb.shape[:2])
        
    def postprocess(self, output, original_size):
        """Postprocess DFormer output."""
        # Get probabilities
        probs = F.softmax(output, dim=1)
        
        # Get labels
        labels = torch.argmax(probs, dim=1)
        
        # Compute entropy-based uncertainty
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        max_entropy = np.log(self.num_classes)
        uncertainty = entropy / max_entropy
        
        # To numpy and resize
        labels = labels.squeeze().cpu().numpy().astype(np.uint8)
        probs = probs.squeeze().permute(1, 2, 0).cpu().numpy()
        uncertainty = uncertainty.squeeze().cpu().numpy()
        
        # Resize to original
        h, w = original_size
        labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
        uncertainty = cv2.resize(uncertainty, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return labels, probs, uncertainty


class ESANetPyTorchBackend(BaseSegmentationBackend):
    """ESANet-R34-NBt1D PyTorch backend."""
    
    def __init__(self, config):
        super().__init__(config)
        
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for ESANet backend")
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fp16 = config.get('fp16', True) and self.device.type == 'cuda'
        
        # Normalization
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        self.depth_mean = 2842.9
        self.depth_std = 1449.3
        
        model_path = config.get('model_path', '')
        self.model = self._load_model(model_path)
        
        rospy.loginfo(f"ESANet PyTorch backend initialized on {self.device}")
        
    def _load_model(self, model_path):
        """Load ESANet model."""
        rospy.logwarn("ESANet model loading - implement with actual ESANet code")
        return None
        
    def preprocess(self, rgb, depth=None):
        """Preprocess for ESANet."""
        rgb_resized = cv2.resize(rgb, (self.input_width, self.input_height))
        rgb_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).float() / 255.0
        rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)
        rgb_tensor = (rgb_tensor - self.rgb_mean) / self.rgb_std
        
        depth_tensor = None
        if depth is not None:
            depth_resized = cv2.resize(depth, (self.input_width, self.input_height),
                                        interpolation=cv2.INTER_NEAREST)
            # Convert to mm if in meters
            if depth_resized.max() < 100:
                depth_resized = depth_resized * 1000
            depth_tensor = torch.from_numpy(depth_resized).float().unsqueeze(0).unsqueeze(0)
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
        
        with torch.no_grad():
            output = self.model(rgb_tensor, depth_tensor)
            
        return self.postprocess(output, rgb.shape[:2])
        
    def postprocess(self, output, original_size):
        """Postprocess ESANet output."""
        probs = F.softmax(output, dim=1)
        labels = torch.argmax(probs, dim=1)
        
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        max_entropy = np.log(self.num_classes)
        uncertainty = entropy / max_entropy
        
        labels = labels.squeeze().cpu().numpy().astype(np.uint8)
        probs = probs.squeeze().permute(1, 2, 0).cpu().numpy()
        uncertainty = uncertainty.squeeze().cpu().numpy()
        
        h, w = original_size
        labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
        uncertainty = cv2.resize(uncertainty, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return labels, probs, uncertainty


class ESANetTensorRTBackend(BaseSegmentationBackend):
    """ESANet TensorRT FP16 backend for Jetson Orin."""
    
    def __init__(self, config):
        super().__init__(config)
        
        if not HAS_TRT:
            raise RuntimeError("TensorRT required for ESANet TRT backend")
            
        self.engine_path = config.get('engine_path', '')
        
        # Normalization
        self.rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.depth_mean = 2842.9
        self.depth_std = 1449.3
        
        self.engine = None
        self.context = None
        self.stream = None
        self.bindings = []
        self.inputs = []
        self.outputs = []
        
        if os.path.exists(self.engine_path):
            self._load_engine()
        else:
            rospy.logwarn(f"TensorRT engine not found: {self.engine_path}")
            
        rospy.loginfo("ESANet TensorRT backend initialized")
        
    def _load_engine(self):
        """Load TensorRT engine."""
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        with open(self.engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # Allocate buffers
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
                
    def preprocess(self, rgb, depth=None):
        """Preprocess for TensorRT."""
        rgb_resized = cv2.resize(rgb, (self.input_width, self.input_height))
        rgb_norm = (rgb_resized.astype(np.float32) / 255.0 - self.rgb_mean) / self.rgb_std
        rgb_chw = rgb_norm.transpose(2, 0, 1).astype(np.float16)
        
        depth_chw = None
        if depth is not None:
            depth_resized = cv2.resize(depth, (self.input_width, self.input_height),
                                        interpolation=cv2.INTER_NEAREST)
            if depth_resized.max() < 100:
                depth_resized = depth_resized * 1000
            depth_norm = (depth_resized.astype(np.float32) - self.depth_mean) / self.depth_std
            depth_chw = depth_norm[np.newaxis, ...].astype(np.float16)
            
        return rgb_chw, depth_chw
        
    def infer(self, rgb, depth=None):
        """Run TensorRT inference."""
        if self.engine is None:
            h, w = rgb.shape[:2]
            labels = np.zeros((h, w), dtype=np.uint8)
            probs = np.ones((h, w, self.num_classes), dtype=np.float32) / self.num_classes
            uncertainty = np.ones((h, w), dtype=np.float32) * 0.5
            return labels, probs, uncertainty
            
        rgb_chw, depth_chw = self.preprocess(rgb, depth)
        
        # Copy to input
        np.copyto(self.inputs[0]['host'], rgb_chw.ravel())
        if len(self.inputs) > 1 and depth_chw is not None:
            np.copyto(self.inputs[1]['host'], depth_chw.ravel())
            
        # Transfer to GPU
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
            
        # Execute
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Transfer back
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            
        self.stream.synchronize()
        
        return self.postprocess(self.outputs[0]['host'], rgb.shape[:2])
        
    def postprocess(self, output, original_size):
        """Postprocess TensorRT output."""
        # Reshape output
        logits = output.reshape(self.num_classes, self.input_height, self.input_width)
        
        # Softmax
        exp_logits = np.exp(logits - logits.max(axis=0, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=0, keepdims=True)
        
        # Labels
        labels = np.argmax(probs, axis=0).astype(np.uint8)
        
        # Uncertainty
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=0)
        max_entropy = np.log(self.num_classes)
        uncertainty = entropy / max_entropy
        
        # Resize
        h, w = original_size
        labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
        uncertainty = cv2.resize(uncertainty, (w, h), interpolation=cv2.INTER_LINEAR)
        probs = probs.transpose(1, 2, 0)
        
        return labels, probs, uncertainty.astype(np.float32)


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
        }
        
        if self.backend_name == 'dformer':
            config['model_path'] = self.dformer_model_path
            return DFormerBackend(config)
        elif self.backend_name == 'esanet_pytorch':
            config['model_path'] = self.esanet_model_path
            return ESANetPyTorchBackend(config)
        elif self.backend_name == 'esanet_trt':
            config['engine_path'] = self.esanet_trt_engine
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
            
    def dyn_callback(self, config, level):
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
            
        with self.lock:
            start_time = time.time()
            
            # Run inference
            labels, probs, uncertainty = self.backend.infer(self.rgb_image, self.depth_image)
            
            elapsed = time.time() - start_time
            self.total_time += elapsed
            self.frame_count += 1
            
            # Log FPS periodically
            if self.frame_count % 30 == 0:
                avg_fps = self.frame_count / self.total_time
                rospy.loginfo(f"Segmentation FPS: {avg_fps:.1f}")
                
            # Create header
            header = Header()
            header.stamp = self.rgb_stamp
            header.frame_id = "camera_color_optical_frame"
            
            # Publish semantic labels
            semantic_msg = self.bridge.cv2_to_imgmsg(labels, encoding='mono8')
            semantic_msg.header = header
            self.semantic_pub.publish(semantic_msg)
            
            # Publish colored visualization
            if self.publish_color:
                colored = NYUv2ColorPalette.colorize(labels)
                colored_msg = self.bridge.cv2_to_imgmsg(colored, encoding='bgr8')
                colored_msg.header = header
                self.semantic_color_pub.publish(colored_msg)
                
            # Publish uncertainty
            if self.publish_uncertainty:
                uncertainty_uint8 = (uncertainty * 255).astype(np.uint8)
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
