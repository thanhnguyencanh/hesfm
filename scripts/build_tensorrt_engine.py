#!/usr/bin/env python3
"""
Build TensorRT Engine for ESANet

Converts ESANet PyTorch model to TensorRT FP16 engine for Jetson deployment.

Usage:
    # From ONNX (recommended)
    python build_tensorrt_engine.py --onnx models/esanet_r34_nyuv2.onnx \
        --output models/esanet_r34_nyuv2_fp16.engine --precision fp16
    
    # From PyTorch (requires ESANet code)
    python build_tensorrt_engine.py --checkpoint models/esanet_r34_nyuv2.pth \
        --output models/esanet_r34_nyuv2_fp16.engine --precision fp16

Requirements:
    - TensorRT 8.5+ (included in JetPack 5.1+)
    - pycuda
    - torch (for PyTorch conversion)
    - onnx (for ONNX validation)

Author: Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
Date: 2026
"""

import argparse
import os
import sys
import time
import numpy as np

# Check TensorRT availability
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
    TRT_VERSION = trt.__version__
except ImportError:
    TRT_AVAILABLE = False
    TRT_VERSION = None
    print("WARNING: TensorRT not available. Install TensorRT to use this script.")

# Check PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Check ONNX availability
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class TensorRTEngineBuilder:
    """Build TensorRT engine from ONNX or PyTorch model."""
    
    def __init__(self, 
                 precision='fp16',
                 max_workspace_size=1 << 30,  # 1GB
                 max_batch_size=1,
                 verbose=False):
        """
        Initialize TensorRT engine builder.
        
        Args:
            precision: 'fp32', 'fp16', or 'int8'
            max_workspace_size: Maximum GPU memory for TensorRT (bytes)
            max_batch_size: Maximum batch size
            verbose: Enable verbose logging
        """
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available")
            
        self.precision = precision
        self.max_workspace_size = max_workspace_size
        self.max_batch_size = max_batch_size
        self.verbose = verbose
        
        # Create TensorRT logger
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        
        # Create builder
        self.builder = trt.Builder(self.logger)
        self.config = self.builder.create_builder_config()
        
        # Set workspace size
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
        
        # Set precision
        if precision == 'fp16':
            if self.builder.platform_has_fast_fp16:
                self.config.set_flag(trt.BuilderFlag.FP16)
                print("FP16 mode enabled")
            else:
                print("WARNING: FP16 not supported on this platform, using FP32")
        elif precision == 'int8':
            if self.builder.platform_has_fast_int8:
                self.config.set_flag(trt.BuilderFlag.INT8)
                print("INT8 mode enabled")
            else:
                print("WARNING: INT8 not supported on this platform, using FP32")
                
    def build_from_onnx(self, onnx_path, output_path, 
                        input_shapes=None,
                        dynamic_axes=False):
        """
        Build TensorRT engine from ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Path to save TensorRT engine
            input_shapes: Dict of input names to shapes, e.g., {'rgb': (1, 3, 480, 640)}
            dynamic_axes: Enable dynamic input shapes
            
        Returns:
            True if successful
        """
        print(f"Building TensorRT engine from ONNX: {onnx_path}")
        print(f"TensorRT version: {TRT_VERSION}")
        
        # Validate ONNX model
        if ONNX_AVAILABLE:
            try:
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                print("ONNX model validation passed")
            except Exception as e:
                print(f"WARNING: ONNX validation failed: {e}")
        
        # Create network
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = self.builder.create_network(network_flags)
        
        # Create ONNX parser
        parser = trt.OnnxParser(network, self.logger)
        
        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"ONNX Parser Error: {parser.get_error(i)}")
                return False
                
        print(f"Network inputs: {network.num_inputs}")
        print(f"Network outputs: {network.num_outputs}")
        
        # Print input/output info
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            print(f"  Input {i}: {input_tensor.name}, shape={input_tensor.shape}, dtype={input_tensor.dtype}")
            
        for i in range(network.num_outputs):
            output_tensor = network.get_output(i)
            print(f"  Output {i}: {output_tensor.name}, shape={output_tensor.shape}, dtype={output_tensor.dtype}")
        
        # Set input shapes (optimization profile)
        if input_shapes or dynamic_axes:
            profile = self.builder.create_optimization_profile()
            
            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                name = input_tensor.name
                
                if input_shapes and name in input_shapes:
                    shape = input_shapes[name]
                else:
                    # Use default shape from ONNX
                    shape = tuple(input_tensor.shape)
                    # Replace -1 (dynamic) with default values
                    shape = tuple(s if s > 0 else 1 for s in shape)
                
                # Set min, opt, max shapes (same for static)
                profile.set_shape(name, shape, shape, shape)
                print(f"  Set shape for {name}: {shape}")
                
            self.config.add_optimization_profile(profile)
        
        # Build engine
        print("Building TensorRT engine (this may take several minutes)...")
        start_time = time.time()
        
        serialized_engine = self.builder.build_serialized_network(network, self.config)
        
        if serialized_engine is None:
            print("ERROR: Failed to build TensorRT engine")
            return False
            
        build_time = time.time() - start_time
        print(f"Engine built in {build_time:.1f} seconds")
        
        # Save engine
        with open(output_path, 'wb') as f:
            f.write(serialized_engine)
            
        print(f"TensorRT engine saved to: {output_path}")
        print(f"Engine size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
        
        return True
        
    def build_from_pytorch(self, checkpoint_path, output_path,
                           model_class=None,
                           input_shapes=None):
        """
        Build TensorRT engine from PyTorch checkpoint.
        
        Args:
            checkpoint_path: Path to PyTorch checkpoint (.pth)
            output_path: Path to save TensorRT engine
            model_class: PyTorch model class (if None, tries to load ESANet)
            input_shapes: Dict of input names to shapes
            
        Returns:
            True if successful
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for checkpoint conversion")
            
        print(f"Converting PyTorch checkpoint: {checkpoint_path}")
        
        # Default input shapes for ESANet
        if input_shapes is None:
            input_shapes = {
                'rgb': (1, 3, 480, 640),
                'depth': (1, 1, 480, 640)
            }
        
        # Load model
        if model_class is None:
            # Try to import ESANet
            try:
                from esanet import ESANet
                model = ESANet(
                    height=480,
                    width=640,
                    num_classes=40,
                    encoder='resnet34',
                    encoder_block='NonBottleneck1D'
                )
            except ImportError:
                print("ERROR: ESANet not found. Please provide model_class or install ESANet.")
                print("       Alternatively, export to ONNX first using the ESANet repository.")
                return False
        else:
            model = model_class()
            
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        print("PyTorch model loaded")
        
        # Export to ONNX
        onnx_path = output_path.replace('.engine', '.onnx')
        
        # Create dummy inputs
        dummy_inputs = {}
        input_names = []
        for name, shape in input_shapes.items():
            dummy_inputs[name] = torch.randn(shape)
            input_names.append(name)
            
        # Export
        print(f"Exporting to ONNX: {onnx_path}")
        torch.onnx.export(
            model,
            tuple(dummy_inputs.values()),
            onnx_path,
            input_names=input_names,
            output_names=['output'],
            opset_version=13,
            do_constant_folding=True
        )
        print("ONNX export complete")
        
        # Build from ONNX
        return self.build_from_onnx(onnx_path, output_path, input_shapes)


def verify_engine(engine_path, input_shapes=None):
    """
    Verify TensorRT engine by running inference.
    
    Args:
        engine_path: Path to TensorRT engine
        input_shapes: Dict of input names to shapes
    """
    if not TRT_AVAILABLE:
        print("TensorRT not available for verification")
        return
        
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        print("PyCUDA not available for verification")
        return
        
    print(f"\nVerifying engine: {engine_path}")
    
    # Load engine
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(f.read())
        
    if engine is None:
        print("ERROR: Failed to load engine")
        return
        
    print(f"Engine loaded successfully")
    print(f"  Num bindings: {engine.num_bindings}")
    
    # Create execution context
    context = engine.create_execution_context()
    
    # Allocate buffers
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        shape = engine.get_binding_shape(i)
        size = trt.volume(shape)
        
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        
        if engine.binding_is_input(i):
            inputs.append({'host': host_mem, 'device': device_mem, 'name': name, 'shape': shape})
            print(f"  Input {i}: {name}, shape={shape}, dtype={dtype}")
        else:
            outputs.append({'host': host_mem, 'device': device_mem, 'name': name, 'shape': shape})
            print(f"  Output {i}: {name}, shape={shape}, dtype={dtype}")
    
    # Fill inputs with random data
    for inp in inputs:
        np.copyto(inp['host'], np.random.randn(*inp['shape']).astype(inp['host'].dtype).ravel())
        cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
        
    # Run inference
    print("\nRunning inference...")
    start_time = time.time()
    
    for _ in range(10):  # Warmup
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    stream.synchronize()
    
    # Benchmark
    num_runs = 100
    start_time = time.time()
    for _ in range(num_runs):
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    stream.synchronize()
    elapsed = time.time() - start_time
    
    # Copy outputs
    for out in outputs:
        cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
    stream.synchronize()
    
    print(f"Inference successful!")
    print(f"  Average latency: {elapsed / num_runs * 1000:.2f} ms")
    print(f"  Throughput: {num_runs / elapsed:.1f} FPS")
    
    # Print output stats
    for out in outputs:
        output_data = out['host'].reshape(out['shape'])
        print(f"  Output '{out['name']}': min={output_data.min():.3f}, max={output_data.max():.3f}, mean={output_data.mean():.3f}")


def main():
    parser = argparse.ArgumentParser(
        description='Build TensorRT engine for ESANet semantic segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build from ONNX with FP16
  python build_tensorrt_engine.py --onnx models/esanet.onnx --output models/esanet_fp16.engine --precision fp16
  
  # Build from PyTorch checkpoint
  python build_tensorrt_engine.py --checkpoint models/esanet.pth --output models/esanet_fp16.engine --precision fp16
  
  # Build with custom input size
  python build_tensorrt_engine.py --onnx models/esanet.onnx --output models/esanet_fp16.engine --height 480 --width 640
  
  # Verify engine
  python build_tensorrt_engine.py --verify models/esanet_fp16.engine
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--onnx', type=str, help='Path to ONNX model')
    input_group.add_argument('--checkpoint', type=str, help='Path to PyTorch checkpoint')
    input_group.add_argument('--verify', type=str, help='Verify existing engine')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output engine path')
    
    # Precision
    parser.add_argument('--precision', type=str, default='fp16',
                        choices=['fp32', 'fp16', 'int8'],
                        help='Inference precision (default: fp16)')
    
    # Input shapes
    parser.add_argument('--height', type=int, default=480, help='Input height')
    parser.add_argument('--width', type=int, default=640, help='Input width')
    parser.add_argument('--num-classes', type=int, default=40, help='Number of classes')
    
    # Builder options
    parser.add_argument('--workspace', type=int, default=1024,
                        help='Max workspace size in MB (default: 1024)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Check TensorRT
    if not TRT_AVAILABLE:
        print("ERROR: TensorRT is not installed")
        print("On Jetson: TensorRT is included in JetPack")
        print("On desktop: pip install tensorrt")
        sys.exit(1)
    
    # Verify mode
    if args.verify:
        verify_engine(args.verify)
        return
        
    # Check inputs
    if not args.onnx and not args.checkpoint:
        parser.error("Either --onnx or --checkpoint is required")
        
    if not args.output:
        if args.onnx:
            args.output = args.onnx.replace('.onnx', f'_{args.precision}.engine')
        else:
            args.output = args.checkpoint.replace('.pth', f'_{args.precision}.engine')
    
    # Input shapes
    input_shapes = {
        'rgb': (1, 3, args.height, args.width),
        'depth': (1, 1, args.height, args.width)
    }
    
    # Build engine
    builder = TensorRTEngineBuilder(
        precision=args.precision,
        max_workspace_size=args.workspace * 1024 * 1024,
        verbose=args.verbose
    )
    
    if args.onnx:
        success = builder.build_from_onnx(args.onnx, args.output, input_shapes)
    else:
        success = builder.build_from_pytorch(args.checkpoint, args.output, input_shapes=input_shapes)
        
    if success:
        print("\n" + "=" * 50)
        print("Build successful!")
        print("=" * 50)
        
        # Verify
        verify_engine(args.output, input_shapes)
    else:
        print("\nBuild failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()