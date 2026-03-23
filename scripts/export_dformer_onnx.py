#!/usr/bin/env python3
"""
DFormer-Large ONNX Export Script for TensorRT Deployment

This script exports DFormer (ICLR 2024) models to ONNX format for TensorRT conversion.
NOTE: This is for DFormer, NOT DFormerv2.

DFormer uses a unified RGB-D encoder that takes concatenated RGB+Depth as input.

Usage:
    python export_dformer_onnx.py \
        --model_size large \
        --dataset sunrgbd \
        --checkpoint checkpoints/trained/SUNRGBD/DFormer_Large.pth \
        --output_name dformer_large_sunrgbd \
        --opset 11

Requirements:
    - PyTorch 2.0+ (or 1.13+)
    - ONNX 1.14+
    - DFormer repository: https://github.com/VCIP-RGBD/DFormer

Author: Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
Date: 2026
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add DFormer to path
DFORMER_PATH = os.environ.get('DFORMER_PATH', os.path.expanduser('~/DFormer'))


def parse_args():
    parser = argparse.ArgumentParser(description='Export DFormer to ONNX')
    
    # Model configuration
    parser.add_argument('--model_size', type=str, default='large',
                        choices=['tiny', 'small', 'base', 'large'],
                        help='DFormer model size')
    parser.add_argument('--dataset', type=str, default='sunrgbd',
                        choices=['nyuv2', 'sunrgbd'],
                        help='Dataset the model was trained on')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint (.pth or .pth.tar)')
    
    # Input configuration
    parser.add_argument('--height', type=int, default=480,
                        help='Input height')
    parser.add_argument('--width', type=int, default=640,
                        help='Input width')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='onnx_models',
                        help='Output directory')
    parser.add_argument('--output_name', type=str, default='dformer_large',
                        help='Output model name (without extension)')
    parser.add_argument('--opset', type=int, default=11,
                        help='ONNX opset version (11 recommended for TRT 8.5)')
    
    # Export options
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX model with onnx-simplifier')
    parser.add_argument('--check', action='store_true',
                        help='Verify ONNX model outputs match PyTorch')
    
    # DFormer path
    parser.add_argument('--dformer_path', type=str, default=DFORMER_PATH,
                        help='Path to DFormer repository')
    
    return parser.parse_args()


def get_num_classes(dataset):
    """Get number of classes for dataset."""
    return {'sunrgbd': 37, 'nyuv2': 40}.get(dataset, 40)


def load_dformer_model(model_size, num_classes, checkpoint_path, dformer_path):
    """Load DFormer model from the official repository."""
    
    # Add DFormer to path
    if os.path.exists(dformer_path):
        sys.path.insert(0, dformer_path)
        print(f"Added DFormer path: {dformer_path}")
    else:
        raise FileNotFoundError(f"DFormer not found at: {dformer_path}")
    
    # Import DFormer modules
    try:
        from models.builder import EncoderDecoder as DFormerModel
        from mmcv.utils import Config
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nMake sure DFormer is properly installed:")
        print("  cd ~/DFormer")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    # Load config based on model size and dataset
    dataset_name = 'NYUDepthv2' if num_classes == 40 else 'SUNRGBD'
    config_name = f'DFormer_{model_size.capitalize()}'
    config_path = os.path.join(dformer_path, 'local_configs', dataset_name, f'{config_name}.py')
    
    if not os.path.exists(config_path):
        # Try alternative path
        config_path = os.path.join(dformer_path, 'local_configs', 'NYUDepthv2', f'{config_name}.py')
    
    print(f"Loading config: {config_path}")
    cfg = Config.fromfile(config_path)
    
    # Build model
    model = DFormerModel(cfg=cfg)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Load weights
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        print("Checkpoint loaded successfully")
    else:
        print(f"WARNING: Checkpoint not found: {checkpoint_path}")
        print("Exporting with random weights (for structure testing only)")
    
    model.eval()
    return model


class DFormerWrapper(nn.Module):
    """Wrapper for DFormer ONNX export.
    
    DFormer takes RGB and Depth as separate inputs internally,
    but we wrap it to accept them as separate ONNX inputs for clarity.
    """
    
    def __init__(self, model, height, width, num_classes):
        super().__init__()
        self.model = model
        self.height = height
        self.width = width
        self.num_classes = num_classes
    
    def forward(self, rgb, depth):
        """
        Args:
            rgb: [1, 3, H, W] - RGB image, normalized (ImageNet stats)
            depth: [1, 3, H, W] - Depth image (3-channel, grayscale replicated)
        
        Returns:
            output: [1, num_classes, H, W] - Segmentation logits
        """
        # DFormer expects depth as 3-channel input
        # The model internally handles the fusion
        output = self.model(rgb, depth)
        
        # Handle different output formats
        if isinstance(output, (list, tuple)):
            output = output[0]
        if isinstance(output, dict):
            output = output.get('pred', output.get('out', list(output.values())[0]))
        
        # Ensure correct output size
        if output.shape[2:] != (self.height, self.width):
            output = nn.functional.interpolate(
                output, size=(self.height, self.width),
                mode='bilinear', align_corners=False
            )
        
        return output


def export_to_onnx(model, height, width, num_classes, output_path, opset_version):
    """Export DFormer model to ONNX format."""
    
    print(f"\nExporting to ONNX (opset {opset_version})...")
    
    # Wrap model
    wrapper = DFormerWrapper(model, height, width, num_classes)
    wrapper.eval()
    
    # Create dummy inputs
    # RGB: standard 3-channel
    # Depth: DFormer expects 3-channel depth (grayscale replicated to 3 channels)
    dummy_rgb = torch.randn(1, 3, height, width)
    dummy_depth = torch.randn(1, 3, height, width)  # 3-channel for DFormer!
    
    # Export
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_rgb, dummy_depth),
            output_path,
            input_names=['rgb', 'depth'],
            output_names=['output'],
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )
    
    print(f"ONNX model saved: {output_path}")
    
    # Get file sizes
    onnx_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ONNX size: {onnx_size:.1f} MB")
    
    data_path = output_path + '.data'
    if os.path.exists(data_path):
        data_size = os.path.getsize(data_path) / (1024 * 1024)
        print(f"  External data: {data_size:.1f} MB")
    
    # Verify ONNX model
    import onnx
    model_onnx = onnx.load(output_path)
    onnx.checker.check_model(model_onnx)
    print("ONNX model verification: PASSED")
    
    # Print input/output info
    print(f"\nModel I/O:")
    for inp in model_onnx.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  Input '{inp.name}': {shape}")
    for out in model_onnx.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  Output '{out.name}': {shape}")
    
    return output_path


def simplify_onnx(input_path):
    """Simplify ONNX model."""
    try:
        import onnxsim
        import onnx
        
        print("\nSimplifying ONNX model...")
        model = onnx.load(input_path)
        model_simp, check = onnxsim.simplify(model)
        
        if check:
            output_path = input_path.replace('.onnx', '_sim.onnx')
            onnx.save(model_simp, output_path)
            
            # Size comparison
            orig_size = os.path.getsize(input_path) / (1024 * 1024)
            simp_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  Original: {orig_size:.1f} MB -> Simplified: {simp_size:.1f} MB")
            
            return output_path
        else:
            print("  Simplification check failed, keeping original")
            return input_path
            
    except ImportError:
        print("  onnx-simplifier not installed (pip install onnx-simplifier)")
        return input_path


def verify_outputs(pytorch_model, onnx_path, height, width):
    """Verify ONNX outputs match PyTorch."""
    import onnxruntime as ort
    
    print("\nVerifying ONNX outputs...")
    
    # Create test input
    test_rgb = torch.randn(1, 3, height, width)
    test_depth = torch.randn(1, 3, height, width)
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        wrapper = DFormerWrapper(pytorch_model, height, width, 37)
        pt_output = wrapper(test_rgb, test_depth).numpy()
    
    # ONNX inference
    session = ort.InferenceSession(onnx_path)
    onnx_output = session.run(
        None, 
        {'rgb': test_rgb.numpy(), 'depth': test_depth.numpy()}
    )[0]
    
    # Compare
    diff = np.abs(pt_output - onnx_output).max()
    print(f"  Max absolute difference: {diff:.6f}")
    
    if diff < 1e-4:
        print("  Verification: PASSED (diff < 1e-4)")
    elif diff < 1e-2:
        print("  Verification: OK (diff < 1e-2, FP precision)")
    else:
        print("  Verification: WARNING (large diff, check model)")


def main():
    args = parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    num_classes = get_num_classes(args.dataset)
    
    print("=" * 60)
    print("DFormer-Large ONNX Export (ICLR 2024)")
    print("=" * 60)
    print(f"Model: DFormer-{args.model_size.capitalize()}")
    print(f"Dataset: {args.dataset} ({num_classes} classes)")
    print(f"Input size: {args.width}x{args.height}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"ONNX opset: {args.opset}")
    
    # Load model
    model = load_dformer_model(
        args.model_size, 
        num_classes, 
        args.checkpoint,
        args.dformer_path
    )
    
    # Export
    output_path = os.path.join(
        args.output_dir,
        f"{args.output_name}_opset{args.opset}.onnx"
    )
    export_to_onnx(model, args.height, args.width, num_classes, output_path, args.opset)
    
    # Simplify
    if args.simplify:
        output_path = simplify_onnx(output_path)
    
    # Verify
    if args.check:
        verify_outputs(model, output_path, args.height, args.width)
    
    # Print deployment instructions
    print("\n" + "=" * 60)
    print("DEPLOYMENT INSTRUCTIONS")
    print("=" * 60)

if __name__ == '__main__':
    main()