#!/usr/bin/env python3
"""
Convert ONNX model to TFLite with INT8 quantization.

This script converts a donkeycar LinearMW ONNX model to TFLite format.
Run this on an x86_64 machine if you have TensorFlow compatibility issues on ARM.

Requirements:
    pip install tensorflow numpy pillow
    pip install onnx==1.10.2 onnx-tf==1.10.0

Usage:
    # Basic conversion (without INT8 quantization)
    python convert_onnx_to_tflite.py --input model.onnx --output model.tflite
    
    # With INT8 quantization using representative dataset
    python convert_onnx_to_tflite.py --input model.onnx --output model.tflite \\
        --data-dry data_dry --data-wet data_wet --data-icy data_icy
    
    # Single data directory
    python convert_onnx_to_tflite.py --input model.onnx --output model.tflite \\
        --data-dry data_all
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path


def load_representative_dataset(data_paths, max_images=500):
    """
    Load representative images from donkeycar tub directories for TFLite quantization.
    
    Args:
        data_paths: List of paths to tub directories (can be directories containing tubs)
        max_images: Maximum number of images to load (default 500)
    
    Returns:
        numpy array of images in shape [N, 120, 160, 3] with float32 values [0, 1]
    """
    print(f"\nLoading representative dataset from {len(data_paths)} data paths...")
    
    images = []
    images_per_path = max(1, max_images // len(data_paths))
    
    for data_path in data_paths:
        if data_path is None:
            continue
            
        path = Path(data_path)
        if not path.exists():
            print(f"  Warning: Path not found: {data_path}")
            continue
        
        # Find all image files in the directory and subdirectories
        # Donkeycar stores images as .jpg files in tub directories
        image_files = []
        
        # Check if this is a single tub or a directory containing tubs
        if (path / 'catalog_manifest.json').exists() or (path / 'manifest.json').exists():
            # Single tub directory
            image_files = sorted(path.glob('images/*.jpg'))[:images_per_path]
        else:
            # Directory potentially containing multiple tubs
            for tub_dir in sorted(path.iterdir()):
                if tub_dir.is_dir():
                    tub_images = sorted(tub_dir.glob('images/*.jpg'))
                    image_files.extend(tub_images)
                    if len(image_files) >= images_per_path:
                        break
            image_files = image_files[:images_per_path]
        
        # Also check for direct .jpg files in case of different structure
        if len(image_files) == 0:
            image_files = sorted(path.glob('**/*.jpg'))[:images_per_path]
        
        print(f"  Loading {len(image_files)} images from {data_path}")
        
        # Load images
        for img_file in image_files:
            try:
                # Use PIL to load image
                try:
                    from PIL import Image
                    img = Image.open(img_file)
                    img_array = np.array(img)
                except ImportError:
                    import cv2
                    img_array = cv2.imread(str(img_file))
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                
                # Normalize to [0, 1] float32
                img_array = img_array.astype('float32') / 255.0
                
                # Ensure shape is (120, 160, 3)
                if img_array.shape != (120, 160, 3):
                    # Resize if needed
                    try:
                        from PIL import Image as PILImage
                        img_pil = PILImage.fromarray((img_array * 255).astype('uint8'))
                        img_pil = img_pil.resize((160, 120))
                        img_array = np.array(img_pil).astype('float32') / 255.0
                    except:
                        print(f"    Warning: Skipping image with shape {img_array.shape}")
                        continue
                
                images.append(img_array)
                
                if len(images) >= max_images:
                    break
                    
            except Exception as e:
                print(f"    Warning: Failed to load {img_file}: {e}")
                continue
        
        if len(images) >= max_images:
            break
    
    if len(images) == 0:
        raise ValueError("No images could be loaded from the provided data paths")
    
    images_array = np.array(images, dtype='float32')
    print(f"  Loaded {len(images)} images total, shape: {images_array.shape}")
    
    return images_array


def convert_onnx_to_tflite(onnx_path, tflite_path, representative_images=None):
    """
    Convert ONNX model to TFLite with optional INT8 quantization.
    
    Args:
        onnx_path: Path to ONNX model
        tflite_path: Path to save TFLite model
        representative_images: Optional numpy array of images [N, 120, 160, 3] in float32 [0, 1]
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow")
    
    try:
        import onnx
        import onnx2tf
    except ImportError:
        raise ImportError("onnx and onnx2tf are required. Install with: pip install onnx onnx2tf")
    
    print(f"\nConverting ONNX to TFLite: {onnx_path} -> {tflite_path}")
    
    # Step 1: Load ONNX model
    print("  Step 1: Loading ONNX model...")
    onnx_model = onnx.load(onnx_path)
    
    # Step 2: Convert ONNX to TensorFlow SavedModel
    tf_model_path = onnx_path.replace('.onnx', '_tf_model')
    print(f"  Step 2: Converting ONNX to TensorFlow SavedModel...")
    print(f"          Output: {tf_model_path}")
    
    # Use onnx2tf for conversion
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=tf_model_path,
        output_signaturedefs=True,
        non_verbose=True
    )
    
    # Step 3: Create representative dataset generator (if provided)
    if representative_images is not None:
        def representative_data_gen():
            """Generator for representative dataset."""
            # Images are [N, 120, 160, 3] in HWC format, need to convert to [N, 3, 120, 160] CHW
            images_chw = np.transpose(representative_images, (0, 3, 1, 2)).astype('float32')
            
            # Generate surface_id values (cycle through 0, 1, 2)
            n_images = len(images_chw)
            surface_ids = np.array([i % 3 for i in range(n_images)], dtype='int64')
            
            for img, surf_id in zip(images_chw, surface_ids):
                # Yield as list of inputs
                yield [
                    img[np.newaxis, ...].astype('float32'),  # image: [1, 3, 120, 160]
                    np.array([surf_id], dtype='int64')  # surface_id: [1]
                ]
    
    # Step 5: Convert to TFLite with quantization
    print(f"  Step 4: Converting to TFLite...")
    
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    
    if representative_images is not None:
        print(f"           Using {len(representative_images)} representative images for INT8 quantization")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        
        # Try INT8 quantization first
        try:
            print("  Attempting full INT8 quantization...")
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            tflite_model = converter.convert()
            print("  INT8 quantization successful!")
        except Exception as e:
            print(f"  Warning: INT8 quantization failed: {e}")
            print("  Falling back to default dynamic range quantization...")
            # Fallback to default quantization without strict INT8
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_data_gen
            tflite_model = converter.convert()
    else:
        print("  No representative dataset provided, using default quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
    
    # Step 6: Save TFLite model
    print(f"  Step 5: Saving TFLite model...")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    tflite_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"  TFLite model saved successfully! Size: {tflite_size_mb:.2f} MB")
    
    # Cleanup temporary TF SavedModel directory
    import shutil
    if os.path.exists(tf_model_path):
        shutil.rmtree(tf_model_path)
        print(f"  Cleaned up temporary TensorFlow model directory")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert ONNX model to TFLite format with optional INT8 quantization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion without INT8 quantization:
  python convert_onnx_to_tflite.py --input model.onnx --output model.tflite
  
  # With INT8 quantization using representative dataset:
  python convert_onnx_to_tflite.py --input model.onnx --output model.tflite \\
      --data-dry data_dry --data-wet data_wet --data-icy data_icy
  
  # Using a single data directory:
  python convert_onnx_to_tflite.py --input model.onnx --output model.tflite \\
      --data-dry data_all
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input ONNX model')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Path to output TFLite model')
    parser.add_argument('--data-dry', type=str, default=None,
                        help='Path to dry surface data directory (for INT8 quantization)')
    parser.add_argument('--data-wet', type=str, default=None,
                        help='Path to wet surface data directory (for INT8 quantization)')
    parser.add_argument('--data-icy', type=str, default=None,
                        help='Path to icy surface data directory (for INT8 quantization)')
    parser.add_argument('--max-images', type=int, default=500,
                        help='Maximum number of images to use for representative dataset (default: 500)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input ONNX file not found: {args.input}")
        sys.exit(1)
    
    # Build list of data paths
    data_paths = []
    for data_arg in ['data_dry', 'data_wet', 'data_icy']:
        path = getattr(args, data_arg, None)
        if path is not None:
            data_paths.append(path)
    
    print("=" * 70)
    print("ONNX to TFLite Converter")
    print("=" * 70)
    print(f"\nInput:  {args.input}")
    print(f"Output: {args.output}")
    
    if data_paths:
        print(f"\nRepresentative dataset:")
        print(f"  Dry data: {args.data_dry if args.data_dry else 'Not provided'}")
        print(f"  Wet data: {args.data_wet if args.data_wet else 'Not provided'}")
        print(f"  Icy data: {args.data_icy if args.data_icy else 'Not provided'}")
        print(f"  Max images: {args.max_images}")
    else:
        print("\nNo representative dataset provided - using default quantization")
    print()
    
    # Load representative dataset if provided
    representative_images = None
    if data_paths:
        try:
            representative_images = load_representative_dataset(data_paths, max_images=args.max_images)
        except Exception as e:
            print(f"Error loading representative dataset: {e}")
            print("Proceeding with default quantization instead.")
    
    # Convert ONNX to TFLite
    try:
        success = convert_onnx_to_tflite(args.input, args.output, representative_images)
        
        if success:
            print("\n" + "=" * 70)
            print("SUCCESS! TFLite model created.")
            print("=" * 70)
            print(f"\nYou can now use the TFLite model: {args.output}")
            print("Copy it to your target device for inference.")
        else:
            print("\nConversion failed.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
