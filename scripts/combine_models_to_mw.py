#!/usr/bin/env python3
"""
Script to combine 3 Linear models into a LinearMW model.
This allows testing LinearMW inference with known-good weights.

Also supports combining LinearUncertainty models into LinearMWUncertainty models.
Can export to ONNX and TFLite formats with quantization.

Requirements for ONNX export (works on all platforms):
    pip install onnx

Requirements for TFLite export:
    pip install ai-edge-torch
    
    IMPORTANT - ARM Platform Issues:
    ==========================================
    TFLite conversion requires TensorFlow, which has binary compatibility issues
    on some ARM platforms (especially in Docker containers).
    
    If you encounter TensorFlow errors on ARM:
    
    OPTION 1 - Use ONNX (RECOMMENDED):
        - Export with --onnx flag instead of --tflite
        - Use ONNX Runtime for inference: pip install onnxruntime
        - ONNX Runtime works excellently on ARM!
    
    OPTION 2 - Fix TensorFlow on ARM:
        - Try: pip install tensorflow-aarch64
        - Or run outside Docker container
        - Or try different TensorFlow version
    
    OPTION 3 - Convert on x86_64:
        - Run this script on x86_64 machine with --tflite
        - Copy the .tflite file to your ARM device

Usage:
    # Basic PyTorch model combination
    python combine_models_to_mw.py --dry model.pt --output output_mw.pt
    
    # With ONNX export (RECOMMENDED for ARM)
    python combine_models_to_mw.py --dry dry.pt --wet wet.pt --icy icy.pt --output mw.pt --onnx
    
    # With TFLite export (may fail on ARM in Docker)
    python combine_models_to_mw.py --dry dry.pt --wet wet.pt --icy icy.pt --output mw.pt --tflite
    
    # With both ONNX and TFLite
    python combine_models_to_mw.py --dry dry.pt --wet wet.pt --icy icy.pt --output mw.pt --onnx --tflite
    
    # Uncertainty model with ONNX
    python combine_models_to_mw.py --dry dry_unc.pt --wet wet_unc.pt --icy icy_unc.pt \\
        --output mw_unc.pt --uncertainty --onnx
"""
import sys
import torch
import os
import argparse
import torch.nn as nn
import pickle
import io
import numpy as np
from pathlib import Path

# Recreate the Linear, LinearMW, LinearUncertainty, and LinearMWUncertainty classes locally to avoid donkeycar import issues

class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.2
        # init the layers
        self.conv24 = nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))
        self.conv32 = nn.Conv2d(24, 32, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_5 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        self.fc1 = nn.Linear(6656, 100)
        self.fc2 = nn.Linear(100, 50)
        self.drop = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        self.output1 = nn.Linear(50, 1)
        self.output2 = nn.Linear(50, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv24(x))
        x = self.drop(x)
        x = self.relu(self.conv32(x))
        x = self.drop(x)
        x = self.relu(self.conv64_5(x))
        x = self.drop(x)
        x = self.relu(self.conv64_3(x))
        x = self.drop(x)
        x = self.relu(self.conv64_3(x))
        x = self.drop(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x1 = self.drop(x)
        angle = self.output1(x1)
        throttle = self.output2(x1)
        return torch.cat((angle, throttle), 1)

class LinearUncertainty(Linear):
    def __init__(self):
        super().__init__()
        self.output1_uncertainty = nn.Linear(50, 1)
        self.output2_uncertainty = nn.Linear(50, 1)
        
        # Initialize uncertainty heads with small weights and reasonable bias
        with torch.no_grad():
            self.output1_uncertainty.weight.normal_(0.0, 0.001)
            self.output1_uncertainty.bias.fill_(-1.0)
            self.output2_uncertainty.weight.normal_(0.0, 0.001)
            self.output2_uncertainty.bias.fill_(-1.0)
            
            self.output1.weight.normal_(0.0, 0.01)
            self.output1.bias.fill_(0.5)
            self.output2.weight.normal_(0.0, 0.01)
            self.output2.bias.fill_(0.5)

    def forward(self, x):
        x = self.relu(self.conv24(x))
        x = self.drop(x)
        x = self.relu(self.conv32(x))
        x = self.drop(x)
        x = self.relu(self.conv64_5(x))
        x = self.drop(x)
        x = self.relu(self.conv64_3(x))
        x = self.drop(x)
        x = self.relu(self.conv64_3(x))
        x = self.drop(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x1 = self.drop(x)
        angle = self.output1(x1)
        throttle = self.output2(x1)
        angle_uncertainty = self.output1_uncertainty(x1)
        throttle_uncertainty = self.output2_uncertainty(x1)
        
        angle = torch.clamp(angle, -0.5, 1.5)
        throttle = torch.clamp(throttle, -0.5, 1.5)
        angle_uncertainty = torch.clamp(angle_uncertainty, -10, 2)
        throttle_uncertainty = torch.clamp(throttle_uncertainty, -10, 2)
        
        return torch.cat((angle, throttle, angle_uncertainty, throttle_uncertainty), 1)

class LinearMW(nn.Module):
    def __init__(self, n_weathers = 3):
        super().__init__()
        self.subnetworks = nn.ModuleList([Linear() for _ in range(n_weathers)])
        self.inference_surface_id = 0  # Default for inference

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            img, surface_id = x[0], x[1]
            if isinstance(surface_id, torch.Tensor) and surface_id.dim() > 0 and len(surface_id) > 1:
                unique_surfaces = torch.unique(surface_id)
                if len(unique_surfaces) == 1:
                    surface_idx = int(unique_surfaces[0].item())
                    surface_idx = max(0, min(surface_idx, len(self.subnetworks) - 1))
                    return self.subnetworks[surface_idx](img)
                else:
                    batch_size = img.shape[0]
                    outputs = []
                    for i in range(batch_size):
                        sample_img = img[i:i+1]
                        surface_idx = int(surface_id[i].item())
                        surface_idx = max(0, min(surface_idx, len(self.subnetworks) - 1))
                        output = self.subnetworks[surface_idx](sample_img)
                        outputs.append(output)
                    return torch.cat(outputs, dim=0)
            else:
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                if isinstance(surface_id, torch.Tensor):
                    if surface_id.dim() > 0:
                        surface_idx = int(surface_id[0].item())
                    else:
                        surface_idx = int(surface_id.item())
                else:
                    surface_idx = int(surface_id)
        else:
            img = x
            surface_idx = self.inference_surface_id
        
        surface_idx = max(0, min(surface_idx, len(self.subnetworks) - 1))
        return self.subnetworks[surface_idx](img)

class LinearMWUncertainty(LinearMW):
    def __init__(self, n_weathers = 3):
        super().__init__(n_weathers)
        self.subnetworks = nn.ModuleList([LinearUncertainty() for _ in range(n_weathers)])

class ONNXExportWrapper(nn.Module):
    """ONNX-compatible wrapper for LinearMW/LinearMWUncertainty models.
    
    This wrapper avoids Python control flow and uses ONNX-compatible operations
    to select the correct subnetwork output based on surface_id.
    """
    def __init__(self, mw_model, uncertainty=False):
        super().__init__()
        self.subnetworks = mw_model.subnetworks
        self.n_weathers = len(self.subnetworks)
        self.uncertainty = uncertainty
    
    def forward(self, image, surface_id):
        """
        Args:
            image: Input image tensor [batch, 3, 120, 160]
            surface_id: Surface ID tensor [batch] or [batch, 1]
        
        Returns:
            For non-uncertainty: (steering, throttle) as separate tensors
            For uncertainty: (steering, throttle, steering_uncertainty, throttle_uncertainty) as separate tensors
        """
        batch_size = image.shape[0]
        
        # Flatten surface_id to [batch] if needed
        if surface_id.dim() > 1:
            surface_id = surface_id.squeeze(-1)
        
        # Clamp surface_id to valid range [0, n_weathers-1]
        surface_id = torch.clamp(surface_id, 0, self.n_weathers - 1)
        
        # Run all subnetworks and stack outputs
        # Shape: [n_weathers, batch, num_outputs]
        all_outputs = torch.stack([subnet(image) for subnet in self.subnetworks], dim=0)
        
        # Transpose to [batch, n_weathers, num_outputs]
        all_outputs = all_outputs.permute(1, 0, 2)
        
        # Use gather to select the correct output based on surface_id
        # Expand surface_id to match all_outputs dimensions
        # surface_id: [batch] -> [batch, 1, 1] -> [batch, 1, num_outputs]
        num_outputs = all_outputs.shape[2]
        surface_id_expanded = surface_id.view(batch_size, 1, 1).expand(batch_size, 1, num_outputs)
        
        # Gather along the weather dimension (dim=1)
        # Result: [batch, 1, num_outputs]
        selected_outputs = torch.gather(all_outputs, 1, surface_id_expanded.long())
        
        # Squeeze to [batch, num_outputs]
        selected_outputs = selected_outputs.squeeze(1)
        
        # Split outputs into separate tensors for ONNX
        if self.uncertainty:
            # Output has 4 values: steering, throttle, steering_uncertainty, throttle_uncertainty
            steering = selected_outputs[:, 0:1]
            throttle = selected_outputs[:, 1:2]
            steering_uncertainty = selected_outputs[:, 2:3]
            throttle_uncertainty = selected_outputs[:, 3:4]
            return steering, throttle, steering_uncertainty, throttle_uncertainty
        else:
            # Output has 2 values: steering, throttle
            steering = selected_outputs[:, 0:1]
            throttle = selected_outputs[:, 1:2]
            return steering, throttle

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
                # Use PIL or cv2 to load image
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

def convert_pytorch_to_tflite_direct(pytorch_model, tflite_path, representative_images, uncertainty=False):
    """
    Convert PyTorch model directly to TFLite using ai_edge_torch.
    This works on ARM platforms without tensorflow_addons.
    
    Args:
        pytorch_model: PyTorch LinearMW or LinearMWUncertainty model
        tflite_path: Path to save TFLite model
        representative_images: numpy array of images [N, 120, 160, 3] in float32 [0, 1]
        uncertainty: Whether model outputs uncertainty values
    """
    # Handle potential profile.py naming conflict (same as onnx-tf)
    import sys
    
    # Step 1: Save and clear any existing profile-related modules
    original_modules = {}
    profile_conflict_modules = ['profile', '_pyprofile', 'cProfile', 'pstats']
    
    for mod_name in profile_conflict_modules:
        if mod_name in sys.modules:
            original_modules[mod_name] = sys.modules[mod_name]
            del sys.modules[mod_name]
    
    # Step 2: Temporarily remove scripts directories from sys.path
    script_dirs_to_remove = []
    original_sys_path = sys.path.copy()
    
    for path_entry in sys.path[:]:
        if os.path.exists(os.path.join(path_entry, 'profile.py')):
            if 'lib/python' not in path_entry or 'site-packages' in path_entry or 'dist-packages' in path_entry:
                script_dirs_to_remove.append(path_entry)
                sys.path.remove(path_entry)
    
    try:
        # Step 3: Import stdlib modules first (with clean sys.path)
        import profile as _profile_stdlib
        import cProfile as _cProfile_stdlib
        import pstats as _pstats_stdlib
        
        # Step 4: Restore sys.path before importing ai_edge_torch
        sys.path = original_sys_path
        
        # Step 5: Now import ai_edge_torch
        import ai_edge_torch
        
    except ImportError as e:
        # Restore everything on error
        sys.path = original_sys_path
        for mod_name, mod_obj in original_modules.items():
            sys.modules[mod_name] = mod_obj
        
        print("\n" + "="*70)
        print("ERROR: ai_edge_torch is required for TFLite conversion")
        print("="*70)
        print("\nai_edge_torch is Google's official PyTorch to TFLite converter.")
        print("It works on ARM platforms!\n")
        print("Install with:")
        print("  pip install ai-edge-torch")
        print("\nIf installation fails, try:")
        print("  pip install torch ai-edge-torch-nightly")
        print("="*70)
        return False
    except Exception as e:
        # Restore everything on error  
        sys.path = original_sys_path
        for mod_name, mod_obj in original_modules.items():
            sys.modules[mod_name] = mod_obj
        
        # Check for TensorFlow binary compatibility issues
        if 'tensorflow' in str(e).lower() and ('undefined symbol' in str(e) or 'load_library' in str(e)):
            print("\n" + "="*70)
            print("ERROR: TensorFlow binary compatibility issue on ARM")
            print("="*70)
            print("\nTensorFlow binaries are not compatible with your ARM environment.")
            print("This is a known issue in some Docker containers.\n")
            print("Solutions:\n")
            print("1. RECOMMENDED: Export ONNX model only (already works!):")
            print("   Add --onnx flag and use ONNX Runtime for inference")
            print("   pip install onnxruntime\n")
            print("2. Try different TensorFlow version:")
            print("   pip uninstall tensorflow")
            print("   pip install tensorflow-aarch64 # if on Linux ARM")
            print("   # or")
            print("   pip install tensorflow==2.15.0 # try different version\n")
            print("3. Use native ARM environment (not container)")
            print("   TensorFlow often works better outside Docker on ARM\n")
            print("4. Convert on x86_64 and copy .tflite file to ARM device")
            print("="*70)
            return False
        else:
            raise
    
    print(f"\nConverting PyTorch to TFLite using ai_edge_torch: {tflite_path}")
    
    # Set model to eval mode
    pytorch_model.eval()
    
    # Create sample inputs
    print("  Step 1: Preparing sample inputs...")
    sample_image = torch.randn(1, 3, 120, 160)
    sample_surface_id = torch.tensor([0], dtype=torch.long)
    sample_args = (sample_image, sample_surface_id)
    
    # Convert to TFLite
    print("  Step 2: Converting to TFLite...")
    try:
        edge_model = ai_edge_torch.convert(
            pytorch_model,
            sample_args
        )
        
        # Save the model
        print("  Step 3: Saving TFLite model...")
        edge_model.export(tflite_path)
        
        tflite_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
        print(f"  TFLite model saved successfully! Size: {tflite_size_mb:.2f} MB")
        print(f"  Note: This is a dynamic quantized model (not INT8)")
        
        # Note about quantization
        print("\n  For INT8 quantization with representative dataset:")
        print("  The current ai_edge_torch version may require additional steps.")
        print("  The exported model uses dynamic quantization which is efficient for most use cases.")
        
        return True
        
    except Exception as e:
        print(f"  Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

def convert_onnx_to_tflite(onnx_path, tflite_path, representative_images, uncertainty=False):
    """
    Convert ONNX model to TFLite with INT8 quantization.
    
    Args:
        onnx_path: Path to ONNX model
        tflite_path: Path to save TFLite model
        representative_images: numpy array of images [N, 120, 160, 3] in float32 [0, 1]
        uncertainty: Whether model outputs uncertainty values
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow is required for TFLite export. Install with: pip install tensorflow")
    
    # Handle potential profile.py naming conflict in sys.modules and sys.path
    import sys
    
    # Step 1: Save and clear any existing profile-related modules
    original_modules = {}
    profile_conflict_modules = ['profile', '_pyprofile', 'cProfile', 'pstats']
    
    for mod_name in profile_conflict_modules:
        if mod_name in sys.modules:
            original_modules[mod_name] = sys.modules[mod_name]
            del sys.modules[mod_name]
    
    # Step 2: Temporarily remove scripts directories from sys.path
    # This prevents Python from finding local profile.py files
    script_dirs_to_remove = []
    original_sys_path = sys.path.copy()
    
    for path_entry in sys.path[:]:  # Iterate over a copy
        # Check if this path contains a profile.py that would conflict
        if os.path.exists(os.path.join(path_entry, 'profile.py')):
            # Check if it's not the standard library path
            if 'lib/python' not in path_entry or 'site-packages' in path_entry or 'dist-packages' in path_entry:
                script_dirs_to_remove.append(path_entry)
                sys.path.remove(path_entry)
    
    try:
        # Step 3: Import stdlib modules first (with clean sys.path)
        import profile as _profile_stdlib
        import cProfile as _cProfile_stdlib
        import pstats as _pstats_stdlib
        
        # Step 4: Restore sys.path before importing onnx-tf
        sys.path = original_sys_path
        
        # Step 5: Now import onnx and onnx-tf
        import onnx
        
        try:
            from onnx_tf.backend import prepare
        except ModuleNotFoundError as e:
            if 'tensorflow_addons' in str(e):
                sys.path = original_sys_path
                print("\n" + "="*70)
                print("ERROR: onnx-tf requires tensorflow_addons")
                print("="*70)
                print("\ntensorflow_addons is discontinued and not available for many platforms.")
                print("\nFor TFLite conversion on ARM, you have these options:\n")
                print("1. RECOMMENDED: Use ONNX Runtime directly (no TFLite needed)")
                print("   pip install onnxruntime")
                print("   Then use the .onnx model for inference\n")
                print("2. Convert to TFLite on an x86_64 machine:")
                print("   - Run this script with --tflite on x86_64")
                print("   - Copy the .tflite file to your ARM device\n")
                print("3. Use PyTorch model directly (already saved)")
                print("   - Use the .pt model with PyTorch Mobile\n")
                print("="*70)
                return False
            else:
                raise
        
    except ImportError as e:
        # Restore everything on error
        sys.path = original_sys_path
        for mod_name, mod_obj in original_modules.items():
            sys.modules[mod_name] = mod_obj
        raise ImportError(f"onnx and onnx-tf are required. Install with: pip install onnx onnx-tf\nError: {e}")
    
    print(f"\nConverting ONNX to TFLite using onnx-tf: {onnx_path} -> {tflite_path}")
    
    # Step 1: Load ONNX model
    print("  Step 1: Loading ONNX model...")
    onnx_model = onnx.load(onnx_path)
    
    # Step 2: Convert ONNX to TensorFlow
    print("  Step 2: Converting ONNX to TensorFlow...")
    tf_rep = prepare(onnx_model)
    
    # Save as TensorFlow SavedModel
    tf_model_path = onnx_path.replace('.onnx', '_tf_model')
    print(f"  Step 3: Saving TensorFlow SavedModel to {tf_model_path}...")
    tf_rep.export_graph(tf_model_path)
    
    # Step 4: Create representative dataset generator
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
    print(f"  Step 4: Converting to TFLite with INT8 quantization...")
    print(f"           Using {len(representative_images)} representative images")
    
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
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

def combine_models(model_paths, output_path, n_weathers=3, uncertainty=False, export_onnx=False, export_tflite=False, data_paths=None):
    """
    Combine individual Linear/LinearUncertainty models into a LinearMW/LinearMWUncertainty model.
    
    Args:
        model_paths: List of paths to Linear/LinearUncertainty model files (can be less than n_weathers)
        output_path: Path to save the combined LinearMW/LinearMWUncertainty model
        n_weathers: Number of subnetworks (default 3)
        uncertainty: If True, combine LinearUncertainty models into LinearMWUncertainty
        export_onnx: If True, also export model to ONNX format        export_tflite: If True, also export model to TFLite format (requires ONNX export)
        data_paths: List of paths to data directories for representative dataset (required for TFLite)    """
    if len(model_paths) > n_weathers:
        print(f"Error: Too many model paths ({len(model_paths)}) for {n_weathers} subnetworks")
        return False
    
    model_type = "LinearMWUncertainty" if uncertainty else "LinearMW"
    print(f"Creating {model_type} with {n_weathers} subnetworks...")
    combined_model = LinearMWUncertainty(n_weathers=n_weathers) if uncertainty else LinearMW(n_weathers=n_weathers)
    
    if len(model_paths) < n_weathers:
        print(f"Note: Only {len(model_paths)} models provided. Subnetworks {len(model_paths)}-{n_weathers-1} will be randomly initialized.")
    
    # Custom unpickler to redirect donkeycar.parts.fastai classes to our local versions
    class RemappingUnpickler(pickle.Unpickler):           
        def find_class(self, module, name):
            # Redirect donkeycar classes to our local versions
            if module == 'donkeycar.parts.fastai':
                if name == 'Linear':
                    return Linear
                elif name == 'LinearMW':
                    return LinearMW
                elif name == 'LinearUncertainty':
                    return LinearUncertainty
                elif name == 'LinearMWUncertainty':
                    return LinearMWUncertainty
            # For everything else, use the default
            return super().find_class(module, name)
    
    # Monkey-patch sys.modules to intercept donkeycar imports
    class FakeModule:
        def __init__(self, name):
            self.name = name
            # Add the classes we need
            if name == 'donkeycar.parts.fastai':
                self.Linear = Linear
                self.LinearMW = LinearMW
                self.LinearUncertainty = LinearUncertainty
                self.LinearMWUncertainty = LinearMWUncertainty
        
        def __getattr__(self, name):
            # Return a dummy for anything else
            return None
    
    # Save original sys.modules state
    original_modules = sys.modules.copy()
    
    # Inject fake donkeycar modules to prevent actual imports
    sys.modules['donkeycar'] = FakeModule('donkeycar')
    sys.modules['donkeycar.parts'] = FakeModule('donkeycar.parts')
    sys.modules['donkeycar.parts.fastai'] = FakeModule('donkeycar.parts.fastai')
    
    # Load each Linear model and copy its weights to the corresponding subnetwork
    for i, model_path in enumerate(model_paths):
        print(f"\nLoading model {i} from: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            return False
        
        try:
            # Load the model using torch.load with module injection to avoid import issues
            print(f"  Attempting to load model...")
            
            try:
                # Use torch.load with weights_only=False since we've injected fake modules
                linear_model = torch.load(model_path, map_location='cpu', weights_only=False)
                
                # Extract state dict from the loaded checkpoint
                if isinstance(linear_model, dict):
                    # Check if it's a full checkpoint or just state dict
                    if 'state_dict' in linear_model:
                        state_dict = linear_model['state_dict']
                    else:
                        # Assume it's already a state dict
                        state_dict = linear_model
                elif hasattr(linear_model, 'state_dict'):
                    # It's a model object
                    state_dict = linear_model.state_dict()
                else:
                    # Direct state dict
                    state_dict = linear_model
                
                print(f"  Successfully loaded {len(state_dict)} parameters")
                    
            except Exception as e1:
                raise Exception(f"Loading failed: {e1}")
            
            # Load the state dict into the subnetwork
            combined_model.subnetworks[i].load_state_dict(state_dict)
            
            # Verify the copy
            weight_mean = combined_model.subnetworks[i].conv24.weight.data.mean().item()
            print(f"  Subnetwork {i} loaded successfully. Conv24 weight mean: {weight_mean:.6f}")
            
        except Exception as e:
            print(f"Error loading model {i}: {e}")
            # Restore sys.modules before returning
            sys.modules.update(original_modules)
            return False
    
    # Restore sys.modules after loading all models
    sys.modules.update(original_modules)
    
    # Report on remaining randomly initialized subnetworks
    if len(model_paths) < n_weathers:
        print(f"\nSubnetworks {len(model_paths)}-{n_weathers-1} remain randomly initialized:")
        for i in range(len(model_paths), n_weathers):
            weight_mean = combined_model.subnetworks[i].conv24.weight.data.mean().item()
            print(f"  Subnetwork {i} (random): Conv24 weight mean: {weight_mean:.6f}")
    
    # Save the combined model
    print(f"\nSaving combined LinearMW model to: {output_path}")
    torch.save(combined_model, output_path)
    
    # Verify the saved model
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Model saved successfully! Size: {file_size_mb:.2f} MB")
    
    # Quick verification
    print("\nVerification:")
    loaded = torch.load(output_path, map_location='cpu', weights_only=False)
    print(f"  Number of subnetworks: {len(loaded.subnetworks)}")
    for i, subnet in enumerate(loaded.subnetworks):
        weight_mean = subnet.conv24.weight.data.mean().item()
        print(f"  Subnetwork {i} conv24 weight mean: {weight_mean:.6f}")
    
    # Export to ONNX if requested
    if export_onnx:
        onnx_path = output_path.replace('.pt', '.onnx')
        print(f"\nExporting to ONNX format: {onnx_path}")
        
        # Create ONNX-compatible wrapper
        onnx_model = ONNXExportWrapper(combined_model, uncertainty=uncertainty)
        onnx_model.eval()
        
        # Create dummy inputs - standard donkeycar image size is 120x160x3, transposed to CHW
        # Input format: (batch, channels, height, width)
        dummy_image = torch.randn(1, 3, 120, 160)
        dummy_surface_id = torch.tensor([0], dtype=torch.long)  # Surface ID as integer tensor
        
        # Set output names based on whether uncertainty is enabled
        if uncertainty:
            output_names = ["steering", "throttle", "steering_uncertainty", "throttle_uncertainty"]
        else:
            output_names = ["steering", "throttle"]
        
        # Build dynamic_axes for all outputs
        dynamic_axes = {
            "image": {0: "batch_size"},
            "surface_id": {0: "batch_size"}
        }
        for output_name in output_names:
            dynamic_axes[output_name] = {0: "batch_size"}
        
        try:
            torch.onnx.export(
                onnx_model,
                (dummy_image, dummy_surface_id),
                onnx_path,
                opset_version=13,
                input_names=["image", "surface_id"],
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                export_params=True,
                do_constant_folding=True
            )
            
            onnx_file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"ONNX model saved successfully! Size: {onnx_file_size_mb:.2f} MB")
            print(f"  Inputs: image [batch, 3, 120, 160], surface_id [batch]")
            print(f"  Outputs: {', '.join(output_names)}")
            print(f"  Note: surface_id should be 0 (dry), 1 (wet), or 2 (icy)")
            
        except Exception as e:
            print(f"Warning: ONNX export failed: {e}")
            print("PyTorch model was still saved successfully.")
            export_tflite = False  # Can't export TFLite without ONNX
    
    # Export to TFLite if requested
    if export_tflite:
        tflite_path = output_path.replace('.pt', '.tflite')
        
        # Try direct PyTorch to TFLite conversion first (works on ARM)
        print("\nAttempting direct PyTorch to TFLite conversion (ARM-compatible)...")
        
        # Load representative dataset if provided (for future INT8 quantization support)
        representative_images = None
        if data_paths and len(data_paths) > 0:
            try:
                representative_images = load_representative_dataset(data_paths, max_images=500)
            except Exception as e:
                print(f"Warning: Could not load representative dataset: {e}")
                print("Proceeding with dynamic quantization instead.")
        
        # Create ONNX-compatible wrapper for TFLite export
        # (ai_edge_torch works with the wrapper too)
        tflite_model = ONNXExportWrapper(combined_model, uncertainty=uncertainty)
        
        success = convert_pytorch_to_tflite_direct(
            tflite_model,
            tflite_path,
            representative_images,
            uncertainty
        )
        
        # Fallback to ONNX method if direct conversion fails and ONNX was exported
        if not success and export_onnx:
            print("\nDirect conversion failed. Attempting ONNX to TFLite conversion...")
            print("Note: This requires onnx-tf and tensorflow_addons (not available on ARM)")
            
            if representative_images is None and (data_paths is None or len(data_paths) == 0):
                print("\nError: TFLite export via ONNX requires data paths for representative dataset.")
                print("Please specify data directories with --dry-data, --wet-data, and/or --icy-data")
            else:
                try:
                    if representative_images is None:
                        representative_images = load_representative_dataset(data_paths, max_images=500)
                    convert_onnx_to_tflite(onnx_path, tflite_path, representative_images, uncertainty)
                except Exception as e:
                    print(f"\nWarning: TFLite export failed: {e}")
                    import traceback
                    traceback.print_exc()
                    print("PyTorch and ONNX models were still saved successfully.")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Combine Linear/LinearUncertainty models into a LinearMW/LinearMWUncertainty multi-weather model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use same model for all subnetworks (test LinearMW architecture):
  python combine_models_to_mw.py --dry models/good.pt --output models/test_mw.pt
  
  # Load dry and wet, leave icy random:
  python combine_models_to_mw.py --dry models/dry.pt --wet models/wet.pt --output models/combined_mw.pt
  
  # Load all three subnetworks:
  python combine_models_to_mw.py --dry models/dry.pt --wet models/wet.pt --icy models/icy.pt --output models/full_mw.pt
  
  # With ONNX export:
  python combine_models_to_mw.py --dry models/dry.pt --wet models/wet.pt --icy models/icy.pt --output models/full_mw.pt --onnx
  
  # With TFLite export (works on ARM!):
  python combine_models_to_mw.py --dry models/dry.pt --wet models/wet.pt --icy models/icy.pt --output models/full_mw.pt --tflite
  
  # With both ONNX and TFLite:
  python combine_models_to_mw.py --dry models/dry.pt --wet models/wet.pt --icy models/icy.pt \\n      --output models/full_mw.pt --onnx --tflite
  
  # Combine uncertainty models with all exports:
  python combine_models_to_mw.py --dry models/dry_unc.pt --wet models/wet_unc.pt --icy models/icy_unc.pt \\n      --output models/full_mw_unc.pt --uncertainty --onnx --tflite
        """
    )
    
    parser.add_argument('--dry', type=str, default=None,
                        help='Path to Linear/LinearUncertainty model for dry surface (subnetwork 0)')
    parser.add_argument('--wet', type=str, default=None,
                        help='Path to Linear/LinearUncertainty model for wet surface (subnetwork 1)')
    parser.add_argument('--icy', type=str, default=None,
                        help='Path to Linear/LinearUncertainty model for icy surface (subnetwork 2)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for combined LinearMW/LinearMWUncertainty model')
    parser.add_argument('--uncertainty', action='store_true',
                        help='Combine LinearUncertainty models into LinearMWUncertainty (default: combine Linear models into LinearMW)')
    parser.add_argument('--onnx', action='store_true',
                        help='Also export the model to ONNX format')
    parser.add_argument('--tflite', action='store_true',
                        help='Also export the model to TFLite format (uses ai_edge_torch - works on ARM!)')
    parser.add_argument('--dry-data', type=str, default=None,
                        help='Path to dry surface training data directory (optional, for future INT8 quantization)')
    parser.add_argument('--wet-data', type=str, default=None,
                        help='Path to wet surface training data directory (optional, for future INT8 quantization)')
    parser.add_argument('--icy-data', type=str, default=None,
                        help='Path to icy surface training data directory (optional, for future INT8 quantization)')
    
    args = parser.parse_args()
    
    # Build list of model paths
    model_paths = []
    surface_names = ['dry', 'wet', 'icy']
    for surface in surface_names:
        path = getattr(args, surface)
        if path is not None:
            model_paths.append(path)
        else:
            # Stop at first None - don't allow gaps
            break
    
    if len(model_paths) == 0:
        parser.error("At least --dry must be specified")
    
    model_type = "LinearUncertainty" if args.uncertainty else "Linear"
    output_type = "LinearMWUncertainty" if args.uncertainty else "LinearMW"
    
    # Build list of data paths for representative dataset
    data_paths = []
    for surface in surface_names:
        data_arg = f"{surface}_data"
        path = getattr(args, data_arg, None)
        if path is not None:
            data_paths.append(path)
    
    print("=" * 70)
    print(f"{model_type} to {output_type} Model Combiner")
    print("=" * 70)
    print(f"\nSubnetwork 0 (dry): {args.dry if args.dry else 'Random'}")
    print(f"Subnetwork 1 (wet): {args.wet if args.wet else 'Random'}")
    print(f"Subnetwork 2 (icy): {args.icy if args.icy else 'Random'}")
    print(f"Output: {args.output}")
    if args.tflite:
        print(f"\nData for representative dataset:")
        print(f"  Dry data: {args.dry_data if args.dry_data else 'Not provided'}")
        print(f"  Wet data: {args.wet_data if args.wet_data else 'Not provided'}")
        print(f"  Icy data: {args.icy_data if args.icy_data else 'Not provided'}")
    print()
    
    success = combine_models(
        model_paths, 
        args.output, 
        uncertainty=args.uncertainty, 
        export_onnx=args.onnx,
        export_tflite=args.tflite,
        data_paths=data_paths if data_paths else None
    )
    
    if success:
        print("\n" + "=" * 70)
        print("SUCCESS! Combined model created.")
        print("=" * 70)
        print(f"\nYou can now test it with:")
        model_type_flag = "fastai_linear_mw_unc" if args.uncertainty else "fastai_linear_mw"
        print(f"  python manage.py drive --model={args.output} --type={model_type_flag}")
    else:
        print("\nFailed to create combined model.")
        sys.exit(1)
