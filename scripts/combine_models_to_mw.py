#!/usr/bin/env python3
"""
Script to combine 3 Linear models into a LinearMW model.
This allows testing LinearMW inference with known-good weights.

Also supports combining LinearUncertainty models into LinearMWUncertainty models.
Can export to ONNX format.

Requirements for ONNX export:
    pip install onnx

For TFLite conversion:
    Use the separate convert_onnx_to_tflite.py script on an x86_64 machine:
        python convert_onnx_to_tflite.py --input model.onnx --output model.tflite \\
            --data-dry data_dry --data-wet data_wet --data-icy data_icy

Usage:
    # Basic PyTorch model combination
    python combine_models_to_mw.py --dry model.pt --output output_mw.pt
    
    # With ONNX export
    python combine_models_to_mw.py --dry dry.pt --wet wet.pt --icy icy.pt --output mw.pt --onnx
    
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

def combine_models(model_paths, output_path, n_weathers=3, uncertainty=False, export_onnx=False):
    """
    Combine individual Linear/LinearUncertainty models into a LinearMW/LinearMWUncertainty model.
    
    Args:
        model_paths: List of paths to Linear/LinearUncertainty model files (can be less than n_weathers)
        output_path: Path to save the combined LinearMW/LinearMWUncertainty model
        n_weathers: Number of subnetworks (default 3)
        uncertainty: If True, combine LinearUncertainty models into LinearMWUncertainty
        export_onnx: If True, also export model to ONNX format
    """
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
  
  # Combine uncertainty models with ONNX export:
  python combine_models_to_mw.py --dry models/dry_unc.pt --wet models/wet_unc.pt --icy models/icy_unc.pt \\n      --output models/full_mw_unc.pt --uncertainty --onnx
  
  # For TFLite conversion, use the separate convert_onnx_to_tflite.py script:
  python convert_onnx_to_tflite.py --input models/full_mw.onnx --output models/full_mw.tflite \\n      --data-dry ../macchinina/data_dry --data-wet ../macchinina/data_wet --data-icy ../macchinina/data_icy
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
    
    print("=" * 70)
    print(f"{model_type} to {output_type} Model Combiner")
    print("=" * 70)
    print(f"\nSubnetwork 0 (dry): {args.dry if args.dry else 'Random'}")
    print(f"Subnetwork 1 (wet): {args.wet if args.wet else 'Random'}")
    print(f"Subnetwork 2 (icy): {args.icy if args.icy else 'Random'}")
    print(f"Output: {args.output}")

    print()
    
    success = combine_models(
        model_paths, 
        args.output, 
        uncertainty=args.uncertainty, 
        export_onnx=args.onnx
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
