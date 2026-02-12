#!/usr/bin/env python3
"""
Script to combine 3 Linear models into a LinearMW model.
This allows testing LinearMW inference with known-good weights.

Usage:
    python combine_models_to_mw.py --dry model.pt --output output_mw.pt
    python combine_models_to_mw.py --dry dry.pt --wet wet.pt --icy icy.pt --output output_mw.pt
"""
import sys
import torch
import os
import argparse
import torch.nn as nn
import pickle
import io

# Recreate the Linear and LinearMW classes locally to avoid donkeycar import issues

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

def combine_models(model_paths, output_path, n_weathers=3):
    """
    Combine individual Linear models into a LinearMW model.
    
    Args:
        model_paths: List of paths to Linear model files (can be less than n_weathers)
        output_path: Path to save the combined LinearMW model
        n_weathers: Number of subnetworks (default 3)
    """
    if len(model_paths) > n_weathers:
        print(f"Error: Too many model paths ({len(model_paths)}) for {n_weathers} subnetworks")
        return False
    
    print(f"Creating LinearMW with {n_weathers} subnetworks...")
    combined_model = LinearMW(n_weathers=n_weathers)
    
    if len(model_paths) < n_weathers:
        print(f"Note: Only {len(model_paths)} models provided. Subnetworks {len(model_paths)}-{n_weathers-1} will be randomly initialized.")
    
    # Custom unpickler to redirect donkeycar.parts.fastai.Linear to our local Linear
    class RemappingUnpickler(pickle.Unpickler):           
        def find_class(self, module, name):
            # Redirect donkeycar classes to our local versions
            if module == 'donkeycar.parts.fastai':
                if name == 'Linear':
                    return Linear
                elif name == 'LinearMW':
                    return LinearMW
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
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Combine Linear models into a LinearMW multi-weather model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use same model for all subnetworks (test LinearMW architecture):
  python combine_models_to_mw.py --dry models/good.pt --output models/test_mw.pt
  
  # Load dry and wet, leave icy random:
  python combine_models_to_mw.py --dry models/dry.pt --wet models/wet.pt --output models/combined_mw.pt
  
  # Load all three subnetworks:
  python combine_models_to_mw.py --dry models/dry.pt --wet models/wet.pt --icy models/icy.pt --output models/full_mw.pt
        """
    )
    
    parser.add_argument('--dry', type=str, default=None,
                        help='Path to Linear model for dry surface (subnetwork 0)')
    parser.add_argument('--wet', type=str, default=None,
                        help='Path to Linear model for wet surface (subnetwork 1)')
    parser.add_argument('--icy', type=str, default=None,
                        help='Path to Linear model for icy surface (subnetwork 2)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for combined LinearMW model')
    
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
    
    print("=" * 70)
    print("Linear to LinearMW Model Combiner")
    print("=" * 70)
    print(f"\nSubnetwork 0 (dry): {args.dry if args.dry else 'Random'}")
    print(f"Subnetwork 1 (wet): {args.wet if args.wet else 'Random'}")
    print(f"Subnetwork 2 (icy): {args.icy if args.icy else 'Random'}")
    print(f"Output: {args.output}\n")
    
    success = combine_models(model_paths, args.output)
    
    if success:
        print("\n" + "=" * 70)
        print("SUCCESS! Combined model created.")
        print("=" * 70)
        print(f"\nYou can now test it with:")
        print(f"  python manage.py drive --model={args.output} --type=fastai_linear_mw")
    else:
        print("\nFailed to create combined model.")
        sys.exit(1)
