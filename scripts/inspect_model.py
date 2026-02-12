#!/usr/bin/env python3
"""
Script to inspect a saved LinearMW model - avoids donkeycar imports
"""
import sys
import pickle
import io

if len(sys.argv) < 2:
    print("Usage: python inspect_model.py <model_path>")
    sys.exit(1)

model_path = sys.argv[1]
print(f"Loading model from: {model_path}")

# Custom unpickler that doesn't need class definitions
class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Allow torch and basic types only
        if module.startswith('torch') or module == 'collections' or module == 'builtins':
            return super().find_class(module, name)
        # For other modules, return a dummy class
        return type(name, (), {})

try:
    import torch
    
    # Try loading with torch first
    try:
        with open(model_path, 'rb') as f:
            # Peek at the data to check structure
            data = torch.load(f, map_location='cpu', weights_only=False)
            
        print(f"\nModel type: {type(data).__name__}")
        
        if hasattr(data, 'subnetworks'):
            print(f"Number of subnetworks: {len(data.subnetworks)}")
            
            for i, subnet in enumerate(data.subnetworks):
                total_params = sum(p.numel() for p in subnet.parameters())
                mean_weight = sum(p.abs().mean().item() for p in subnet.parameters() if p.requires_grad) / sum(1 for p in subnet.parameters() if p.requires_grad)
                print(f"  Subnetwork {i}: {total_params:,} params, mean abs weight: {mean_weight:.6f}")
            
            print("\nFirst conv layer weight statistics:")
            for i, subnet in enumerate(data.subnetworks):
                weight_mean = subnet.conv24.weight.data.mean().item()
                weight_std = subnet.conv24.weight.data.std().item()
                print(f"  Subnetwork {i}: mean={weight_mean:.6f}, std={weight_std:.6f}")
        else:
            print("Not a multi-weather model!")
            total_params = sum(p.numel() for p in data.parameters())
            print(f"Total parameters: {total_params:,}")
            
    except Exception as e1:
        print(f"Standard loading failed: {e1}")
        print("\nTrying alternative method...")
        # Alternative: just check file size as rough estimate
        import os
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model file size: {size_mb:.2f} MB")
        if size_mb > 6:  # Rough estimate: 3 subnetworks > 6MB
            print("Size suggests this is a multi-weather model (3 subnetworks)")
        else:
            print("Size suggests this is a single network model")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
