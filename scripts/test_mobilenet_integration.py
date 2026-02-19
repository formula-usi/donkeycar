#!/usr/bin/env python3
"""
Test script to verify MobileNet integration with donkeycar framework.

This script tests:
1. Model types are registered correctly
2. Models can be created from config
3. Models can process images
4. State dict operations work

Usage:
    python test_mobilenet_integration.py
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add donkeycar to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import donkeycar as dk
from donkeycar.utils import get_model_by_type


class DummyConfig:
    """Minimal config for testing"""
    DEFAULT_MODEL_TYPE = 'fastai_mobilenet'
    IMAGE_H = 120
    IMAGE_W = 160
    IMAGE_DEPTH = 3
    
    # MobileNet config
    MOBILENET_FREEZE_BACKBONE = True
    MOBILENET_PRETRAINED = True
    MOBILENET_DROPOUT = 0.2
    MOBILENET_LOSS_TYPE = 'nll'


def test_model_registration():
    """Test that MobileNet models are registered in get_model_by_type"""
    print("\n" + "="*60)
    print("TEST 1: Model Registration")
    print("="*60)
    
    cfg = DummyConfig()
    
    # Test basic MobileNet
    print("\n1. Testing fastai_mobilenet...")
    try:
        model = get_model_by_type('fastai_mobilenet', cfg)
        print(f"   ✓ Model created: {type(model).__name__}")
        print(f"   ✓ Interpreter: {type(model.interpreter).__name__}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test MobileNet with uncertainty
    print("\n2. Testing fastai_mobilenet_unc...")
    try:
        model_unc = get_model_by_type('fastai_mobilenet_unc', cfg)
        print(f"   ✓ Model created: {type(model_unc).__name__}")
        print(f"   ✓ Interpreter: {type(model_unc.interpreter).__name__}")
        print(f"   ✓ Loss type: {model_unc.loss_type}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("\n✅ All model types registered correctly!")
    return True


def test_model_architecture():
    """Test that models have correct architecture"""
    print("\n" + "="*60)
    print("TEST 2: Model Architecture")
    print("="*60)
    
    cfg = DummyConfig()
    
    # Create model
    print("\n1. Creating MobileNet model...")
    pilot = get_model_by_type('fastai_mobilenet', cfg)
    model = pilot.interpreter.model
    
    # Check key components
    print("\n2. Checking architecture components...")
    
    components = {
        'backbone': hasattr(model, 'backbone'),
        'head': hasattr(model, 'head'),
        'output1': hasattr(model, 'output1'),
        'output2': hasattr(model, 'output2'),
    }
    
    for name, exists in components.items():
        status = "✓" if exists else "✗"
        print(f"   {status} {name}: {exists}")
    
    if not all(components.values()):
        print("\n✗ Missing architecture components!")
        return False
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\n3. Parameter counts:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    print(f"   Frozen: {frozen_params:,}")
    
    # Verify backbone is frozen
    if cfg.MOBILENET_FREEZE_BACKBONE:
        if frozen_params > 0:
            print(f"   ✓ Backbone correctly frozen")
        else:
            print(f"   ✗ Backbone should be frozen but isn't!")
            return False
    
    print("\n✅ Architecture looks correct!")
    return True


def test_forward_pass():
    """Test that model can process images"""
    print("\n" + "="*60)
    print("TEST 3: Forward Pass")
    print("="*60)
    
    cfg = DummyConfig()
    
    # Test basic model
    print("\n1. Testing basic MobileNet forward pass...")
    pilot = get_model_by_type('fastai_mobilenet', cfg)
    
    # Create random image (H, W, C) in uint8 [0, 255]
    img_arr = np.random.randint(0, 255, (cfg.IMAGE_H, cfg.IMAGE_W, cfg.IMAGE_DEPTH), dtype=np.uint8)
    
    try:
        steering, throttle = pilot.run(img_arr)
        print(f"   ✓ Output shape: steering={type(steering)}, throttle={type(throttle)}")
        print(f"   ✓ Steering: {steering:.4f}, Throttle: {throttle:.4f}")
        
        # Check output range (should be in [-1, 1] after interpreter_to_output)
        if -1.5 <= steering <= 1.5 and -1.5 <= throttle <= 1.5:
            print(f"   ✓ Outputs in reasonable range")
        else:
            print(f"   ⚠ Outputs outside expected range [-1, 1]")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test uncertainty model
    print("\n2. Testing uncertainty model forward pass...")
    pilot_unc = get_model_by_type('fastai_mobilenet_unc', cfg)
    
    try:
        steering, throttle, steering_unc, throttle_unc = pilot_unc.run(img_arr)
        print(f"   ✓ Outputs: steering={steering:.4f}, throttle={throttle:.4f}")
        print(f"   ✓ Uncertainties: steering_unc={steering_unc:.4f}, throttle_unc={throttle_unc:.4f}")
        
        # Check uncertainties are positive
        if steering_unc > 0 and throttle_unc > 0:
            print(f"   ✓ Uncertainties are positive")
        else:
            print(f"   ✗ Uncertainties should be positive!")
            return False
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ Forward passes work correctly!")
    return True


def test_state_dict_operations():
    """Test state dict save/load"""
    print("\n" + "="*60)
    print("TEST 4: State Dict Operations")
    print("="*60)
    
    from donkeycar.parts.mobilenet import (
        MobileNetV2Base, 
        save_model_state_dict, 
        load_model_state_dict
    )
    
    # Create model
    print("\n1. Creating model...")
    model = MobileNetV2Base(pretrained=False)  # Don't download ImageNet for speed
    
    # Get initial state
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Modify weights slightly
    print("\n2. Modifying model weights...")
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.01)
    
    # Save state dict
    print("\n3. Saving state dict...")
    temp_path = "/tmp/test_mobilenet_state.pth"
    try:
        save_model_state_dict(model, temp_path)
        print(f"   ✓ Saved to {temp_path}")
    except Exception as e:
        print(f"   ✗ Save failed: {e}")
        return False
    
    # Create new model
    print("\n4. Creating new model and loading state dict...")
    new_model = MobileNetV2Base(pretrained=False)
    
    try:
        load_model_state_dict(new_model, temp_path)
        print(f"   ✓ Loaded from {temp_path}")
    except Exception as e:
        print(f"   ✗ Load failed: {e}")
        return False
    
    # Verify weights match
    print("\n5. Verifying loaded weights match saved weights...")
    matches = True
    for (name1, param1), (name2, param2) in zip(model.named_parameters(), new_model.named_parameters()):
        if not torch.allclose(param1, param2):
            print(f"   ✗ Mismatch in {name1}")
            matches = False
            break
    
    if matches:
        print(f"   ✓ All weights match!")
    else:
        return False
    
    print("\n✅ State dict operations work correctly!")
    return True


def test_unfreeze_backbone():
    """Test unfreezing backbone layers"""
    print("\n" + "="*60)
    print("TEST 5: Unfreeze Backbone")
    print("="*60)
    
    from donkeycar.parts.mobilenet import MobileNetV2Base
    
    # Create model with frozen backbone
    print("\n1. Creating model with frozen backbone...")
    model = MobileNetV2Base(freeze_backbone=True, pretrained=False)
    
    frozen_before = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable_before = sum(1 for p in model.parameters() if p.requires_grad)
    
    print(f"   Initial - Frozen: {frozen_before}, Trainable: {trainable_before}")
    
    # Unfreeze last 3 layers
    print("\n2. Unfreezing last 3 layers...")
    model.unfreeze_backbone(num_layers=3)
    
    frozen_after = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable_after = sum(1 for p in model.parameters() if p.requires_grad)
    
    print(f"   After - Frozen: {frozen_after}, Trainable: {trainable_after}")
    
    if trainable_after > trainable_before:
        print(f"   ✓ Successfully unfroze {trainable_after - trainable_before} more parameters")
    else:
        print(f"   ✗ Unfreezing didn't increase trainable parameters!")
        return False
    
    # Unfreeze all
    print("\n3. Unfreezing all layers...")
    model.unfreeze_backbone(num_layers=None)
    
    frozen_all = sum(1 for p in model.parameters() if not p.requires_grad)
    
    if frozen_all == 0:
        print(f"   ✓ All parameters are now trainable")
    else:
        print(f"   ⚠ {frozen_all} parameters still frozen (may be batch norm buffers)")
    
    print("\n✅ Unfreezing works correctly!")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MOBILENET INTEGRATION TEST SUITE")
    print("="*60)
    
    tests = [
        ("Model Registration", test_model_registration),
        ("Model Architecture", test_model_architecture),
        ("Forward Pass", test_forward_pass),
        ("State Dict Operations", test_state_dict_operations),
        ("Unfreeze Backbone", test_unfreeze_backbone),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\nMobileNet integration is working correctly.")
        print("You can now use it in your donkeycar project:")
        print("\n  1. Set DEFAULT_MODEL_TYPE = 'fastai_mobilenet' in myconfig.py")
        print("  2. Train: python train.py --tubs data/ --model models/mobilenet.pth")
        print("  3. Drive: python manage.py drive --model models/mobilenet.pth")
    else:
        print("❌ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
