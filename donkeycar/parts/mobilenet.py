"""
MobileNet-based FastAI models for donkeycar.

This module demonstrates:
1. Using pretrained MobileNetV2 from torchvision
2. Proper state_dict management for saving/loading
3. Transfer learning with frozen/unfrozen layers
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Tuple, Union, List
from logging import getLogger

import torchvision.models as models
from fastai.vision.all import *

from donkeycar.parts.fastai import FastAiPilot, FastAILinear
from donkeycar.parts.interpreter import FastAIInterpreter, Interpreter
from donkeycar.parts.pytorch.torch_data import get_default_transform

logger = getLogger(__name__)


class MobileNetV2Base(nn.Module):
    """
    MobileNetV2 base model pretrained on ImageNet. ADAPTED FOR DONKEYCAR.
    
    IMPORTANT: This model handles the input size mismatch between ImageNet (224x224)
    and donkeycar (120x160) by using adaptive pooling. However, for best results,
    consider resizing inputs to 224x224 or training without pretrained weights.
    
    This model:
    - Loads pretrained MobileNetV2 weights from torchvision
    - Replaces the classifier with custom head for steering/throttle
    - Supports freezing backbone for transfer learning
    - Uses adaptive pooling to handle variable input sizes
    """
    
    def __init__(self, num_outputs: int = 2, freeze_backbone: bool = True, 
                 dropout: float = 0.2, pretrained: bool = True,
                 use_imagenet_normalization: bool = True):
        """
        Args:
            num_outputs: Number of output values (2 for angle/throttle, 4 for uncertainty)
            freeze_backbone: If True, freeze MobileNet feature extractor
            dropout: Dropout rate for classifier head
            pretrained: If True, load ImageNet pretrained weights
            use_imagenet_normalization: If True, expect ImageNet-normalized inputs.
                                       If False, expect [0,1] normalized inputs.
        """
        super().__init__()
        
        self.use_imagenet_normalization = use_imagenet_normalization
        
        # Load pretrained MobileNetV2
        # Note: newer torchvision uses 'weights' instead of 'pretrained'
        try:
            from torchvision.models import MobileNet_V2_Weights
            weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.mobilenet_v2(weights=weights)
            logger.info(f"Loaded MobileNetV2 with weights={weights}")
        except ImportError:
            # Fallback for older torchvision
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            logger.info(f"Loaded MobileNetV2 with pretrained={pretrained}")
        
        # WARNING: If using pretrained weights, they expect 224x224 inputs!
        if pretrained:
            logger.warning(
                "Using ImageNet pretrained weights with donkeycar's 120x160 images. "
                "This may reduce transfer learning effectiveness. "
                "Consider: (1) resizing to 224x224, or (2) training without pretrained weights."
            )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Froze MobileNetV2 backbone weights")
        
        # Get the number of features from the last layer
        # MobileNetV2 has 1280 features before the classifier
        num_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier with our custom head
        # Original classifier: Sequential(Dropout(0.2), Linear(1280, 1000))
        self.backbone.classifier = nn.Identity()  # Remove original classifier
        
        # Add adaptive pooling to handle variable input sizes
        # This ensures we always get 1280-dimensional features regardless of input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Custom head for donkeycar
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 50),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        
        # Output layers for angle and throttle
        self.output1 = nn.Linear(50, 1)  # angle
        self.output2 = nn.Linear(50, 1)  # throttle
        
        # Initialize the custom layers
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights for custom layers"""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize output layers to predict center values (0.5)
        nn.init.normal_(self.output1.weight, 0.0, 0.01)
        nn.init.constant_(self.output1.bias, 0.5)
        nn.init.normal_(self.output2.weight, 0.0, 0.01)
        nn.init.constant_(self.output2.bias, 0.5)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image tensor [batch, 3, H, W]
               Should be in [0, 1] range if use_imagenet_normalization=False
               Should be ImageNet normalized if use_imagenet_normalization=True
            
        Returns:
            Tensor [batch, 2] with [angle, throttle]
        """
        # Extract features using MobileNetV2 backbone features only (not classifier)
        # MobileNetV2 features output shape: [batch, 1280, H/32, W/32]
        # For 120x160 input: [batch, 1280, 3, 5]
        # For 224x224 input: [batch, 1280, 7, 7]
        x = self.backbone.features(x)
        
        # Apply adaptive pooling to get fixed size regardless of input dimensions
        # Output: [batch, 1280, 1, 1]
        x = self.adaptive_pool(x)
        
        # Flatten to [batch, 1280]
        x = torch.flatten(x, 1)
        
        # Pass through custom head
        x = self.head(x)
        
        # Get angle and throttle predictions
        angle = self.output1(x)
        throttle = self.output2(x)
        
        return torch.cat((angle, throttle), 1)
    
    def unfreeze_backbone(self, num_layers: int = None):
        """
        Unfreeze the backbone for fine-tuning.
        
        Args:
            num_layers: If specified, only unfreeze the last N layers.
                       If None, unfreeze all layers.
        """
        if num_layers is None:
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("Unfroze all MobileNetV2 backbone layers")
        else:
            # Unfreeze last N layers
            # MobileNetV2 has features grouped in InvertedResidual blocks
            layers = list(self.backbone.features.children())
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            logger.info(f"Unfroze last {num_layers} MobileNetV2 layers")


class MobileNetV2Uncertainty(MobileNetV2Base):
    """
    MobileNetV2 model with uncertainty estimation.
    
    Outputs: [angle_mean, throttle_mean, angle_log_var, throttle_log_var]
    """
    
    def __init__(self, freeze_backbone: bool = True, dropout: float = 0.2, 
                 pretrained: bool = True, use_imagenet_normalization: bool = True):
        super().__init__(num_outputs=4, freeze_backbone=freeze_backbone, 
                        dropout=dropout, pretrained=pretrained,
                        use_imagenet_normalization=use_imagenet_normalization)
        
        # Additional uncertainty heads
        self.output1_uncertainty = nn.Linear(50 + 1, 1)  # angle log-variance
        self.output2_uncertainty = nn.Linear(50 + 1, 1)  # throttle log-variance
        
        # Initialize uncertainty heads
        with torch.no_grad():
            self.output1_uncertainty.weight.normal_(0.0, 0.001)
            self.output1_uncertainty.bias.fill_(-1.0)
            self.output2_uncertainty.weight.normal_(0.0, 0.001)
            self.output2_uncertainty.bias.fill_(-1.0)
    
    def forward(self, x):
        """
        Forward pass with uncertainty
        
        Returns:
            Tensor [batch, 4] with [angle, throttle, angle_log_var, throttle_log_var]
        """
        # Extract features using backbone
        x = self.backbone.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        # Pass through custom head
        x = self.head(x)
        
        # Get predictions
        angle = self.output1(x)
        throttle = self.output2(x)
        
        # Get uncertainties (concatenate with predictions for conditioning)
        angle_uncertainty = self.output1_uncertainty(torch.cat([x, angle], dim=1))
        throttle_uncertainty = self.output2_uncertainty(torch.cat([x, throttle], dim=1))
        
        # Clamp outputs to prevent extreme values
        angle = torch.clamp(angle, -0.5, 1.5)
        throttle = torch.clamp(throttle, -0.5, 1.5)
        angle_uncertainty = torch.clamp(angle_uncertainty, -10, 2)
        throttle_uncertainty = torch.clamp(throttle_uncertainty, -10, 2)
        
        return torch.cat((angle, throttle, angle_uncertainty, throttle_uncertainty), 1)


class FastAIMobileNet(FastAILinear):
    """
    FastAI pilot using MobileNetV2 pretrained on ImageNet.
    
    AUTOMATIC RESIZING: This model automatically resizes input images to 224x224
    when using pretrained weights, so you can keep your 120x160 data!
    
    Example usage:
        # No config changes needed! Resizing is automatic.
        pilot = FastAIMobileNet(freeze_backbone=True, pretrained=True)
        pilot.compile()
        
        # Train with frozen backbone first
        history = pilot.train(...)
        
        # Optionally fine-tune by unfreezing backbone
        pilot.interpreter.model.unfreeze_backbone(num_layers=3)
        history = pilot.train(...)
    """
    
    def __init__(self,
                 interpreter: Interpreter = FastAIInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_outputs: int = 2,
                 freeze_backbone: bool = True,
                 dropout: float = 0.2,
                 pretrained: bool = True,
                 use_imagenet_normalization: bool = True,
                 auto_resize_224: bool = True):
        """
        Args:
            freeze_backbone: Freeze MobileNet backbone for transfer learning
            dropout: Dropout rate
            pretrained: Load ImageNet pretrained weights
            use_imagenet_normalization: Use ImageNet mean/std normalization
            auto_resize_224: Automatically resize inputs to 224x224 (recommended for pretrained)
        """
        self.freeze_backbone = freeze_backbone
        self.dropout = dropout
        self.pretrained = pretrained
        self.use_imagenet_normalization = use_imagenet_normalization
        self.auto_resize_224 = auto_resize_224 and pretrained  # Only resize if using pretrained
        
        if self.auto_resize_224:
            logger.info("MobileNet will automatically resize inputs to 224x224 for pretrained weights")
        
        super().__init__(interpreter, input_shape, num_outputs)
        # Ensure model starts in eval mode
        if hasattr(self.interpreter, 'model') and self.interpreter.model is not None:
            self.interpreter.model.eval()
        
    def create_model(self):
        model = MobileNetV2Base(
            num_outputs=self.num_outputs,
            freeze_backbone=self.freeze_backbone,
            dropout=self.dropout,
            pretrained=self.pretrained,
            use_imagenet_normalization=self.use_imagenet_normalization
        )
        # Set to eval mode by default for proper inference
        model.eval()
        return model
    
    def run(self, img_arr: np.ndarray, other_arr: List[float] = None) \
            -> Tuple[Union[float, torch.tensor], ...]:
        """
        Donkeycar parts interface to run the part in the loop.
        Automatically resizes to 224x224 if using pretrained weights.

        :param img_arr:     uint8 [0,255] numpy array with image data
        :param other_arr:   numpy array of additional data (not used)
        :return:            tuple of (angle, throttle)
        """
        from torchvision import transforms
        
        # Create transform with optional resize
        if self.auto_resize_224:
            # Resize to 224x224 for pretrained weights
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            # Use original size
            transform = get_default_transform(resize=False)
        
        norm_arr = transform(img_arr)
        tensor_other_array = torch.FloatTensor(other_arr) if other_arr else None
        return self.inference(norm_arr, tensor_other_array)
    
    def get_train_transform(self):
        """
        Get the transform to use for training.
        This is called by the training pipeline to ensure consistency.
        """
        from torchvision import transforms
        
        if self.auto_resize_224:
            # For pretrained: resize to 224x224
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            # For non-pretrained: use original size
            return get_default_transform(resize=False)
    
    def output_shapes(self):
        """Return the input shape expected by the model"""
        if self.auto_resize_224:
            # When auto-resizing, model expects 224x224
            return (224, 224, 3)
        else:
            # Otherwise use the configured input shape
            return self.get_input_shape('img')[1:]


class FastAIMobileNetUncertainty(FastAILinear):
    """
    FastAI pilot using MobileNetV2 with uncertainty estimation.
    
    Outputs: steering, throttle, angle_std, throttle_std
    
    AUTOMATIC RESIZING: This model automatically resizes input images to 224x224
    when using pretrained weights, so you can keep your 120x160 data!
    """
    
    def __init__(self,
                 interpreter: Interpreter = FastAIInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_outputs: int = 4,
                 freeze_backbone: bool = True,
                 dropout: float = 0.2,
                 pretrained: bool = True,
                 loss_type: str = 'nll',
                 use_imagenet_normalization: bool = True,
                 auto_resize_224: bool = True):
        """
        Args:
            freeze_backbone: Freeze MobileNet backbone for transfer learning
            dropout: Dropout rate
            pretrained: Load ImageNet pretrained weights
            loss_type: 'nll' or 'simple' loss function
            use_imagenet_normalization: Use ImageNet mean/std normalization
            auto_resize_224: Automatically resize inputs to 224x224 (recommended for pretrained)
        """
        self.freeze_backbone = freeze_backbone
        self.dropout = dropout
        self.pretrained = pretrained
        self.loss_type = loss_type
        self.use_imagenet_normalization = use_imagenet_normalization
        self.auto_resize_224 = auto_resize_224 and pretrained  # Only resize if using pretrained
        
        if self.auto_resize_224:
            logger.info("MobileNet Uncertainty will automatically resize inputs to 224x224 for pretrained weights")
        
        # Import loss functions from fastai module
        from donkeycar.parts.fastai import UncertaintyLoss, SimpleUncLoss, NLLLoss
        
        super().__init__(interpreter, input_shape, num_outputs)
        
        # Set appropriate loss function
        if loss_type == 'unc':
            self.loss = UncertaintyLoss()
        elif loss_type == 'nll':
            self.loss = NLLLoss()
        else:
            self.loss = SimpleUncLoss()
        
        # Ensure model starts in eval mode
        if hasattr(self.interpreter, 'model') and self.interpreter.model is not None:
            self.interpreter.model.eval()
    
    def create_model(self):
        model = MobileNetV2Uncertainty(
            freeze_backbone=self.freeze_backbone,
            dropout=self.dropout,
            pretrained=self.pretrained,
            use_imagenet_normalization=self.use_imagenet_normalization
        )
        # Set to eval mode by default for proper inference
        model.eval()
        return model
    
    def run(self, img_arr: np.ndarray, other_arr: List[float] = None) \
            -> Tuple[Union[float, torch.tensor], ...]:
        """
        Donkeycar parts interface to run the part in the loop.
        Automatically resizes to 224x224 if using pretrained weights.

        :param img_arr:     uint8 [0,255] numpy array with image data
        :param other_arr:   numpy array of additional data (not used)
        :return:            tuple of (angle, throttle, angle_std, throttle_std)
        """
        from torchvision import transforms
        
        # Create transform with optional resize
        if self.auto_resize_224:
            # Resize to 224x224 for pretrained weights
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            # Use original size
            transform = get_default_transform(resize=False)
        
        norm_arr = transform(img_arr)
        tensor_other_array = torch.FloatTensor(other_arr) if other_arr else None
        return self.inference(norm_arr, tensor_other_array)
    
    def get_train_transform(self):
        """
        Get the transform to use for training.
        This is called by the training pipeline to ensure consistency.
        """
        from torchvision import transforms
        
        if self.auto_resize_224:
            # For pretrained: resize to 224x224
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            # For non-pretrained: use original size
            return get_default_transform(resize=False)
    
    def output_shapes(self):
        """Return the input shape expected by the model"""
        if self.auto_resize_224:
            # When auto-resizing, model expects 224x224
            return (224, 224, 3)
        else:
            # Otherwise use the configured input shape
            return self.get_input_shape('img')[1:]
    
    def interpreter_to_output(self, interpreter_out):
        """
        Convert raw model outputs to final outputs with uncertainties.
        
        The model outputs [angle, throttle, angle_log_var, throttle_log_var] in [0, 1] range.
        We need to:
        1. Scale angle and throttle from [0, 1] to [-1, 1]
        2. Convert log-variance to standard deviation
        
        Args:
            interpreter_out: Raw model output [angle, throttle, angle_log_var, throttle_log_var]
            
        Returns:
            (angle_mean, throttle_mean, angle_std, throttle_std)
        """
        # Scale outputs from [0, 1] to [-1, 1] for angle and throttle
        angle_mean = (interpreter_out[0] * 2) - 1
        throttle_mean = (interpreter_out[1] * 2) - 1
        
        # Convert log-variance to standard deviation
        angle_log_var = interpreter_out[2]
        throttle_log_var = interpreter_out[3]
        angle_std = np.sqrt(np.exp(angle_log_var))
        throttle_std = np.sqrt(np.exp(throttle_log_var))
        
        return angle_mean, throttle_mean, angle_std, throttle_std


# ==================== STATE DICT UTILITIES ====================

def save_model_state_dict(model: nn.Module, path: str):
    """
    Save only the model's state_dict (weights) instead of the whole model.
    
    This is more portable and recommended for production.
    
    Args:
        model: PyTorch model
        path: Path to save state dict (recommend .pth extension)
    """
    path = Path(path)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': type(model).__name__,
    }, path)
    logger.info(f"Saved model state_dict to {path}")


def load_model_state_dict(model: nn.Module, path: str, strict: bool = True):
    """
    Load weights from a state_dict file.
    
    Args:
        model: PyTorch model (must be created first with same architecture)
        path: Path to state dict file
        strict: If True, requires exact match of keys
        
    Example:
        # Create model
        model = MobileNetV2Base()
        # Load pretrained weights
        load_model_state_dict(model, 'model_weights.pth')
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # Assume the file is just a state_dict
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=strict)
    logger.info(f"Loaded model state_dict from {path}")


def transfer_weights_from_pretrained(target_model: nn.Module, 
                                     pretrained_path: str,
                                     freeze_transferred: bool = True):
    """
    Transfer weights from a pretrained model to a new model.
    
    This is useful when you want to use weights from one model as a starting
    point for another model with slightly different architecture.
    
    Args:
        target_model: Model to load weights into
        pretrained_path: Path to pretrained model weights
        freeze_transferred: Freeze the transferred weights
        
    Example:
        # Train a basic model first
        basic_model = MobileNetV2Base(num_outputs=2)
        # ... train it ...
        save_model_state_dict(basic_model, 'basic_model.pth')
        
        # Create uncertainty model and transfer backbone weights
        unc_model = MobileNetV2Uncertainty()
        transfer_weights_from_pretrained(unc_model, 'basic_model.pth')
    """
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    
    if 'model_state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['model_state_dict']
    
    target_dict = target_model.state_dict()
    
    # Filter out keys that don't match or have different sizes
    filtered_dict = {
        k: v for k, v in pretrained_dict.items() 
        if k in target_dict and v.shape == target_dict[k].shape
    }
    
    # Update target model
    target_dict.update(filtered_dict)
    target_model.load_state_dict(target_dict)
    
    logger.info(f"Transferred {len(filtered_dict)}/{len(pretrained_dict)} "
                f"weights from {pretrained_path}")
    
    # Optionally freeze transferred weights
    if freeze_transferred:
        for name, param in target_model.named_parameters():
            if name in filtered_dict:
                param.requires_grad = False
        logger.info("Froze transferred weights")


# ==================== EXAMPLE USAGE ====================

def example_usage():
    """
    Example showing different ways to use the MobileNet models.
    """
    print("=" * 60)
    print("MobileNet Model Examples")
    print("=" * 60)
    
    # 1. Basic MobileNet model
    print("\n1. Creating basic MobileNet model...")
    pilot = FastAIMobileNet(freeze_backbone=True, pretrained=True)
    print(f"   Model created: {pilot}")
    print(f"   Number of trainable parameters: "
          f"{sum(p.numel() for p in pilot.interpreter.model.parameters() if p.requires_grad):,}")
    
    # 2. MobileNet with uncertainty
    print("\n2. Creating MobileNet with uncertainty...")
    unc_pilot = FastAIMobileNetUncertainty(freeze_backbone=True, pretrained=True)
    print(f"   Model created: {unc_pilot}")
    
    # 3. Demonstrating state dict save/load
    print("\n3. State dict operations...")
    model = MobileNetV2Base(pretrained=True)
    
    # Save state dict
    temp_path = "/tmp/mobilenet_state.pth"
    save_model_state_dict(model, temp_path)
    
    # Create new model and load state dict
    new_model = MobileNetV2Base(pretrained=False)  # Don't load ImageNet weights
    load_model_state_dict(new_model, temp_path)
    print(f"   State dict saved and loaded successfully")
    
    # 4. Fine-tuning strategy
    print("\n4. Fine-tuning strategy...")
    print("   Step 1: Train with frozen backbone")
    print("   Step 2: Unfreeze last few layers")
    model.unfreeze_backbone(num_layers=3)
    print("   Step 3: Fine-tune with lower learning rate")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    example_usage()
