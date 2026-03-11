"""

fastai.py

Methods to create, use, save and load pilots. Pilots contain the highlevel
logic used to determine the angle and throttle of a vehicle. Pilots can
include one or more models to help direct the vehicles motion.

"""
from abc import ABC, abstractmethod
import os

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List, Sequence, Callable
from logging import getLogger

import donkeycar as dk
import torch
from donkeycar.utils import normalize_image, linear_bin
from donkeycar.pipeline.types import TubRecord, TubDataset
from donkeycar.pipeline.sequence import TubSequence
from donkeycar.parts.interpreter import FastAIInterpreter, Interpreter, KerasInterpreter
from donkeycar.parts.pytorch.torch_data import TorchTubDataset, get_default_transform

from fastai.vision.all import *
from fastai.data.transforms import *
from fastai import optimizer as fastai_optimizer
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from torch.nn.modules.loss import GaussianNLLLoss, _Loss
from torch.nn import functional as F
from PIL import Image
ONE_BYTE_SCALE = 1.0 / 255.0

# type of x
XY = Union[float, np.ndarray, Tuple[Union[float, np.ndarray], ...]]

logger = getLogger(__name__)

class SimpleUncLoss(nn.Module):
    """
    Simple uncertainty loss that combines MSE with uncertainty calibration.
    
    The loss has three components:
    1. MSE on predictions (angle and throttle means)
    2. Uncertainty calibration: trains log_var to match actual squared errors
    3. Variance regularization: prevents extreme variance values
    """

    def __init__(self, lambda_uncertainty: float = 0.8, var_reg: float = 0.01):
        super().__init__()
        self.lambda_uncertainty = lambda_uncertainty
        self.var_reg = var_reg
        
    def forward(self, pred, target):
        # pred has shape [batch_size, 4]: [angle_mean, throttle_mean, angle_log_var, throttle_log_var]
        # target has shape [batch_size, 2]: [angle_target, throttle_target]
        
        angle_mean = pred[:, 0]
        throttle_mean = pred[:, 1]
        angle_log_var = pred[:, 2]
        throttle_log_var = pred[:, 3]
        
        angle_target = target[:, 0]
        throttle_target = target[:, 1]
        
        # Clamp log_var to prevent extreme values and numerical instability
        # Range: exp(-10) ≈ 0.00005 to exp(2) ≈ 7.4
        angle_log_var = torch.clamp(angle_log_var, -10, 2)
        throttle_log_var = torch.clamp(throttle_log_var, -10, 2)
        
        # Component 1: MSE for predictions
        angle_squared_error = (angle_mean - angle_target) ** 2
        throttle_squared_error = (throttle_mean - throttle_target) ** 2
        mse = angle_squared_error + throttle_squared_error
        
        # Component 2: Uncertainty calibration loss
        # Train the log_var to match the actual log of squared errors
        angle_unc_mse = (angle_log_var - torch.log(angle_squared_error + 1e-8))**2
        throttle_unc_mse = (throttle_log_var - torch.log(throttle_squared_error + 1e-8))**2
        unc_loss = angle_unc_mse + throttle_unc_mse
        
        # Component 3: Regularization to penalize extreme variances
        # Encourages log_var to stay near 0 (variance near 1)
        var_penalty = self.var_reg * (angle_log_var**2 + throttle_log_var**2)
        
        # Combine all components
        total_loss = self.lambda_uncertainty * mse + (1 - self.lambda_uncertainty) * unc_loss + var_penalty
        
        # Check for NaN/inf to prevent lr_find crashes
        result = torch.mean(total_loss)
        if torch.isnan(result) or torch.isinf(result):
            return torch.tensor(1e6, device=result.device, dtype=result.dtype)
        return result


class NLLLoss(_Loss):

    def __init__(
        self, *, full: bool = False, eps: float = 1e-6, reduction: str = "mean"
    ) -> None:
        super().__init__(None, None, reduction)
        self.full = full
        self.eps = eps

    def forward(self, pred, target):
        
        angle_mean = pred[:, 0]
        throttle_mean = pred[:, 1]
        angle_var = torch.exp(pred[:, 2])**2
        throttle_var = torch.exp(pred[:, 3])**2
        angle_target = target[:, 0]
        throttle_target = target[:, 1]

        return (F.gaussian_nll_loss(
            angle_mean, angle_target, angle_var, full=self.full, eps=self.eps, reduction=self.reduction) +  F.gaussian_nll_loss(throttle_mean, throttle_target, throttle_var, full=self.full, eps=self.eps, reduction=self.reduction))/2
               

class UncertaintyLoss(nn.Module):
    """
    Negative Log-Likelihood loss for Gaussian distributions.
    
    The model predicts [angle_mean, throttle_mean, angle_log_var, throttle_log_var]
    where log_var = log(σ²) is the log-variance.
    
    For a Gaussian distribution N(μ, σ²), the NLL is:
        NLL = 0.5 * log(σ²) + 0.5 * (y - μ)² / σ²
            = 0.5 * log_var + 0.5 * (y - μ)² * exp(-log_var)
    
    This allows the model to learn both the prediction (mean) and its uncertainty (variance).
    """
    def __init__(self, throttle_weight: float = 1.0, max_log_var: float = 1.0, min_log_var: float = -6.0):
        super().__init__()
        self.throttle_weight = throttle_weight
        self.max_log_var = max_log_var
        self.min_log_var = min_log_var
    
    def forward(self, pred, target):
        # pred has shape [batch_size, 4]: [angle_mean, throttle_mean, angle_log_var, throttle_log_var]
        # target has shape [batch_size, 2]: [angle_target, throttle_target]
        
        angle_mean = pred[:, 0]
        throttle_mean = pred[:, 1]
        angle_log_var = pred[:, 2]
        throttle_log_var = pred[:, 3]
        
        angle_target = target[:, 0]
        throttle_target = target[:, 1]
        
        # Clamp log_var to prevent extreme uncertainties
        # min_log_var = -6 means min variance ≈ 0.0025 (very confident)
        # max_log_var = 1 means max variance ≈ 2.7 (reasonable uncertainty)
        angle_log_var = torch.clamp(angle_log_var, self.min_log_var, self.max_log_var)
        throttle_log_var = torch.clamp(throttle_log_var, self.min_log_var, self.max_log_var)
        
        # Negative Log-Likelihood for Gaussian distribution
        # NLL = 0.5 * log(σ²) + 0.5 * (y - μ)² / σ²
        angle_loss = 0.5 * angle_log_var + 0.5 * (angle_mean - angle_target) ** 2 * torch.exp(-angle_log_var)
        throttle_loss = 0.5 * throttle_log_var + 0.5 * (throttle_mean - throttle_target) ** 2 * torch.exp(-throttle_log_var)
        
        # Use configurable weight for throttle (default 1.0 for equal weighting)
        total_loss = angle_loss + self.throttle_weight * throttle_loss
        
        # Add small epsilon and check for NaN/inf to prevent lr_find crashes
        result = torch.mean(total_loss)
        if torch.isnan(result) or torch.isinf(result):
            # Return a large but finite loss instead of NaN
            return torch.tensor(1e6, device=result.device, dtype=result.dtype)
        return result


class FastAiPilot(ABC):
    """
    Base class for Fast AI models that will provide steering and throttle to
    guide a car.
    """

    def __init__(self,
                 interpreter: Interpreter = FastAIInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3)) -> None:
        self.model: Optional[Model] = None
        self.input_shape = input_shape
        self.optimizer = "adam"
        self.interpreter = interpreter
        self.interpreter.set_model(self)
        self.learner = None
        logger.info(f'Created {self} with interpreter: {interpreter}')

    def load(self, model_path):
        logger.info(f'Loading model {model_path}')
        self.interpreter.load(model_path)

    def load_weights(self, model_path: str, by_name: bool = True) -> None:
        self.interpreter.load_weights(model_path, by_name=by_name)

    def shutdown(self) -> None:
        pass

    def compile(self) -> None:
        pass

    @abstractmethod
    def create_model(self):
        pass

    def set_optimizer(self, optimizer_type: str,
                      rate: float, decay: float) -> None:
        if optimizer_type == "adam":
            optimizer = fastai_optimizer.Adam(lr=rate, wd=decay)
        elif optimizer_type == "sgd":
            optimizer = fastai_optimizer.SGD(lr=rate, wd=decay)
        elif optimizer_type == "rmsprop":
            optimizer = fastai_optimizer.RMSprop(lr=rate, wd=decay)
        else:
            raise Exception(f"Unknown optimizer type: {optimizer_type}")
        self.interpreter.set_optimizer(optimizer)

    # shape
    def get_input_shape(self, input_name):
        return self.interpreter.get_input_shape(input_name)

    def seq_size(self) -> int:
        return 0

    def run(self, img_arr: np.ndarray, other_arr: List[float] = None) \
            -> Tuple[Union[float, torch.tensor], ...]:
        """
        Donkeycar parts interface to run the part in the loop.

        :param img_arr:     uint8 [0,255] numpy array with image data
        :param other_arr:   numpy array of additional data to be used in the
                            pilot, like IMU array for the IMU model or a
                            state vector in the Behavioural model
        :return:            tuple of (angle, throttle)
        """
        transform = get_default_transform(resize=False, for_inference=True)
        norm_arr = transform(img_arr)

        folder = 'inference_images'
        #get the highest index in the folder
        os.makedirs(folder, exist_ok=True)
        #get the highest index of the images in the folder
        existing_images = [f for f in os.listdir(folder) if f.endswith('.jpg')]
        if existing_images:
            latest_index = max(int(f.split('.')[0]) for f in existing_images)
        else:
            latest_index = -1
        next_index = latest_index + 1

        if next_index < 50:
            # norm_arr is already the transformed tensor
            # Denormalize and convert tensor back to PIL Image
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            denormalized = norm_arr * std + mean
            # Clamp to [0, 1] and convert to [0, 255]
            denormalized = torch.clamp(denormalized, 0, 1)
            # Convert from CHW to HWC and to numpy
            img_numpy = (denormalized.permute(1, 2, 0).numpy() * 255).astype('uint8')
            Image.fromarray(img_numpy).save(os.path.join(folder, f'{next_index}.jpg'))


        tensor_other_array = torch.FloatTensor(other_arr) if other_arr else None
        return self.inference(norm_arr, tensor_other_array)

    def inference(self, img_arr: torch.tensor, other_arr: Optional[torch.tensor]) \
            -> Tuple[Union[float, torch.tensor], ...]:
        """ Inferencing using the interpreter
            :param img_arr:     float32 [0,1] numpy array with normalized image
                                data
            :param other_arr:   tensor array of additional data to be used in the
                                pilot, like IMU array for the IMU model or a
                                state vector in the Behavioural model
            :return:            tuple of (angle, throttle)
        """
        out = self.interpreter.predict(img_arr, other_arr)
        return self.interpreter_to_output(out)

    def inference_from_dict(self, input_dict: Dict[str, np.ndarray]) \
            -> Tuple[Union[float, np.ndarray], ...]:
        """ Inferencing using the interpreter
            :param input_dict:  input dictionary of str and np.ndarray
            :return:            typically tuple of (angle, throttle)
        """
        output = self.interpreter.predict_from_dict(input_dict)
        return self.interpreter_to_output(output)

    @abstractmethod
    def interpreter_to_output(
            self,
            interpreter_out: Sequence[Union[float, np.ndarray]]) \
            -> Tuple[Union[float, np.ndarray], ...]:
        """ Virtual method to be implemented by child classes for conversion
            :param interpreter_out:  input data
            :return:                 output values, possibly tuple of np.ndarray
        """
        pass

    def train(self,
              model_path: str,
              train_data: TorchTubDataset,
              train_steps: int,
              batch_size: int,
              validation_data: TorchTubDataset,
              validation_steps: int,
              epochs: int,
              verbose: int = 1,
              min_delta: float = .0005,
              patience: int = 5,
              show_plot: bool = False):
        """
        trains the model
        """
        assert isinstance(self.interpreter, FastAIInterpreter)
        model = self.interpreter.model

        # Detect device and log info
        use_cuda = torch.cuda.is_available()
        logger.info(f"CUDA available: {use_cuda}")
        if use_cuda:
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

        # Create DataLoaders first (on CPU, data will be moved per batch)
        # Use num_workers=0 to avoid multiprocessing issues with CUDA
        logger.info(f"Training dataset size: {len(train_data)}")
        logger.info(f"Validation dataset size: {len(validation_data)}")
        logger.info(f"Batch size: {batch_size}")
        
        # Check if dataset is too small
        if len(train_data) == 0:
            raise ValueError("Training dataset is empty!")
        if len(validation_data) == 0:
            raise ValueError("Validation dataset is empty!")
        
        dataLoader = DataLoaders.from_dsets(train_data, validation_data, bs=batch_size, shuffle=False, num_workers=0)
        #get the first batch 

        

        callbacks = [
            EarlyStoppingCallback(monitor='valid_loss',
                                  patience=patience,
                                  min_delta=min_delta),
            SaveModelCallback(monitor='valid_loss',
                              every_epoch=False
                              )
        ]

        self.learner = Learner(dataLoader, model, loss_func=self.loss, path=Path(model_path).parent)
        
        # Move learner to GPU after creation
        if use_cuda:
            self.learner.model = self.learner.model.cuda()
            self.learner.dls = self.learner.dls.cuda()
            logger.info("Learner moved to CUDA")
        
        logger.info(f"Learner device: {next(self.learner.model.parameters()).device}")
        
        # Log model architecture info
        if hasattr(model, 'subnetworks'):
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model has {len(model.subnetworks)} subnetworks with total {total_params:,} trainable parameters")
            # Save initial weights to check if unused subnetworks change
            initial_weights = {}
            for i, subnet in enumerate(model.subnetworks):
                initial_weights[i] = subnet.conv24.weight.data.clone()
                logger.info(f"Subnetwork {i} initial weight mean: {initial_weights[i].mean().item():.6f}")

        logger.info(self.learner.summary())
        logger.info(self.learner.loss_func)

        # Try to find optimal learning rate, fallback to default if it fails
        try:
            lr_result = self.learner.lr_find()
            suggestedLr = float(lr_result[0])
            logger.info(f"Suggested Learning Rate {suggestedLr}")
        except (IndexError, ValueError, RuntimeError) as e:
            # lr_find failed (common with uncertainty models during early training)
            # Use a reasonable default learning rate
            suggestedLr = 1e-3
            logger.warning(f"lr_find failed ({e}), using default learning rate: {suggestedLr}")

        self.learner.fit_one_cycle(epochs, suggestedLr, cbs=callbacks)
        
        # Check if unused subnetworks changed
        if hasattr(model, 'subnetworks') and 'initial_weights' in locals():
            logger.info("\nWeight changes after training:")
            for i, subnet in enumerate(model.subnetworks):
                weight_diff = (subnet.conv24.weight.data - initial_weights[i]).abs().mean().item()
                logger.info(f"Subnetwork {i}: weight change = {weight_diff:.6e}")

        torch.save(self.learner.model, model_path)

        if show_plot:
            self.learner.recorder.plot_loss()
            plt.savefig(Path(model_path).with_suffix('.png'))

        history = { "loss" : list(map((lambda x: x.item()), self.learner.recorder.losses)) }
        return history

    def __str__(self) -> str:
        """ For printing model initialisation """
        return type(self).__name__


class FastAILinear(FastAiPilot):
    """
    The KerasLinear pilot uses one neuron to output a continuous value via
    the Keras Dense layer with linear activation. One each for steering and
    throttle. The output is not bounded.
    """

    def __init__(self,
                 interpreter: Interpreter = FastAIInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_outputs: int = 2):
        self.num_outputs = num_outputs
        self.loss = MSELossFlat()

        super().__init__(interpreter, input_shape)

    def create_model(self):
        return Linear()

    def compile(self):
        self.optimizer = self.optimizer
        self.loss = 'mse'

    def interpreter_to_output(self, interpreter_out):
        interpreter_out = (interpreter_out * 2) - 1
        steering = interpreter_out[0]
        throttle = interpreter_out[1]
        return steering, throttle

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        assert isinstance(record, TubRecord), 'TubRecord expected'
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        return {'n_outputs0': angle, 'n_outputs1': throttle}

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shape('img')[1:]
        return img_shape

class FastAIUncertainty(FastAILinear):
    """
    The FastAIUncertainty pilot predicts Gaussian distributions for steering and throttle.
    
    Outputs 4 values: [angle_mean, throttle_mean, angle_log_variance, throttle_log_variance]
    - The means are the actual predictions used for control
    - The log-variances represent the uncertainty/confidence in each prediction
    - Higher variance = less confident prediction
    """

    def __init__(self,
                 interpreter: Interpreter = FastAIInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_outputs: int = 4,
                 loss_type: str = 'nll',  # 'nll' or 'simple'
                 lambda_uncertainty: float = 0.8,
                 var_reg: float = 0.01,
                 throttle_weight: float = 1.0,
                 max_log_var: float = 1.0,
                 min_log_var: float = -6.0):
        """
        Args:
            loss_type: 'nll' for NegativeLogLikelihood or 'simple' for SimpleUncLoss
            lambda_uncertainty: For SimpleUncLoss, balance between MSE and uncertainty (default 0.8)
            var_reg: For SimpleUncLoss, variance regularization strength (default 0.01)
            throttle_weight: For NLL, relative weight of throttle vs angle (default 1.0)
            max_log_var: Maximum allowed log-variance (default 1.0, variance ≈ 2.7)
            min_log_var: Minimum allowed log-variance (default -6.0, variance ≈ 0.0025)
        """
        self.num_outputs = num_outputs
        self.loss_type = loss_type
        self.lambda_uncertainty = lambda_uncertainty
        self.var_reg = var_reg
        self.throttle_weight = throttle_weight
        self.max_log_var = max_log_var
        self.min_log_var = min_log_var

        super().__init__(interpreter, input_shape)
        # Set loss after super().__init__() to avoid it being overwritten
        if loss_type == 'unc':
            self.loss = UncertaintyLoss(
                throttle_weight=self.throttle_weight,
                max_log_var=self.max_log_var,
                min_log_var=self.min_log_var
            )
        elif loss_type == 'nll':
            self.loss = NLLLoss()
        else:
            self.loss = SimpleUncLoss(
                lambda_uncertainty=self.lambda_uncertainty,
                var_reg=self.var_reg
            )

    def create_model(self):
        return LinearUncertainty()
    
    def compile(self):
        self.optimizer = self.optimizer
        if self.loss_type == 'nll':
            self.loss = UncertaintyLoss(
                throttle_weight=self.throttle_weight,
                max_log_var=self.max_log_var,
                min_log_var=self.min_log_var
            )
        else:
            self.loss = SimpleUncLoss(
                lambda_uncertainty=self.lambda_uncertainty,
                var_reg=self.var_reg
            )

    def interpreter_to_output(self, interpreter_out):
        # Scale outputs from [0, 1] to [-1, 1] for angle and throttle
        # Note: log_variance is already in the correct scale (can be any real number)
        angle_mean = (interpreter_out[0] * 2) - 1
        throttle_mean = (interpreter_out[1] * 2) - 1
        angle_log_var = interpreter_out[2]
        throttle_log_var = interpreter_out[3]
        
        # Convert log-variance to standard deviation for easier interpretation
        # Use numpy operations since interpreter_out is numpy array
        angle_std = np.sqrt(np.exp(angle_log_var))
        throttle_std = np.sqrt(np.exp(throttle_log_var))
        
        return angle_mean, throttle_mean, angle_std, throttle_std

class FastAILinearMW(FastAILinear):
    """
    The FastAILinearMW pilot uses one subnetwork per weather condition. Each
    subnetwork is a FastAILinear model. The output is not bounded.
    """

    def __init__(self,
                 interpreter: Interpreter = FastAIInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_outputs: int = 2,
                 n_weathers: int = 3,
                 surface_id: int = 0):
        self.n_weathers = n_weathers
        self.surface_id = surface_id  # Default surface for inference
        super().__init__(interpreter, input_shape, num_outputs)

    def create_model(self):
        return LinearMW(self.n_weathers)
    
    def set_surface_id(self, surface_id: int) -> None:
        """Set the current surface/weather condition for inference"""
        if surface_id >= self.n_weathers:
            logger.warning(f"Surface ID {surface_id} exceeds number of weathers {self.n_weathers}, using 0")
            surface_id = 0
        self.surface_id = surface_id
        # Update the model's surface_id for inference
        if hasattr(self.interpreter, 'model') and self.interpreter.model is not None:
            self.interpreter.model.inference_surface_id = surface_id
    
    def run(self, img_arr: np.ndarray, other_arr: List[float] = None) \
            -> Tuple[Union[float, torch.tensor], ...]:
        """
        Override run to handle surface_id as integer (not float).
        If other_arr is provided, it's assumed to be [surface_id] or surface_id.
        """
        transform = get_default_transform(resize=False, for_inference=True)
        norm_arr = transform(img_arr)
        
        # If other_arr is provided, use it as surface_id (convert to LongTensor)
        if other_arr is not None:
            surface_id_value = other_arr[0] if isinstance(other_arr, list) else other_arr
            tensor_other_array = torch.LongTensor([int(surface_id_value)])
        else:
            tensor_other_array = None
        
        return self.inference(norm_arr, tensor_other_array)
    

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
        self.output1_uncertainty = nn.Linear(50+1, 1)
        self.output2_uncertainty = nn.Linear(50+1, 1)
        
        # Initialize uncertainty heads with small weights and reasonable bias
        # This makes initial log_variance close to -1 (variance ≈ 0.37)
        # Prevents starting with overly confident or overly uncertain predictions
        with torch.no_grad():
            self.output1_uncertainty.weight.normal_(0.0, 0.001)
            self.output1_uncertainty.bias.fill_(-1.0)
            self.output2_uncertainty.weight.normal_(0.0, 0.001)
            self.output2_uncertainty.bias.fill_(-1.0)
            
            # Also ensure mean prediction heads have reasonable initialization
            # to keep predictions in [0, 1] range initially
            self.output1.weight.normal_(0.0, 0.01)
            self.output1.bias.fill_(0.5)  # Center of [0, 1]
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
        angle_uncertainty = self.output1_uncertainty(torch.cat([x1, angle], dim=1))
        throttle_uncertainty = self.output2_uncertainty(torch.cat([x1, throttle], dim=1))
        
        # Clamp outputs to prevent extreme values during training
        # Means should stay roughly in [0, 1] since targets are normalized to this range
        angle = torch.clamp(angle, -0.5, 1.5)
        throttle = torch.clamp(throttle, -0.5, 1.5)
        
        # Clamp uncertainties to prevent NaN during lr_find
        angle_uncertainty = torch.clamp(angle_uncertainty, -10, 2)
        throttle_uncertainty = torch.clamp(throttle_uncertainty, -10, 2)
        
        return torch.cat((angle, throttle, angle_uncertainty, throttle_uncertainty), 1)

class LinearMW(nn.Module):
    def __init__(self, n_weathers = 3):
        super().__init__()
        self.subnetworks = nn.ModuleList([Linear() for _ in range(n_weathers)])
        self.inference_surface_id = 0  # Default for inference

    def forward(self, x):
        # During training: x is a tuple of (image, surface_id)
        # During inference with surface passed: x is a list [image, surface_id]
        # During inference without surface: x is just the image tensor
        
        # Log training mode status occasionally
        
        if isinstance(x, (tuple, list)):
            # Training/inference mode with surface_id provided
            img, surface_id = x[0], x[1]
            
            # Handle batched training vs single inference
            if isinstance(surface_id, torch.Tensor) and surface_id.dim() > 0 and len(surface_id) > 1:
                # Check if all surface_ids in batch are the same
                unique_surfaces = torch.unique(surface_id)
                if len(unique_surfaces) == 1:
                    # All same surface - process as one batch (fast path)
                    surface_idx = int(unique_surfaces[0].item())
                    surface_idx = max(0, min(surface_idx, len(self.subnetworks) - 1))
                    return self.subnetworks[surface_idx](img)
                else:
                    # Mixed surfaces in batch - process each sample individually
                    batch_size = img.shape[0]
                    outputs = []
                    for i in range(batch_size):
                        sample_img = img[i:i+1]  # Keep batch dimension
                        surface_idx = int(surface_id[i].item())
                        surface_idx = max(0, min(surface_idx, len(self.subnetworks) - 1))
                        output = self.subnetworks[surface_idx](sample_img)
                        outputs.append(output)
                    return torch.cat(outputs, dim=0)
            else:
                # Single inference
                # Check if batch dimension is missing and add it
                if img.dim() == 3:  # [C, H, W] -> need [1, C, H, W]
                    img = img.unsqueeze(0)
                
                # Extract scalar value from surface_id tensor
                if isinstance(surface_id, torch.Tensor):
                    if surface_id.dim() > 0:
                        surface_idx = int(surface_id[0].item())
                    else:
                        surface_idx = int(surface_id.item())
                else:
                    surface_idx = int(surface_id)
        else:
            # Inference mode without surface_id: use the stored inference_surface_id
            img = x
            surface_idx = self.inference_surface_id
        
        # Ensure surface_idx is valid
        surface_idx = max(0, min(surface_idx, len(self.subnetworks) - 1))
        
        # Use the appropriate subnetwork based on surface_id
        return self.subnetworks[surface_idx](img)
    
class LinearMWUncertainty(LinearMW):
    def __init__(self, n_weathers = 3):
        super().__init__(n_weathers)
        self.subnetworks = nn.ModuleList([LinearUncertainty() for _ in range(n_weathers)])

class FastAILinearMWUncertainty(FastAIUncertainty):
    """
    The FastAILinearMWUncertainty pilot uses one LinearUncertainty subnetwork per weather condition.
    Each subnetwork predicts steering, throttle, and their uncertainties.
    
    Outputs 4 values per prediction: [angle_mean, throttle_mean, angle_std, throttle_std]
    """

    def __init__(self,
                 interpreter: Interpreter = FastAIInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_outputs: int = 4,
                 n_weathers: int = 3,
                 surface_id: int = 0,
                 loss_type: str = 'nll',
                 lambda_uncertainty: float = 0.8,
                 var_reg: float = 0.01,
                 throttle_weight: float = 1.0,
                 max_log_var: float = 1.0,
                 min_log_var: float = -6.0):
        self.n_weathers = n_weathers
        self.surface_id = surface_id  # Default surface for inference
        super().__init__(interpreter, input_shape, num_outputs, loss_type, 
                         lambda_uncertainty, var_reg, throttle_weight, 
                         max_log_var, min_log_var)

    def create_model(self):
        return LinearMWUncertainty(self.n_weathers)
    
    def set_surface_id(self, surface_id: int) -> None:
        """Set the current surface/weather condition for inference"""
        if surface_id >= self.n_weathers:
            logger.warning(f"Surface ID {surface_id} exceeds number of weathers {self.n_weathers}, using 0")
            surface_id = 0
        self.surface_id = surface_id
        # Update the model's surface_id for inference
        if hasattr(self.interpreter, 'model') and self.interpreter.model is not None:
            self.interpreter.model.inference_surface_id = surface_id
    
    def run(self, img_arr: np.ndarray, other_arr: List[float] = None) \
            -> Tuple[Union[float, torch.tensor], ...]:
        """
        Override run to handle surface_id as integer (not float).
        If other_arr is provided, it's assumed to be [surface_id] or surface_id.
        """
        transform = get_default_transform(resize=False, for_inference=True)
        norm_arr = transform(img_arr)
        
        
        # If other_arr is provided, use it as surface_id (convert to LongTensor)
        if other_arr is not None:
            surface_id_value = other_arr[0] if isinstance(other_arr, list) else other_arr
            tensor_other_array = torch.LongTensor([int(surface_id_value)])
        else:
            tensor_other_array = None
        
        return self.inference(norm_arr, tensor_other_array)
