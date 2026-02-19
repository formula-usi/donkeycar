#!/usr/bin/env python3
"""
Validation script for uncertainty estimation models.

This script evaluates a trained uncertainty model on validation data and:
1. Creates plots for each image showing ground truth, predictions, and uncertainties
2. Separates images into confident/uncertain folders based on variance threshold
3. Generates summary statistics

Usage:
    python validate_uncertainty.py --model models/my_model.pt \
                                   --tub data/my_tub \
                                   --indexes validation_indexes.json \
                                   --threshold 0.1 \
                                   --output validation_results
"""

import argparse
import json
import os
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import logging

# Add parent directory to path to import donkeycar
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from donkeycar.parts.datastore_v2 import TubDataset
from donkeycar.parts.pytorch.torch_data import get_default_transform
from donkeycar.utils import get_model_by_type

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_validation_indexes(json_path: str) -> List[int]:
    """Load validation indexes from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Support different JSON formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        if 'validation_indexes' in data:
            return data['validation_indexes']
        elif 'indexes' in data:
            return data['indexes']
    
    raise ValueError(f"Invalid JSON format in {json_path}. Expected list or dict with 'validation_indexes' key")


def load_model(model_path: str, model_type: str = 'fastai_linear_unc'):
    """Load the trained uncertainty model."""
    logger.info(f"Loading model from {model_path}")
    
    # Get the model instance
    kl = get_model_by_type(model_type, cfg=None)
    
    # Load the trained weights
    kl.load(model_path)
    
    # Set to evaluation mode
    if hasattr(kl.interpreter, 'model'):
        kl.interpreter.model.eval()
    
    return kl


def evaluate_single_image(model, img_array: np.ndarray, record: Dict) -> Tuple[float, float, float, float, float, float]:
    """
    Evaluate a single image and return predictions and ground truth.
    
    Returns:
        (angle_pred, throttle_pred, angle_std, throttle_std, angle_gt, throttle_gt)
    """
    # Get ground truth
    angle_gt = record['user/angle']
    throttle_gt = record['user/throttle']
    
    # Run inference
    angle_pred, throttle_pred, angle_std, throttle_std = model.run(img_array)
    
    # Convert tensors to floats if needed
    if isinstance(angle_pred, torch.Tensor):
        angle_pred = angle_pred.item()
    if isinstance(throttle_pred, torch.Tensor):
        throttle_pred = throttle_pred.item()
    if isinstance(angle_std, torch.Tensor):
        angle_std = angle_std.item()
    if isinstance(throttle_std, torch.Tensor):
        throttle_std = throttle_std.item()
    
    return angle_pred, throttle_pred, angle_std, throttle_std, angle_gt, throttle_gt


def create_plot(img_array: np.ndarray, 
                angle_pred: float, throttle_pred: float,
                angle_std: float, throttle_std: float,
                angle_gt: float, throttle_gt: float,
                output_path: str):
    """Create a visualization plot for a single prediction."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Show the image
    ax_img = plt.subplot(1, 2, 1)
    ax_img.imshow(img_array)
    ax_img.set_title('Input Image')
    ax_img.axis('off')
    
    # Create bar chart for predictions vs ground truth
    ax_bars = plt.subplot(1, 2, 2)
    
    x = np.arange(2)
    width = 0.35
    
    # Ground truth bars
    gt_values = [angle_gt, throttle_gt]
    pred_values = [angle_pred, throttle_pred]
    std_values = [angle_std, throttle_std]
    
    bars1 = ax_bars.bar(x - width/2, gt_values, width, label='Ground Truth', alpha=0.7)
    bars2 = ax_bars.bar(x + width/2, pred_values, width, label='Predicted', alpha=0.7, yerr=std_values, capsize=5)
    
    ax_bars.set_ylabel('Value')
    ax_bars.set_title('Predictions vs Ground Truth')
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(['Angle', 'Throttle'])
    ax_bars.legend()
    ax_bars.grid(axis='y', alpha=0.3)
    ax_bars.set_ylim(-1.5, 1.5)
    
    # Add text with exact values
    textstr = f'Angle: GT={angle_gt:.3f}, Pred={angle_pred:.3f} ± {angle_std:.3f}\n'
    textstr += f'Throttle: GT={throttle_gt:.3f}, Pred={throttle_pred:.3f} ± {throttle_std:.3f}\n'
    textstr += f'Angle Error: {abs(angle_pred - angle_gt):.3f}\n'
    textstr += f'Throttle Error: {abs(throttle_pred - throttle_gt):.3f}'
    
    ax_bars.text(0.02, 0.98, textstr, transform=ax_bars.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Validate uncertainty estimation model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pt file)')
    parser.add_argument('--tub', type=str, required=True, help='Path to tub directory')
    parser.add_argument('--indexes', type=str, required=True, help='Path to JSON file with validation indexes')
    parser.add_argument('--threshold', type=float, default=0.46, 
                       help='Variance threshold for separating confident/uncertain predictions')
    parser.add_argument('--output', type=str, default='validation_results', 
                       help='Output directory for results')
    parser.add_argument('--model-type', type=str, default='fastai_linear_unc',
                       help='Type of model (e.g., fastai_linear_unc)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (for testing)')
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    confident_dir = output_dir / 'confident'
    uncertain_dir = output_dir / 'uncertain'
    plots_dir = output_dir / 'plots'
    
    confident_dir.mkdir(exist_ok=True)
    uncertain_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    logger.info(f"Output directories created at {output_dir}")
    
    # Load validation indexes
    logger.info(f"Loading validation indexes from {args.indexes}")
    val_indexes = load_validation_indexes(args.indexes)
    logger.info(f"Found {len(val_indexes)} validation samples")
    
    if args.max_samples:
        val_indexes = val_indexes[:args.max_samples]
        logger.info(f"Limiting to {len(val_indexes)} samples")
    
    # Load model
    model = load_model(args.model, args.model_type)
    
    # Load tub dataset
    logger.info(f"Loading tub from {args.tub}")
    tub = TubDataset(config=None, tub_paths=[args.tub], verbose=True)
    
    # Statistics tracking
    results = {
        'angle_errors': [],
        'throttle_errors': [],
        'angle_stds': [],
        'throttle_stds': [],
        'confident_count': 0,
        'uncertain_count': 0
    }
    
    # Process each validation sample
    logger.info("Starting evaluation...")
    for i, idx in enumerate(val_indexes):
        if i % 100 == 0:
            logger.info(f"Processing sample {i+1}/{len(val_indexes)}")
        
        # Get record
        record = tub[idx]
        img_array = record['cam/image_array']
        
        # Evaluate
        angle_pred, throttle_pred, angle_std, throttle_std, angle_gt, throttle_gt = \
            evaluate_single_image(model, img_array, record)
        
        # Calculate errors
        angle_error = abs(angle_pred - angle_gt)
        throttle_error = abs(throttle_pred - throttle_gt)
        
        # Store statistics
        results['angle_errors'].append(angle_error)
        results['throttle_errors'].append(throttle_error)
        results['angle_stds'].append(angle_std)
        results['throttle_stds'].append(throttle_std)
        
        # Create plot
        plot_path = plots_dir / f'sample_{idx:06d}.png'
        create_plot(img_array, angle_pred, throttle_pred, angle_std, throttle_std,
                   angle_gt, throttle_gt, str(plot_path))
        
        # Determine if confident or uncertain
        is_confident = (angle_std < args.threshold) and (throttle_std < args.threshold)
        
        if is_confident:
            results['confident_count'] += 1
            dest_dir = confident_dir
        else:
            results['uncertain_count'] += 1
            dest_dir = uncertain_dir
        
        # Copy image to appropriate folder
        img_dest = dest_dir / f'sample_{idx:06d}.jpg'
        Image.fromarray(img_array).save(img_dest)
    
    # Calculate summary statistics
    logger.info("\n" + "="*60)
    logger.info("VALIDATION RESULTS")
    logger.info("="*60)
    logger.info(f"Total samples evaluated: {len(val_indexes)}")
    logger.info(f"Confident predictions (both std < {args.threshold}): {results['confident_count']} ({results['confident_count']/len(val_indexes)*100:.1f}%)")
    logger.info(f"Uncertain predictions (any std >= {args.threshold}): {results['uncertain_count']} ({results['uncertain_count']/len(val_indexes)*100:.1f}%)")
    logger.info("")
    logger.info("Angle Statistics:")
    logger.info(f"  Mean Error: {np.mean(results['angle_errors']):.4f}")
    logger.info(f"  Median Error: {np.median(results['angle_errors']):.4f}")
    logger.info(f"  Mean Std: {np.mean(results['angle_stds']):.4f}")
    logger.info(f"  Median Std: {np.median(results['angle_stds']):.4f}")
    logger.info("")
    logger.info("Throttle Statistics:")
    logger.info(f"  Mean Error: {np.mean(results['throttle_errors']):.4f}")
    logger.info(f"  Median Error: {np.median(results['throttle_errors']):.4f}")
    logger.info(f"  Mean Std: {np.mean(results['throttle_stds']):.4f}")
    logger.info(f"  Median Std: {np.median(results['throttle_stds']):.4f}")
    logger.info("="*60)
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Error vs Uncertainty scatter plots
    axes[0, 0].scatter(results['angle_stds'], results['angle_errors'], alpha=0.5)
    axes[0, 0].set_xlabel('Predicted Std (Angle)')
    axes[0, 0].set_ylabel('Absolute Error (Angle)')
    axes[0, 0].set_title('Angle: Error vs Uncertainty')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(results['throttle_stds'], results['throttle_errors'], alpha=0.5)
    axes[0, 1].set_xlabel('Predicted Std (Throttle)')
    axes[0, 1].set_ylabel('Absolute Error (Throttle)')
    axes[0, 1].set_title('Throttle: Error vs Uncertainty')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histograms
    axes[1, 0].hist(results['angle_stds'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(args.threshold, color='r', linestyle='--', label=f'Threshold ({args.threshold})')
    axes[1, 0].set_xlabel('Predicted Std (Angle)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Angle Uncertainty')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(results['throttle_stds'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(args.threshold, color='r', linestyle='--', label=f'Threshold ({args.threshold})')
    axes[1, 1].set_xlabel('Predicted Std (Throttle)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Throttle Uncertainty')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_plot_path = output_dir / 'summary_statistics.png'
    plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"\nSummary plot saved to {summary_plot_path}")
    
    # Save results to JSON
    results_json = {
        'total_samples': len(val_indexes),
        'confident_count': results['confident_count'],
        'uncertain_count': results['uncertain_count'],
        'threshold': args.threshold,
        'angle_mean_error': float(np.mean(results['angle_errors'])),
        'angle_median_error': float(np.median(results['angle_errors'])),
        'angle_mean_std': float(np.mean(results['angle_stds'])),
        'angle_median_std': float(np.median(results['angle_stds'])),
        'throttle_mean_error': float(np.mean(results['throttle_errors'])),
        'throttle_median_error': float(np.median(results['throttle_errors'])),
        'throttle_mean_std': float(np.mean(results['throttle_stds'])),
        'throttle_median_std': float(np.median(results['throttle_stds'])),
    }
    
    results_json_path = output_dir / 'results.json'
    with open(results_json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    logger.info(f"Results saved to {results_json_path}")
    logger.info(f"\nValidation complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
