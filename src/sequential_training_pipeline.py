#!/usr/bin/env python3
"""
Sequential Model Training Pipeline for Liver Segmentation
Trains models one by one to reduce memory usage and training time
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# MONAI imports
from monai.data import Dataset, DataLoader as MonaiDataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandShiftIntensityd, ToTensord,
    Resized, EnsureTyped
)
from monai.networks.nets import UNet, SegResNet, SwinUNETR, VNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SequentialSegmentationPipeline:
    def __init__(self, data_dir: str, results_dir: str = "sequential_results"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Set deterministic training
        set_determinism(seed=42)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Model configurations
        self.spatial_size = (128, 128, 64)
        self.models_to_train = ["UNet", "SegResNet", "SwinUNETR", "VNet"]
        
        # Training parameters
        self.num_epochs = 30
        self.batch_size = 2
        self.learning_rate = 1e-4
        self.val_interval = 3
        
        # Initialize data splits
        self.train_files = []
        self.val_files = []
        self.test_files = []
        
    def setup_data_splits(self):
        """Setup train/validation/test splits"""
        # Load existing splits if available
        splits_file = self.results_dir / "data_splits.json"
        if splits_file.exists():
            logger.info("Loading existing data splits...")
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            self.train_files = splits['train']
            self.val_files = splits['val']
            self.test_files = splits['test']
            return
        
        # Create new splits
        logger.info("Creating new data splits...")
        
        # Get all image files
        original_images_dir = self.data_dir / "imagesTr"
        enhanced_images_dir = self.data_dir / "imagesTr_enhanced"
        labels_dir = self.data_dir / "labelsTr"
        
        # Get list of available cases
        image_files = sorted(list(original_images_dir.glob("*.nii")))
        case_ids = [f.stem for f in image_files]
        
        # Split data: 70% train, 15% val, 15% test
        np.random.seed(42)
        np.random.shuffle(case_ids)
        
        n_total = len(case_ids)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        train_ids = case_ids[:n_train]
        val_ids = case_ids[n_train:n_train + n_val]
        test_ids = case_ids[n_train + n_val:]
        
        # Create file dictionaries for both datasets
        def create_file_dicts(ids, dataset_type):
            files = []
            for case_id in ids:
                # Original dataset
                files.append({
                    'image': str(original_images_dir / f"{case_id}.nii"),
                    'label': str(labels_dir / f"{case_id}.nii"),
                    'dataset': 'original'
                })
                # Enhanced dataset
                enhanced_file = enhanced_images_dir / f"{case_id}_bbhe_enhanced.nii"
                if enhanced_file.exists():
                    files.append({
                        'image': str(enhanced_file),
                        'label': str(labels_dir / f"{case_id}.nii"),
                        'dataset': 'enhanced'
                    })
            return files
        
        self.train_files = create_file_dicts(train_ids, 'train')
        self.val_files = create_file_dicts(val_ids, 'val')
        self.test_files = create_file_dicts(test_ids, 'test')
        
        # Save splits
        splits = {
            'train': self.train_files,
            'val': self.val_files,
            'test': self.test_files
        }
        
        with open(splits_file, 'w') as f:
            json.dump(splits, f, indent=2)
        
        logger.info(f"Data splits created: {len(self.train_files)} train, {len(self.val_files)} val, {len(self.test_files)} test")
    
    def get_transforms(self, mode="train"):
        """Get data transforms for training or validation"""
        if mode == "train":
            return Compose([
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Resized(keys=["image", "label"], spatial_size=self.spatial_size, mode=("trilinear", "nearest")),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=self.spatial_size,
                    pos=1,
                    neg=1,
                    num_samples=2,
                    image_key="image",
                    image_threshold=0,
                ),
                RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
                RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
                RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
                RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
                RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
                EnsureTyped(keys=["image", "label"]),
            ])
        else:
            return Compose([
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Resized(keys=["image", "label"], spatial_size=self.spatial_size, mode=("trilinear", "nearest")),
                EnsureTyped(keys=["image", "label"]),
            ])
    
    def create_model(self, model_name: str) -> nn.Module:
        """Create a model based on the model name"""
        spatial_size = self.spatial_size
        
        if model_name == "UNet":
            model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=3,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm="batch",
            )
        elif model_name == "SegResNet":
            model = SegResNet(
                spatial_dims=3,
                init_filters=32,
                in_channels=1,
                out_channels=3,
                dropout_prob=0.2,
            )
        elif model_name == "SwinUNETR":
            model = SwinUNETR(
                img_size=spatial_size,
                in_channels=1,
                out_channels=3,
                feature_size=48,
                use_checkpoint=True,
            )
        elif model_name == "VNet":
            model = VNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=3,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model.to(self.device)
    
    def train_single_model(self, model_name: str, dataset_type: str = "original") -> Dict[str, Any]:
        """Train a single model on specified dataset"""
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_name} on {dataset_type} dataset")
        logger.info(f"{'='*50}")
        
        # Ensure data splits are loaded
        if not self.train_files:
            self.setup_data_splits()
        
        # Filter data for specific dataset
        train_data = [f for f in self.train_files if f['dataset'] == dataset_type]
        val_data = [f for f in self.val_files if f['dataset'] == dataset_type]
        
        logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
        
        # Create datasets and loaders
        train_transforms = self.get_transforms("train")
        val_transforms = self.get_transforms("val")
        
        train_dataset = Dataset(data=train_data, transform=train_transforms)
        val_dataset = Dataset(data=val_data, transform=val_transforms)
        
        train_loader = MonaiDataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        val_loader = MonaiDataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=2
        )
        
        # Create model
        model = self.create_model(model_name)
        
        # Loss function and optimizer
        loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        optimizer = Adam(model.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        
        # Metrics
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        
        # Training loop
        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = []
        metric_values = []
        
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            model.train()
            epoch_loss = 0
            step = 0
            
            for batch_data in train_loader:
                step += 1
                inputs, labels = batch_data["image"].to(self.device), batch_data["label"].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if step % 20 == 0:
                    logger.info(f"  Step {step}, Loss: {loss.item():.4f}")
            
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            logger.info(f"  Epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            
            scheduler.step()
            
            # Validation
            if (epoch + 1) % self.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    for val_data in val_loader:
                        val_inputs, val_labels = val_data["image"].to(self.device), val_data["label"].to(self.device)
                        roi_size = self.spatial_size
                        sw_batch_size = 2
                        val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                        val_outputs = torch.softmax(val_outputs, 1)
                        dice_metric(y_pred=val_outputs, y=val_labels)
                    
                    metric = dice_metric.aggregate().item()
                    dice_metric.reset()
                    metric_values.append(metric)
                    
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        # Save best model
                        model_path = self.results_dir / f"{model_name}_{dataset_type}_best.pth"
                        torch.save(model.state_dict(), model_path)
                        logger.info(f"  New best model saved! Dice: {metric:.4f}")
                    
                    logger.info(f"  Validation Dice: {metric:.4f}, Best: {best_metric:.4f}")
        
        # Save final model
        final_model_path = self.results_dir / f"{model_name}_{dataset_type}_final.pth"
        torch.save(model.state_dict(), final_model_path)
        
        # Clear GPU memory
        del model
        torch.cuda.empty_cache()
        
        results = {
            'model_name': model_name,
            'dataset_type': dataset_type,
            'best_metric': best_metric,
            'best_metric_epoch': best_metric_epoch,
            'final_loss': epoch_loss_values[-1],
            'loss_history': epoch_loss_values,
            'metric_history': metric_values
        }
        
        logger.info(f"\n{model_name} on {dataset_type} completed!")
        logger.info(f"Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")
        
        return results
    
    def run_sequential_training(self) -> Dict[str, Any]:
        """Run sequential training for all models"""
        logger.info("Starting Sequential Segmentation Pipeline")
        logger.info(f"Device: {self.device}")
        
        # Setup data
        self.setup_data_splits()
        
        all_results = {}
        
        # Train each model on both datasets
        for model_name in self.models_to_train:
            for dataset_type in ["original", "enhanced"]:
                try:
                    results = self.train_single_model(model_name, dataset_type)
                    key = f"{model_name}_{dataset_type}"
                    all_results[key] = results
                    
                    # Save intermediate results
                    results_file = self.results_dir / "training_results.json"
                    with open(results_file, 'w') as f:
                        json.dump(all_results, f, indent=2)
                    
                except Exception as e:
                    logger.error(f"Error training {model_name} on {dataset_type}: {str(e)}")
                    continue
        
        logger.info("\n" + "="*60)
        logger.info("SEQUENTIAL TRAINING COMPLETED")
        logger.info("="*60)
        
        # Print summary
        for key, results in all_results.items():
            logger.info(f"{key}: Best Dice = {results['best_metric']:.4f}")
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description="Sequential Liver Segmentation Training")
    parser.add_argument("--data_dir", type=str, 
                       default="Dataset/Liver Segmentation",
                       help="Path to dataset directory")
    parser.add_argument("--results_dir", type=str,
                       default="sequential_results",
                       help="Path to results directory")
    parser.add_argument("--model", type=str, choices=["UNet", "SegResNet", "SwinUNETR", "VNet"],
                       help="Train only specific model (optional)")
    parser.add_argument("--dataset", type=str, choices=["original", "enhanced"],
                       help="Train on specific dataset (optional)")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = SequentialSegmentationPipeline(
        data_dir=args.data_dir,
        results_dir=args.results_dir
    )
    
    if args.model and args.dataset:
        # Train single model on single dataset
        results = pipeline.train_single_model(args.model, args.dataset)
        logger.info(f"Training completed: {results}")
    elif args.model:
        # Train single model on both datasets
        pipeline.setup_data_splits()
        for dataset_type in ["original", "enhanced"]:
            results = pipeline.train_single_model(args.model, dataset_type)
            logger.info(f"Training completed: {results}")
    else:
        # Train all models
        results = pipeline.run_sequential_training()
        logger.info(f"All training completed: {len(results)} models trained")

if __name__ == "__main__":
    main()