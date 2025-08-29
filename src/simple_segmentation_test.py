#!/usr/bin/env python3
"""
Simple 3D Liver Segmentation Test using MONAI
Tests one model on both original and enhanced datasets.
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import logging
import nibabel as nib

# MONAI imports
import monai
from monai.data import Dataset, DataLoader, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, ToTensord, Resized, MapTransform
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConvertToMultiClassSegmentation(MapTransform):
    """
    Convert labels to proper format for multi-class segmentation.
    Assumes labels are: 0=background, 1=liver, 2=tumor
    """
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            # Keep original labels but ensure they're in correct range
            d[key] = torch.clamp(d[key], 0, 2).long()
        return d

def inspect_data_sample():
    """Inspect a sample to understand data format"""
    base_dir = Path("Dataset/Liver Segmentation")
    original_images_dir = base_dir / "imagesTr"
    labels_dir = base_dir / "labelsTr"
    
    # Load a sample
    sample_files = list(original_images_dir.glob("*.nii"))[:1]
    
    for img_file in sample_files:
        img_path = str(img_file)
        label_path = str(labels_dir / img_file.name)
        
        logger.info(f"Inspecting: {img_file.name}")
        
        # Load with nibabel
        img_nib = nib.load(img_path)
        label_nib = nib.load(label_path)
        
        img_data = img_nib.get_fdata()
        label_data = label_nib.get_fdata()
        
        logger.info(f"Image shape: {img_data.shape}, dtype: {img_data.dtype}")
        logger.info(f"Image range: [{img_data.min():.2f}, {img_data.max():.2f}]")
        logger.info(f"Label shape: {label_data.shape}, dtype: {label_data.dtype}")
        logger.info(f"Label unique values: {np.unique(label_data)}")
        
        # Test MONAI loading
        transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
        ])
        
        data_dict = {"image": img_path, "label": label_path}
        transformed = transforms(data_dict)
        
        logger.info(f"MONAI Image shape: {transformed['image'].shape}")
        logger.info(f"MONAI Label shape: {transformed['label'].shape}")
        logger.info(f"MONAI Label unique: {torch.unique(transformed['label'])}")

def simple_training_test():
    """Simple training test with one model"""
    logger.info("Starting simple training test...")
    
    # Set determinism
    set_determinism(seed=42)
    
    # Paths
    base_dir = Path("Dataset/Liver Segmentation")
    original_images_dir = base_dir / "imagesTr"
    enhanced_images_dir = base_dir / "imagesTr_3d_enhanced"
    labels_dir = base_dir / "labelsTr"
    results_dir = Path("simple_segmentation_results")
    results_dir.mkdir(exist_ok=True)
    
    # Get file list and create small subset for testing
    image_files = sorted([f.stem for f in original_images_dir.glob("*.nii")])
    random.seed(42)
    random.shuffle(image_files)
    
    # Use only first 20 files for quick test
    test_files = image_files[:20]
    train_files = test_files[:15]
    val_files = test_files[15:]
    
    logger.info(f"Using {len(train_files)} training, {len(val_files)} validation files")
    
    # Create data dictionaries
    def create_data_dicts(file_list, dataset_type="original"):
        data_dicts = []
        for file_stem in file_list:
            if dataset_type == "original":
                image_path = original_images_dir / f"{file_stem}.nii"
            else:
                image_path = enhanced_images_dir / f"{file_stem}_bbhe_3d_enhanced.nii"
            
            label_path = labels_dir / f"{file_stem}.nii"
            
            if image_path.exists() and label_path.exists():
                data_dicts.append({
                    "image": str(image_path),
                    "label": str(label_path)
                })
        return data_dicts
    
    # Simple transforms
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(2.0, 2.0, 3.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=1024, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(keys=["image", "label"], spatial_size=(64, 64, 64), mode=("trilinear", "nearest")),
        ConvertToMultiClassSegmentation(keys=["label"]),
        ToTensord(keys=["image", "label"]),
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(2.0, 2.0, 3.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=1024, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ConvertToMultiClassSegmentation(keys=["label"]),
        ToTensord(keys=["image", "label"]),
    ])
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    results = {}
    
    # Test both datasets
    for dataset_type in ["original", "enhanced"]:
        logger.info(f"\n{'='*40}")
        logger.info(f"Testing {dataset_type.upper()} dataset")
        logger.info(f"{'='*40}")
        
        # Create datasets
        train_data = create_data_dicts(train_files, dataset_type)
        val_data = create_data_dicts(val_files, dataset_type)
        
        train_ds = Dataset(data=train_data, transform=train_transforms)
        val_ds = Dataset(data=val_data, transform=val_transforms)
        
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)
        
        # Create model for 3-class segmentation (background, liver, tumor)
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,  # Changed to 3 classes
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(device)
        
        # Loss and optimizer
        loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        
        # Training loop (short)
        max_epochs = 10
        best_metric = -1
        train_losses = []
        val_dices = []
        
        for epoch in range(max_epochs):
            logger.info(f"Epoch {epoch + 1}/{max_epochs}")
            model.train()
            epoch_loss = 0
            step = 0
            
            for batch_data in train_loader:
                step += 1
                inputs, labels = (
                    batch_data["image"].to(device),
                    batch_data["label"].to(device),
                )
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if step % 5 == 0:
                    logger.info(f"Step {step}, Loss: {loss.item():.4f}")
            
            epoch_loss /= step
            train_losses.append(epoch_loss)
            logger.info(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            
            # Validation every 2 epochs
            if (epoch + 1) % 2 == 0:
                model.eval()
                with torch.no_grad():
                    for val_data in val_loader:
                        val_inputs, val_labels = (
                            val_data["image"].to(device),
                            val_data["label"].to(device),
                        )
                        
                        roi_size = (64, 64, 64)
                        val_outputs = sliding_window_inference(val_inputs, roi_size, 1, model)
                        
                        # Convert to predictions
                        val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)
                        dice_metric(y_pred=val_outputs, y=val_labels)
                
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                val_dices.append(metric)
                
                if metric > best_metric:
                    best_metric = metric
                    # Save best model
                    torch.save(model.state_dict(), results_dir / f"unet_{dataset_type}_best.pth")
                
                logger.info(f"Validation Dice: {metric:.4f}, Best: {best_metric:.4f}")
        
        results[dataset_type] = {
            "train_losses": train_losses,
            "val_dices": val_dices,
            "best_dice": best_metric
        }
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # Save results
    with open(results_dir / "simple_test_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training losses
    ax1.plot(results["original"]["train_losses"], label="Original", alpha=0.8)
    ax1.plot(results["enhanced"]["train_losses"], label="Enhanced", alpha=0.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Best Dice scores
    datasets = ["Original", "Enhanced"]
    dice_scores = [results["original"]["best_dice"], results["enhanced"]["best_dice"]]
    
    ax2.bar(datasets, dice_scores, alpha=0.8, color=['blue', 'orange'])
    ax2.set_ylabel("Best Dice Score")
    ax2.set_title("Best Dice Score Comparison")
    ax2.grid(True, alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(dice_scores):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(results_dir / "simple_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate report
    improvement = (results["enhanced"]["best_dice"] - results["original"]["best_dice"]) / results["original"]["best_dice"] * 100
    
    report = f"""
SIMPLE SEGMENTATION TEST RESULTS
================================

Dataset: Liver CT Segmentation
Model: 3D UNet (3-class: background, liver, tumor)
Training samples: {len(train_files)}
Validation samples: {len(val_files)}
Epochs: {max_epochs}

RESULTS:
--------
Original Dataset:
  Best Dice Score: {results["original"]["best_dice"]:.4f}
  Final Training Loss: {results["original"]["train_losses"][-1]:.4f}

Enhanced Dataset:
  Best Dice Score: {results["enhanced"]["best_dice"]:.4f}
  Final Training Loss: {results["enhanced"]["train_losses"][-1]:.4f}

IMPROVEMENT:
-----------
Dice Score Improvement: {improvement:+.2f}%

CONCLUSION:
----------
"""
    
    if improvement > 0:
        report += "✅ Enhancement improves segmentation performance\n"
    else:
        report += "❌ Enhancement does not improve performance\n"
    
    with open(results_dir / "simple_test_report.txt", 'w') as f:
        f.write(report)
    
    logger.info("Simple test completed!")
    logger.info(f"Original best Dice: {results['original']['best_dice']:.4f}")
    logger.info(f"Enhanced best Dice: {results['enhanced']['best_dice']:.4f}")
    logger.info(f"Improvement: {improvement:+.2f}%")
    
    return results

if __name__ == "__main__":
    # First inspect data
    logger.info("Inspecting data sample...")
    inspect_data_sample()
    
    # Then run simple test
    logger.info("\nRunning simple training test...")
    results = simple_training_test()