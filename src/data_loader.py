#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loading and preprocessing utilities for the ExplainableCropAI project.

This module handles loading and preprocessing of paddy disease images,
including data augmentation and normalization.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np


class PaddyDiseaseDataset(Dataset):
    """
    Dataset class for paddy disease images.
    
    Supports loading images from a directory structure or from a CSV file
    with image paths and labels.
    """
    
    def __init__(self, root_dir=None, csv_file=None, transform=None, mode='train'):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Root directory of the dataset.
            csv_file (str): Path to CSV file with image paths and labels.
            transform (callable): Optional transform to be applied to images.
            mode (str): 'train', 'val', or 'test' mode.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        if csv_file is not None:
            self.data = pd.read_csv(csv_file)
            self.image_paths = self.data['image_path'].values
            self.labels = self.data['label'].values
            self.class_names = sorted(self.data['label'].unique())
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
            self.label_indices = np.array([self.class_to_idx[label] for label in self.labels])
        else:
            # Load from directory structure
            self.class_names = sorted(os.listdir(root_dir))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
            
            self.image_paths = []
            self.label_indices = []
            
            for class_idx, class_name in enumerate(self.class_names):
                class_dir = os.path.join(root_dir, class_name)
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if os.path.isfile(img_path) and self._is_image_file(img_name):
                        self.image_paths.append(img_path)
                        self.label_indices.append(class_idx)
            
            self.label_indices = np.array(self.label_indices)
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            dict: Sample containing image, label, path, and original image.
        """
        img_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Keep original image for visualization
        original_image = image.copy()
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.label_indices[idx]
        
        # Return as a dictionary for flexibility
        sample = {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'path': img_path,
            'original_image': original_image,
            'class_name': self.class_names[label]
        }
        
        return sample
    
    def _is_image_file(self, filename):
        """Check if a file is an image based on its extension."""
        img_extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif']
        return any(filename.lower().endswith(ext) for ext in img_extensions)


def get_transforms(mode='train', input_size=224):
    """
    Get transforms for different dataset modes.
    
    Args:
        mode (str): 'train', 'val', or 'test' mode.
        input_size (int): Target input size for the model.
        
    Returns:
        transforms.Compose: Composed transforms.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(input_size * 1.14)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize
        ])


def create_data_loaders(data_dir=None, csv_file=None, batch_size=32, num_workers=4, input_size=224):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir (str): Root directory of the dataset.
        csv_file (str): Path to CSV file with image paths and labels.
        batch_size (int): Batch size for data loaders.
        num_workers (int): Number of workers for data loading.
        input_size (int): Target input size for the model.
        
    Returns:
        dict: Data loaders for 'train', 'val', and 'test'.
    """
    train_transform = get_transforms(mode='train', input_size=input_size)
    val_transform = get_transforms(mode='val', input_size=input_size)
    test_transform = get_transforms(mode='test', input_size=input_size)
    
    data_loaders = {}
    
    if data_dir:
        # Assuming train/val/test directory structure
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        test_dir = os.path.join(data_dir, 'test')
        
        if os.path.exists(train_dir):
            train_dataset = PaddyDiseaseDataset(root_dir=train_dir, transform=train_transform, mode='train')
            data_loaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        if os.path.exists(val_dir):
            val_dataset = PaddyDiseaseDataset(root_dir=val_dir, transform=val_transform, mode='val')
            data_loaders['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        if os.path.exists(test_dir):
            test_dataset = PaddyDiseaseDataset(root_dir=test_dir, transform=test_transform, mode='test')
            data_loaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    elif csv_file:
        # Load from CSV file
        data = pd.read_csv(csv_file)
        
        if 'split' in data.columns:
            # Split based on the 'split' column
            train_data = data[data['split'] == 'train']
            val_data = data[data['split'] == 'val']
            test_data = data[data['split'] == 'test']
            
            train_csv = csv_file.replace('.csv', '_train.csv')
            val_csv = csv_file.replace('.csv', '_val.csv')
            test_csv = csv_file.replace('.csv', '_test.csv')
            
            train_data.to_csv(train_csv, index=False)
            val_data.to_csv(val_csv, index=False)
            test_data.to_csv(test_csv, index=False)
            
            train_dataset = PaddyDiseaseDataset(csv_file=train_csv, transform=train_transform, mode='train')
            val_dataset = PaddyDiseaseDataset(csv_file=val_csv, transform=val_transform, mode='val')
            test_dataset = PaddyDiseaseDataset(csv_file=test_csv, transform=test_transform, mode='test')
            
            data_loaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            data_loaders['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            data_loaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return data_loaders


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Sample data directory (assuming it exists)
    sample_data_dir = "../data/paddy_disease"
    
    if os.path.exists(sample_data_dir):
        loaders = create_data_loaders(data_dir=sample_data_dir, batch_size=4)
        
        if 'train' in loaders:
            train_loader = loaders['train']
            batch = next(iter(train_loader))
            
            images = batch['image']
            labels = batch['label']
            class_names = [batch['class_name'][i] for i in range(len(labels))]
            
            print(f"Batch shape: {images.shape}")
            print(f"Labels: {labels}")
            print(f"Class names: {class_names}")
            
            # Visualize a few samples
            fig, axes = plt.subplots(1, 4, figsize=(12, 3))
            for i in range(4):
                # Convert tensor to numpy and transpose to (H, W, C)
                img = images[i].numpy().transpose(1, 2, 0)
                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                axes[i].imshow(img)
                axes[i].set_title(class_names[i])
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig('../results/sample_batch.png')
            print("Sample batch visualization saved to ../results/sample_batch.png")
    else:
        print(f"Sample data directory {sample_data_dir} not found. Please update the path.")
