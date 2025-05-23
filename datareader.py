import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold 

class TransformSubset(torch.utils.data.Dataset):
    """
    Melakukan transformasi pada subset dataset.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform 
        self.classes = subset.dataset.classes
        self.class_to_idx = subset.dataset.class_to_idx

    def __getitem__(self, index): 
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)

def load_data(data_dir):
    return datasets.ImageFolder(root=data_dir, transform=transforms.Compose([
        transforms.Resize(224)
    ]))

def get_kfold_loaders(full_dataset, batch_size, input_size=224, k=5, seed=2003, num_workers=0):
    """
    Create K-Fold loaders for training and validation.
    """
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.0087, 0.0090, 0.0076], [0.0045, 0.0045, 0.0052])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.0087, 0.0090, 0.0076], [0.0045, 0.0045, 0.0052])
    ])

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    fold_loaders = []

    for train_idx, val_idx in kf.split(range(len(full_dataset))):
        train_subset = TransformSubset(
            Subset(full_dataset, train_idx), 
            train_transform
        )
        val_subset = TransformSubset(
            Subset(full_dataset, val_idx), 
            val_transform
        )
        
        train_loader = DataLoader(train_subset, 
                               batch_size=batch_size, 
                               shuffle=True,
                               num_workers=num_workers)
        val_loader = DataLoader(val_subset, 
                              batch_size=batch_size, 
                              shuffle=False,
                              num_workers=num_workers)
        
        fold_loaders.append((train_loader, val_loader))
    
    return fold_loaders
