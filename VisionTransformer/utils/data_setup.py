import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from pathlib import Path


def create_dataloader(train_dir:Path, 
                      test_dir:Path,
                      batch_size:int = 32):
    """This function takes the images from train and test dirs and create dataloaders 
    from"""

    train_images_folder = datasets.ImageFolder(
        root=train_dir,
        transform=transforms.Compose,
    )

    test_image_folder = datasets.ImageFolder(
        root=test_dir,
        transform=transforms.Compose,
    )

    #Extract class names
    class_names = train_images_folder.classes
    
    #create dataloaders

    train_dataloader = DataLoader(
        dataset=train_images_folder,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        root=test_image_folder,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names

