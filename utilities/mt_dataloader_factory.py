# dataloader_factory.py (Config-Driven Version with Separate Real SSL Support)

import os
import random
import torch

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image

# Dataloader Classes
class SemiSupervisedDataset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform:
            x = self.transform(x)
        return x, y

class UnlabeledDataset(Dataset):
    def __init__(self, dataset, indices, weak_transform=None, strong_transform=None):
        self.dataset = dataset
        self.indices = indices
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, _ = self.dataset[self.indices[idx]]
        x_w = self.weak_transform(x)
        x_s = self.strong_transform(x)
        return x_w, x_s

class UnlabeledFolderDataset(Dataset):
    def __init__(self, root, weak_transform, strong_transform):
        self.paths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif'))]
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.weak_transform(img), self.strong_transform(img)

# Transform Functions
class TabularWeakTransform:
    def __init__(self, noise_std=0.01):
        self.noise_std = noise_std

    def __call__(self, x):
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

class TabularStrongTransform:
    def __init__(self, noise_std=0.05, mask_ratio=0.1):
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio

    def __call__(self, x):
        # Add stronger Gaussian noise
        x_aug = x + torch.randn_like(x) * self.noise_std

        # Randomly mask features
        mask = torch.rand_like(x_aug) < self.mask_ratio
        x_aug[mask] = 0  # or use mean imputation if needed

        return x_aug

class TabularValTransform:
    def __call__(self, x):
        return x  # No augmentation for validation

# Dataloader Functions
def create_image_dataloaders(config, base_transform):
    num_labels_per_class = config["num_labels"] // len(config["image_classes"])
    batch_size = config.get("batch_size", 64)

    # Define transforms
    transform_weak = transforms.Compose([
        base_transform,
        transforms.RandomHorizontalFlip(),
    ])
    transform_strong = transforms.Compose([
        base_transform,
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.RandomGrayscale(p=0.2),
    ])
    transform_val = base_transform

    # Load dataset and create classwise indices
    full_dataset = ImageFolder(root=config["dataset_path"])
    class_to_idx = {cls: i for cls, i in full_dataset.class_to_idx.items() if cls in config["image_classes"]}
    selected_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label in class_to_idx.values()]

    # Limit to ~300 per class
    classwise_limited_indices = {cls: [] for cls in class_to_idx.values()}
    for idx in selected_indices:
        _, label = full_dataset.samples[idx]
        if len(classwise_limited_indices[label]) < 1000:
            classwise_limited_indices[label].append(idx)
    selected_indices = [i for sublist in classwise_limited_indices.values() for i in sublist]

    # Stratified split into train_pool and val_indices
    all_labels = [full_dataset.samples[i][1] for i in selected_indices]
    train_pool_indices, val_indices = train_test_split(
        selected_indices,
        test_size=0.2,
        stratify=all_labels,
        random_state=42
    )

    classwise_indices = {cls_idx: [] for cls_idx in class_to_idx.values()}
    for i in train_pool_indices:
        _, label = full_dataset.samples[i]
        classwise_indices[label].append(i)

    # Split indices into labeled and unlabeled
    lb_indices = []
    ulb_indices = []
    for _, indices in classwise_indices.items():
        random.shuffle(indices)
        lb = indices[:num_labels_per_class]
        ulb = indices[num_labels_per_class:]
        lb_indices.extend(lb)
        ulb_indices.extend(ulb)

    # Instatiate datasets and dataloaders
    lb_dataset = SemiSupervisedDataset(full_dataset, lb_indices, transform_weak)
    ulb_dataset = UnlabeledDataset(full_dataset, ulb_indices, transform_weak, transform_strong)
    val_dataset = SemiSupervisedDataset(full_dataset, val_indices, transform_val)

    lb_loader = DataLoader(lb_dataset, batch_size=batch_size, shuffle=True)
    ulb_loader = DataLoader(ulb_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return lb_loader, ulb_loader, val_loader

def create_tabular_dataloaders(config, X, y):
    num_labels = config.get("num_labels", 400)
    batch_size = config.get("batch_size", 64)

    # Stratified sampling for labeled data
    stratifier = StratifiedShuffleSplit(n_splits=1, train_size=num_labels, random_state=42)
    lb_idx, rest_idx = next(stratifier.split(X, y))

    # From the rest, sample a validation set (same size as lb_idx)
    val_size = 1 - num_labels
    stratifier_val = StratifiedShuffleSplit(n_splits=1, train_size=val_size, random_state=99)
    val_idx, ulb_idx = next(stratifier_val.split(X[rest_idx], y[rest_idx]))
    val_idx = rest_idx[val_idx]
    ulb_idx = rest_idx[ulb_idx]

    base_dataset = TensorDataset(X, y)

    transform_weak = TabularWeakTransform(noise_std=0.01)
    transform_strong = TabularStrongTransform(noise_std=0.05, mask_ratio=0.1)
    transform_val = TabularValTransform()

    lb_dataset = SemiSupervisedDataset(base_dataset, lb_idx, transform_weak)
    ulb_dataset = UnlabeledDataset(base_dataset, ulb_idx, transform_weak, transform_strong)
    val_dataset = SemiSupervisedDataset(base_dataset, val_idx, transform_val)

    lb_loader = DataLoader(lb_dataset, batch_size=batch_size, shuffle=True)
    ulb_loader = DataLoader(ulb_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return lb_loader, ulb_loader, val_loader

# Factory
def dataloader_factory(config, **kwargs):
    if config["input_type"] == "image":
        return create_image_dataloaders(config, **kwargs)
    elif config["input_type"] == "tabular":
        return create_tabular_dataloaders(config, **kwargs)       

def real_dataloader_factory(config):
    labeled_path = config["labeled_path"]
    unlabeled_path = config["unlabeled_path"]
    batch_size = config.get("batch_size", 64)

    transform_weak = config["transform_weak"]
    transform_strong = config["transform_strong"]
    transform_val = config["transform_val"]

    lb_dataset = ImageFolder(root=labeled_path, transform=transform_weak)
    ulb_dataset = UnlabeledFolderDataset(unlabeled_path, weak_transform=transform_weak, strong_transform=transform_strong)
    val_dataset = ImageFolder(root=labeled_path, transform=transform_val)  # Optional: separate val folder if needed

    lb_loader = DataLoader(lb_dataset, batch_size=batch_size, shuffle=True)
    ulb_loader = DataLoader(ulb_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return lb_loader, ulb_loader, val_loader
