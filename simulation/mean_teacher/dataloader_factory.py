
import random
import torch
import random

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset, Subset, DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers.tokenization_utils_base import BatchEncoding
from scipy.sparse import issparse

# Dataloader Classes
class LabeledFolderDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

class UnlabeledFolderDataset(Dataset):
    def __init__(self, dataset, weak_transform=None, strong_transform=None):
        self.dataset = dataset
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx][0]  # Safely get only x
        x_w = self.weak_transform(x)
        x_s = self.strong_transform(x)
        return x_w, x_s

class LabeledTextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        if isinstance(self.encodings, BatchEncoding) or isinstance(self.encodings, dict):
            item = {k: v[idx] for k, v in self.encodings.items()}
        elif isinstance(self.encodings, torch.Tensor):  # For TF-IDF tensors
            item = self.encodings[idx]
        else:
            raise ValueError("Unsupported encoding type")

        label = torch.tensor(self.labels[idx]) if not isinstance(self.labels[idx], torch.Tensor) else self.labels[idx]
        return item, label

    def __len__(self):
        return len(self.labels)

class UnlabeledTextDataset(Dataset):
    def __init__(self, texts, weak_transform, strong_transform):
        self.texts = texts
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __getitem__(self, idx):
        text = self.texts[idx]

        weak = self.weak_transform(text)
        strong = self.strong_transform(text)

        return weak, strong

    def __len__(self):
        return len(self.texts)
    
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

class TextWeakTransform:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, text):
        if isinstance(self.tokenizer.tokenizer, TfidfVectorizer):
            vector = self.tokenizer.tokenizer.transform([text]).toarray()[0]
            return torch.tensor(vector, dtype=torch.float32)
        else:
            tokens = self.tokenizer.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            return {k: v.squeeze(0) for k, v in tokens.items()}

class TextStrongTransform:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def synonym_replace(self, text, p=0.1):
        # Optional: Add a real synonym replacement logic using WordNet or external API
        return text  # ← Just placeholder for now

    def __call__(self, text):
        # Strong transformation (e.g., synonym replacement)
        aug_text = self.synonym_replace(text)

        if isinstance(self.tokenizer.tokenizer, TfidfVectorizer):
            vector = self.tokenizer.tokenizer.transform([text]).toarray()[0]
            return torch.tensor(vector, dtype=torch.float32)
        else:
            tokens = self.tokenizer.tokenizer(
                aug_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            return {k: v.squeeze(0) for k, v in tokens.items()}

# Dataloader Functions
def create_image_dataloaders(num_labels, image_classes, dataset_path, base_transform, batch_size=64):
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
    full_dataset = ImageFolder(root=dataset_path)
    num_labels_per_class = int(num_labels * len(full_dataset) / len(image_classes))
    class_to_idx = {cls: i for cls, i in full_dataset.class_to_idx.items() if cls in image_classes}
    selected_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label in class_to_idx.values()]

    # Limit to 1000 examples per class
    classwise_limited_indices = {cls: [] for cls in class_to_idx.values()}
    for idx in selected_indices:
        _, label = full_dataset.samples[idx]
        if len(classwise_limited_indices[label]) < 2000:
            classwise_limited_indices[label].append(idx)
    selected_indices = [i for sublist in classwise_limited_indices.values() for i in sublist]

    # Stratified split of labeled dataset into train_pool and val_indices
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
    lb_dataset = LabeledFolderDataset(Subset(full_dataset, lb_indices), transform_weak)
    ulb_dataset = UnlabeledFolderDataset(Subset(full_dataset, ulb_indices), transform_weak, transform_strong)
    val_dataset = LabeledFolderDataset(Subset(full_dataset, val_indices), transform_val)

    lb_loader = DataLoader(lb_dataset, batch_size=batch_size, shuffle=True)
    ulb_loader = DataLoader(ulb_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return lb_loader, ulb_loader, val_loader

def create_text_dataloaders(X_train, y_train, X_val, y_val, X_unlabeled, tokenizer, batch_size=64):
    # TF-IDF case (X_train is sparse)
    is_tfidf = issparse(X_train)
    if is_tfidf:
        X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
        y_train = torch.tensor(y_train)

        X_val = torch.tensor(X_val.toarray(), dtype=torch.float32)
        y_val = torch.tensor(y_val)

        # X_unlabeled = torch.tensor(X_unlabeled.toarray(), dtype=torch.float32)

    # Weak and strong transforms
    weak_transform = TextWeakTransform(tokenizer)
    strong_transform = TextStrongTransform(tokenizer)

    # Datasets
    lb_dataset = LabeledTextDataset(X_train, y_train)
    val_dataset = LabeledTextDataset(X_val, y_val)
    ulb_dataset = UnlabeledTextDataset(X_unlabeled, weak_transform, strong_transform)

    # Data loaders
    lb_loader = DataLoader(lb_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    ulb_loader = DataLoader(ulb_dataset, batch_size=batch_size, shuffle=True)

    return lb_loader, val_loader, ulb_loader

def create_tabular_dataloaders(X_train, y_train, X_val, y_val, X_unlabeled, batch_size=64):
    base_lb_dataset = TensorDataset(X_train, y_train)
    base_val_dataset = TensorDataset(X_val, y_val)
    base_ulb_dataset = TensorDataset(X_unlabeled)

    transform_weak = TabularWeakTransform(noise_std=0.01)
    transform_strong = TabularStrongTransform(noise_std=0.05, mask_ratio=0.1)
    transform_val = TabularValTransform()

    lb_dataset = LabeledFolderDataset(base_lb_dataset, transform=transform_weak)
    val_dataset = LabeledFolderDataset(base_val_dataset, transform=transform_val)
    ulb_dataset = UnlabeledFolderDataset(base_ulb_dataset, transform_weak, transform_strong)

    lb_loader = DataLoader(lb_dataset, batch_size=batch_size, shuffle=True)
    ulb_loader = DataLoader(ulb_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return lb_loader, ulb_loader, val_loader

# Factory
def dataloader_factory(input_type, **kwargs):
    if input_type == "image":
        return create_image_dataloaders(**kwargs)
    elif input_type == "text":
        return create_text_dataloaders(**kwargs)
    elif input_type == "tabular":
        return create_tabular_dataloaders(**kwargs)    
       
