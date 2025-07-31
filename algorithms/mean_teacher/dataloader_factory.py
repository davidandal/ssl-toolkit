
import torch

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from transformers.tokenization_utils_base import BatchEncoding
from scipy.sparse import issparse

# Dataloader Classes
class GeneralLabeledDataset(Dataset):
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

class GeneralUnlabeledDataset(Dataset):
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
        elif isinstance(self.encodings, torch.Tensor):
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
class TextWeakTransform:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = 128

    def __call__(self, text):
        # TF-IDF case (tokenizer is not pre-trained)
        if isinstance(self.tokenizer.tokenizer, TfidfVectorizer):
            vector = self.tokenizer.tokenizer.transform([text]).toarray()[0]
            return torch.tensor(vector, dtype=torch.float32)
        # Tokenizer is pre-trained
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
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = 128

    def synonym_replace(self, text):
        # TODO: Add a synonym replacement logic
        return text

    def __call__(self, text):
        augmented_text = self.synonym_replace(text)

        # Tokenizer is not pre-trained
        if isinstance(self.tokenizer.tokenizer, TfidfVectorizer):
            vector = self.tokenizer.tokenizer.transform([text]).toarray()[0]
            return torch.tensor(vector, dtype=torch.float32)
        # Tokenizer is pre-trained
        else:
            tokens = self.tokenizer.tokenizer(
                augmented_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            return {k: v.squeeze(0) for k, v in tokens.items()}

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
        x_aug = x + torch.randn_like(x) * self.noise_std

        mask = torch.rand_like(x_aug) < self.mask_ratio
        x_aug[mask] = 0

        return x_aug

class TabularValTransform:
    def __call__(self, x):
        return x  # No augmentation for validation

# Dataloader Functions
def create_image_dataloaders(train, validation, unlabeled, batch_size):
    # Define transforms
    transform_weak = transforms.Compose([
        transforms.RandomHorizontalFlip(),
    ])
    transform_strong = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.RandomGrayscale(p=0.2),
    ])

    # Instatiate datasets and dataloaders
    labeled_dataset = GeneralLabeledDataset(train, transform_weak)
    validation_dataset = GeneralLabeledDataset(unlabeled)
    unlabeled_dataset = GeneralUnlabeledDataset(validation, transform_weak, transform_strong)

    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)

    return labeled_loader, unlabeled_loader, validation_loader

def create_text_dataloaders(X_train, y_train, X_validation, y_validation, X_unlabeled, tokenizer, batch_size):
    # TF-IDF case (not pre-trained), convert to tensors
    if issparse(X_train):
        X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
        y_train = torch.tensor(y_train)

        X_validation = torch.tensor(X_validation.toarray(), dtype=torch.float32)
        y_validation = torch.tensor(y_validation)

    # Define transforms
    weak_transform = TextWeakTransform(tokenizer)
    strong_transform = TextStrongTransform(tokenizer)

    # Instatiate datasets and dataloaders
    labeled_dataset = LabeledTextDataset(X_train, y_train)
    validation_dataset = LabeledTextDataset(X_validation, y_validation)
    unlabeled_dataset = UnlabeledTextDataset(X_unlabeled, weak_transform, strong_transform)

    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
    validation_laoder = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)

    return labeled_loader, unlabeled_loader, validation_laoder

def create_tabular_dataloaders(X_train, y_train, X_validation, y_validation, X_unlabeled, batch_size):
    base_labeled_dataset = TensorDataset(X_train, y_train)
    base_validation_dataset = TensorDataset(X_validation, y_validation)
    base_unlabeled_dataset = TensorDataset(X_unlabeled)

    weak_transform = TabularWeakTransform(0.01)
    strong_transform = TabularStrongTransform(0.05, 0.1)

    labeled_dataset = GeneralLabeledDataset(base_labeled_dataset, weak_transform)
    validation_dataset = GeneralLabeledDataset(base_validation_dataset)
    unlabeled_dataset = GeneralUnlabeledDataset(base_unlabeled_dataset, weak_transform, strong_transform)

    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)

    return labeled_loader, unlabeled_loader, validation_loader

# Factory
def dataloader_factory(input_type, **kwargs):
    if input_type == "image":
        return create_image_dataloaders(**kwargs)
    elif input_type == "text":
        return create_text_dataloaders(**kwargs)
    elif input_type == "tabular":
        return create_tabular_dataloaders(**kwargs)    
       
