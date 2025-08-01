import torch

from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]
        
def dataloader_factory(X_train, y_train, X_validation, y_validation, X_unlabeled, batch_size):
    labeled_dataset = TabularDataset(X_train, y_train)
    validation_dataset = TabularDataset(X_validation, y_validation)
    unlabeled_dataset = TabularDataset(X_unlabeled)

    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False) 
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)

    return labeled_loader, unlabeled_loader, validation_loader

def combined_dataloader_factory(X_combined, y_combined, batch_size):
    combined_dataset = TabularDataset(X_combined, y_combined)
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
