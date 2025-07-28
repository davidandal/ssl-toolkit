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
        
def dataloader_factory(X_train, y_train, X_val, y_val, X_unlabeled):
    train_dataset = TabularDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TabularDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    unlabeled_dataset = TabularDataset(torch.tensor(X_unlabeled, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) 
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, unlabeled_loader

def combined_dataloader_factory(X_combined, y_combined):
    combined_dataset = TabularDataset(torch.tensor(X_combined, dtype=torch.float32), torch.tensor(y_combined, dtype=torch.long))
    return DataLoader(combined_dataset, batch_size=32, shuffle=True)
