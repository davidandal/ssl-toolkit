
import torch.nn as nn

# Tabular Model (MLP)
class TabularModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[64, 32], dropout=0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Model Factory
def model_factory(**kwargs):
    return TabularModel(**kwargs)
    