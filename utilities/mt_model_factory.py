
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel

# Image Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()

        self.pretrained = pretrained

        if self.pretrained:
            # Use pretrained ResNet18
            self.model = models.resnet18(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        else:
            # Use custom CNN
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 224 → 112

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 112 → 56

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 56 → 28
            )
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(128 * 28 * 28, num_classes)

    def forward(self, x):
        if self.pretrained:
            return self.model(x)
        else:
            x = self.model(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            return self.fc(x)

# Text Model
class BERTClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_classes=4):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(output.pooler_output)

# Tabular Model
class TabularMLP(nn.Module):
    ...
    # def __init__(self, input_dim=10, num_classes=4):
    #     super().__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(input_dim, 64),
    #         nn.ReLU(),
    #         nn.Linear(64, 64),
    #         nn.ReLU(),
    #         nn.Linear(64, num_classes)
    #     )

    # def forward(self, x):
    #     return self.net(x)

# Model Factory
def model_factory(input_type, **kwargs):
    if input_type == 'image':
        return SimpleCNN(**kwargs)
    elif input_type == 'text':
        return BERTClassifier(**kwargs)
    elif input_type == 'tabular':
        return TabularMLP(**kwargs)
    else:
        raise ValueError(f'Unsupported input type: {input_type}')