
import torch
import torch.nn as nn
import torchvision.models as models

from transformers import AutoModel

# Image Model (ResNet or CNN)
class ImageModel(nn.Module):
    def __init__(self, num_classes, pretrained):
        super().__init__()

        self.pretrained = pretrained

        # Use pretrained ResNet18
        if self.pretrained:
            self.model = models.resnet18(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        # Use custom CNN
        else:
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
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

# Text Model (BERT)
class TextModel(nn.Module):
    def __init__(self, num_classes, pretrained, tfidf_dim):
        super().__init__()
        self.pretrained = pretrained

        if pretrained:
            self.bert = AutoModel.from_pretrained("bert-base-uncased")
            hidden_size = self.bert.config.hidden_size
            self.classifier = nn.Linear(hidden_size, num_classes)
        else:
            assert tfidf_dim is not None, "For non-pretrained, you must pass tfidf_dim"
            self.bert = None  # No encoder used
            self.classifier = nn.Linear(tfidf_dim, num_classes)

    def forward(self, x):
        if isinstance(x, dict) and "input_ids" in x and "attention_mask" in x:
            # Hugging Face tokenizer (pretrained transformers)
            input_ids = x["input_ids"]
            attention_mask = x["attention_mask"]
            token_type_ids = x.get("token_type_ids", None)  # Optional

            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            cls_token = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
            return self.classifier(cls_token)

        elif isinstance(x, torch.Tensor):
            # TF-IDF path (simple MLP or linear classifier)
            return self.classifier(x)

        else:
            raise TypeError("Unsupported input type for TextModel.forward")

# Tabular Model (MLP)
class TabularModel(nn.Module):
    def __init__(self, input_dim, num_classes, regression, hidden_dims=[64, 32], dropout=0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        output_dim = 1 if regression else num_classes
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Model Factory
def model_factory(input_type, **kwargs):
    if input_type == "image":
        return ImageModel(**kwargs)
    elif input_type == "text":
        return TextModel(**kwargs)
    elif input_type == "tabular":
        return TabularModel(**kwargs)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")