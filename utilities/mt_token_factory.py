
import torch
from torchvision import transforms
from transformers import AutoTokenizer
import numpy as np

class ImageTokenizer:
    def __init__(self, image_size=(224, 224)):
        self.transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    def __call__(self, image):
        return self.transforms(image)

class TextTokenizer:
    ...
    # def __init__(self, model_name='bert-base-uncased'):
    #     self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    # def __call__(self, text):
    #     tokens = self.tokenizer(
    #         text,
    #         return_tensors='pt',
    #         padding='max_length',
    #         truncation=True,
    #         max_length=128
    #     )
    #     return tokens.input_ids.squeeze(0), tokens.attention_mask.squeeze(0)

class TabularTokenizer:
    ...
    # def __init__(self, scaler=None, encoder=None):
    #     self.scaler = scaler
    #     self.encoder = encoder

    # def __call__(self, row):
    #     numeric = self.scaler.transform([row['numerical']])
    #     categorical = self.encoder.transform([row['categorical']])
    #     features = np.concatenate([numeric, categorical], axis=1)
    #     return torch.tensor(features, dtype=torch.float32)

def token_factory(input_type, **kwargs):
    if input_type == "image":
        return ImageTokenizer(**kwargs)
    elif input_type == "text":
        return TextTokenizer(**kwargs)
    elif input_type == "tabular":
        return TabularTokenizer(**kwargs)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")
