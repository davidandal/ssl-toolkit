
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

class TabularTokenizer:
    def __init__(self, categorical_columns, numeric_columns, target_column):
        self.categorical_cols = categorical_columns
        self.numeric_cols = numeric_columns
        self.target_col = target_column

        self.cat_encoders = {col: LabelEncoder() for col in categorical_columns}
        self.scaler = StandardScaler()
        self.target_encoder = LabelEncoder()

    def fit(self, df):
        for col in self.categorical_cols:
            self.cat_encoders[col].fit(df[col].astype(str))
        self.scaler.fit(df[self.numeric_cols])

        self.target_encoder.fit(df[self.target_col])

    def transform(self, df):
        cat_features = []
        for col in self.categorical_cols:
            col_data = df[col].astype(str)
            known_classes = set(self.cat_encoders[col].classes_)

            safe_col = col_data.apply(lambda x: x if x in known_classes else "<UNK>")

            if "<UNK>" not in self.cat_encoders[col].classes_:
                self.cat_encoders[col].classes_ = np.append(self.cat_encoders[col].classes_, "<UNK>")

            encoded = self.cat_encoders[col].transform(safe_col)
            cat_features.append(encoded)

        cat_features = np.stack(cat_features, axis=1)
        num_features = self.scaler.transform(df[self.numeric_cols])

        return np.concatenate([num_features, cat_features], axis=1)

    def transform_target(self, df):
        return self.target_encoder.transform(df[self.target_col])
        
def token_factory(input_type, **kwargs):
    if input_type == "image":
        return ImageTokenizer(**kwargs)
    elif input_type == "text":
        return TextTokenizer(**kwargs)
    elif input_type == "tabular":
        return TabularTokenizer(**kwargs)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")
