
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
    def __init__(self, categorical_columns, numeric_columns, target_column, is_target_categorical):
        self.categorical_cols = categorical_columns
        self.numeric_cols = numeric_columns
        self.target_col = target_column
        self.is_target_categorical = is_target_categorical

        self.cat_encoders = {col: LabelEncoder() for col in categorical_columns}
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame):
        for col in self.categorical_cols:
            self.cat_encoders[col].fit(df[col].astype(str))
        self.scaler.fit(df[self.numeric_cols])

    def transform(self, df: pd.DataFrame):
        cat_features = [self.cat_encoders[col].transform(df[col].astype(str)) 
                        for col in self.categorical_cols]
        cat_features = np.stack(cat_features, axis=1)

        num_features = self.scaler.transform(df[self.numeric_cols])
        X = np.concatenate([num_features, cat_features], axis=1)

        # Encode target column if it exists
        if self.target_col in df.columns:
            y = df[self.target_col]
            if self.is_target_categorical:
                y = y.astype(str)
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y)
                y_tensor = torch.tensor(y, dtype=torch.long)
            else:
                y = y.astype(float)
                y_tensor = torch.tensor(y, dtype=torch.float32)
    
            return torch.tensor(X, dtype=torch.float32), y_tensor
        else:
            return torch.tensor(X, dtype=torch.float32), None

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
    
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
