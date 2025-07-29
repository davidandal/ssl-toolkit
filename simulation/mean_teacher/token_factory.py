
import torch
import numpy as np
import pandas as pd

from torchvision import transforms
from transformers import AutoTokenizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

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
    def __init__(self, text_column, target_column, pretrained=True, model_name="bert-base-uncased"):
        self.pretrained = pretrained
        self.text_col = text_column
        self.target_col = target_column

        self.target_encoder = LabelEncoder()

        if pretrained:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = TfidfVectorizer(max_features=10000)

    def fit(self, df):
        if not self.pretrained:
            self.tokenizer.fit(df[self.text_col])
        self.target_encoder.fit(df[self.target_col].astype(str))

    def transform(self, df):
        if self.pretrained:
            return self.tokenizer(
                df[self.text_col].astype(str).tolist(), padding=True, truncation=True, return_tensors="pt"
            )
        else:
            return torch.tensor(self.tokenizer.transform(df[self.text_col]).toarray(), dtype=torch.float32)

    def transform_target(self, df):
        return torch.tensor(self.target_encoder.transform(df[self.target_col]), dtype=torch.long)

class TabularTokenizer:
    def __init__(self, categorical_columns, numeric_columns, target_column, is_target_categorical):
        self.categorical_cols = categorical_columns
        self.numeric_cols = numeric_columns
        self.target_col = target_column
        self.is_target_categorical = is_target_categorical

        self.cat_encoders = {col: LabelEncoder() for col in categorical_columns}
        self.target_encoder = LabelEncoder() if is_target_categorical else None
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame):
        for col in self.categorical_cols:
            self.cat_encoders[col].fit(df[col].astype(str))
        self.scaler.fit(df[self.numeric_cols])

        if self.is_target_categorical:
            self.target_encoder.fit(df[self.target_col].astype(str))

    def transform(self, df: pd.DataFrame):
        # cat_features = [self.cat_encoders[col].transform(df[col].astype(str)) 
        #                 for col in self.categorical_cols]
        # cat_features = np.stack(cat_features, axis=1)

        # num_features = self.scaler.transform(df[self.numeric_cols])
        # X = np.concatenate([num_features, cat_features], axis=1)

        # # Encode target column if it exists
        # if self.target_col in df.columns:
        #     y = df[self.target_col]
        #     if self.is_target_categorical:
        #         y = y.astype(str)
        #         target_encoder = LabelEncoder()
        #         y = target_encoder.fit_transform(y)
        #         y_tensor = torch.tensor(y, dtype=torch.long)
        #     else:
        #         y = y.astype(float)
        #         y_tensor = torch.tensor(y, dtype=torch.float32)
    
        #     return torch.tensor(X, dtype=torch.float32), y_tensor
        # else:
        #     return torch.tensor(X, dtype=torch.float32), None
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
        y = df[self.target_col]
        if self.is_target_categorical:
            return self.target_encoder.transform(y.astype(str))
        else:
            return y.astype(float)
    
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
