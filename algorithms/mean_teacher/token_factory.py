
import torch
import numpy as np
import pandas as pd

from torchvision import transforms
from transformers import AutoTokenizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

class ImageTokenizer:
    def __init__(self, image_size):
        self.transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    def __call__(self, dataset):
        if hasattr(dataset, "dataset"):
            dataset.dataset.transform = self.transforms
        else:
            dataset.transform = self.transforms
        return dataset

class TextTokenizer:
    def __init__(self, text_column, target_column, pretrained):
        self.text_column = text_column
        self.target_column = target_column
        self.pretrained = pretrained
        self.target_encoder = LabelEncoder()

        if pretrained:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        else:
            self.tokenizer = TfidfVectorizer(max_features=10000)

    def fit(self, dataframe):
        # No need to fit if the model used is pre-trained
        if not self.pretrained:
            self.tokenizer.fit(dataframe[self.text_column])

        self.target_encoder.fit(dataframe[self.target_column].astype(str))

    def transform(self, dataframe):
        if self.pretrained:
            return self.tokenizer(
                dataframe[self.text_column].astype(str).tolist(), padding=True, truncation=True, return_tensors="pt"
            )
        else:
            return torch.tensor(self.tokenizer.transform(dataframe[self.text_column]).toarray(), dtype=torch.float32)

    def transform_target(self, dataframe):
        return torch.tensor(self.target_encoder.transform(dataframe[self.target_column]), dtype=torch.long)

class TabularTokenizer:
    def __init__(self, categorical_columns, numeric_columns, target_column, is_target_categorical):
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.target_column = target_column
        self.is_target_categorical = is_target_categorical

        self.categorical_encoders = {col: LabelEncoder() for col in categorical_columns}
        self.scaler = StandardScaler()
        self.target_encoder = LabelEncoder() if is_target_categorical else None

    def fit(self, dataframe):
        for column in self.categorical_columns:
            self.categorical_encoders[column].fit(dataframe[column].astype(str))
        self.scaler.fit(dataframe[self.numeric_columns])

        # Only fit target if it is categorical
        if self.is_target_categorical:
            self.target_encoder.fit(dataframe[self.target_column].astype(str))

    def transform(self, dataframe):
        categorical_features = []
        for column in self.categorical_columns:
            column_data = dataframe[column].astype(str)
            known_classes = set(self.categorical_encoders[column].classes_)

            safe_column = column_data.apply(lambda x: x if x in known_classes else "<UNK>")
            if "<UNK>" not in self.categorical_encoders[column].classes_:
                self.categorical_encoders[column].classes_ = np.append(self.categorical_encoders[column].classes_, "<UNK>")

            encoded = self.categorical_encoders[column].transform(safe_column)
            categorical_features.append(encoded)

        categorical_features = np.stack(categorical_features, axis=1)
        num_features = self.scaler.transform(dataframe[self.numeric_columns])

        return np.concatenate([num_features, categorical_features], axis=1)
    
    def transform_target(self, dataframe):
        y = dataframe[self.target_column]
        if self.is_target_categorical:
            return self.target_encoder.transform(y.astype(str))
        else:
            return y.astype(float)

def token_factory(input_type, **kwargs):
    if input_type == "image":
        return ImageTokenizer(**kwargs)
    elif input_type == "text":
        return TextTokenizer(**kwargs)
    elif input_type == "tabular":
        return TabularTokenizer(**kwargs)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")
