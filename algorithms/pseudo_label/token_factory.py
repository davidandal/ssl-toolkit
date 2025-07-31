
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class TabularTokenizer:
    def __init__(self, categorical_columns, numeric_columns, target_column):
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.target_columns = target_column

        self.categorical_encoders = {col: LabelEncoder() for col in categorical_columns}
        self.target_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def fit(self, dataframe):
        for col in self.categorical_columns:
            self.categorical_encoders[col].fit(dataframe[col].astype(str))
        self.scaler.fit(dataframe[self.numeric_columns])

        self.target_encoder.fit(dataframe[self.target_columns])

    def transform(self, dataframe):
        categorical_features = []
        for column in self.categorical_columns:
            column_data = dataframe[column].astype(str)
            known_classes = set(self.categorical_encoders[column].classes_)

            safe_col = column_data.apply(lambda x: x if x in known_classes else "<UNK>")
            if "<UNK>" not in self.categorical_encoders[column].classes_:
                self.categorical_encoders[column].classes_ = np.append(self.categorical_encoders[column].classes_, "<UNK>")

            encoded = self.categorical_encoders[column].transform(safe_col)
            categorical_features.append(encoded)

        categorical_features = np.stack(categorical_features, axis=1)
        num_features = self.scaler.transform(dataframe[self.numeric_columns])

        return np.concatenate([num_features, categorical_features], axis=1)

    def transform_target(self, dataframe):
        return self.target_encoder.transform(dataframe[self.target_columns])
        
def token_factory(**kwargs):
    return TabularTokenizer(**kwargs)
