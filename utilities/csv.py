import pandas as pd
from sklearn.model_selection import train_test_split

FULL_CSV_PATH = "../datasets/text_2.csv"
LABEL_COLUMN = "product"
LABELED_CSV_PATH = "../datasets/text_2_labeled.csv"
UNLABELED_CSV_PATH = "../datasets/text_2_unlabeled.csv"
UNLABELED_RATIO = 0.6
RANDOM_STATE = 69

df = pd.read_csv(FULL_CSV_PATH)

df_unlabeled, df_labeled = train_test_split(
    df,
    test_size=(1 - UNLABELED_RATIO),
    stratify=df[LABEL_COLUMN],
    random_state=RANDOM_STATE
)

df_unlabeled[LABEL_COLUMN] = ""

df_labeled.to_csv(LABELED_CSV_PATH, index=False)
df_unlabeled.to_csv(UNLABELED_CSV_PATH, index=False)

print("✅ Split complete. 'labeled.csv' and 'unlabeled.csv' craeted.")
