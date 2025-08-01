# Mean Teacher Algorithm

A modular implementation of the **Mean Teacher algorithm** for training models using a mix of labeled and unlabeled data.  
This version supports **image**, **text**, and **tabular** inputs, making it adaptable to a wide range of tasks such as classfication and regression (for tabular inputs only).

---

## How to Use

This notebook trains a model using the Mean Teacher algorithm, based on your specified data type and configuration, in just 2 easy steps.

### 1. Setup Input

Prepare and place in the repository two datasets (labeled and unlabeled) in the following formats depending on the input type:

| Input Type | Labeled Data Format | Unlabeled Data Format |
|------------|----------------------|------------------------|
| **Image**  | Folder with subfolders (each subfolder is a class containing images) | Folder with one subfolder containing all unlabeled images |
| **Text**   | `.csv` file with one column for text and one for the label | `.csv` file with one column for text and another empty column for label |
| **Tabular**| `.csv` file with feature columns and a target column | `.csv` file with same feature columns and an empty target column |

### 2. Configure

Edit the `config` dictionary at the top of the notebook to match your dataset and training preferences.

| Variable | Description | Sample Value | Notes |
|----------|-------------|--------------|-------|
| `training_session` | Identifier for saving model outputs | `1` |  |
| `seed` | Random seed for reproducibility | `27` |  |
| `pre_trained` | Whether to use a pretrained model (e.g., BERT or ResNet) | `False` | Not applicable to tabular inputs |
| `learning_rate` | Learning rate for the optimizer | `3e-4` |  |
| `alpha` | EMA decay factor for teacher updates | `0.99` |  |
| `lambda_u` | Weight for unsupervised (consistency) loss | `1.0` |  |
| `epochs` | Number of training epochs | `20` |  |
| `input_type` | Type of input data (`"image"`, `"text"`, `"tabular"`) | `"text"` |  |
| `labeled_dataset_path` | Path to labeled dataset | `"datasets/labeled.csv"` |  |
| `unlabeled_dataset_path` | Path to unlabeled dataset | `"datasets/unlabeled_images_folder"` |  |
| `validation_set_percentage` | Proportion of labeled data used for validation | `0.2` | Values must be between 0 and 1 (exclusive) |
| `batch_size` | Batch size for training | `64` |  |
| `image_size` | Resize target for image inputs (H, W) | `(224, 224)` |  |
| `text_column` | Column name for text input | `"text"` |  |
| `text_target_column` | Column name for text label | `"label"` |  |
| `categorical_columns` | List of categorical feature column names | `["gender", "city"]` | Include target column for classification outputs |
| `numeric_columns` | List of numerical feature column names | `["age", "income"]` | Do not include target column for regression outputs |
| `tabular_target_column` | Column name for target label in tabular data | `"defaulted"` |  |
| `is_tabular_target_categorical` | Whether the tabular task is classification | `True` |  |

### 3. Output

Once your configuration is set and your datasets are prepared, simply **run the notebook** (`mean_teacher.ipynb`).

#### What to Expect:

-  **Training statistics** (loss, accuracy/MAE) will be printed at the bottom of the notebook during and after training.
-  Best model will be saved automatically to: `models/mean_teacher/best_model_<input_type>_<training_session>.pt`
