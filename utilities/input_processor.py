
from transformers import AutoTokenizer
from torchvision import transforms
import torch
import numpy as np

def get_text_tokenizer(model_name='bert-base-uncased', max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return lambda texts: tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )

def get_image_transform(img_size=224, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def get_tabular_transform():
    return lambda row: torch.tensor(np.array(list(row.values())), dtype=torch.float32)

def get_processor(input_type, **kwargs):
    if input_type == 'text':
        return get_text_tokenizer(**kwargs)
    elif input_type == 'image':
        return get_image_transform(**kwargs)
    elif input_type == 'tabular':
        return get_tabular_transform()
    else:
        raise ValueError(f'Unsupported input type: {input_type}')
