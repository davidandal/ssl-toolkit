import os
import shutil
import random

INPUT_DIR = "../datasets/rvl_cdip_kyc"
MAX_PER_CLASS = 1000
UNLABELED_PER_CLASS = 600 
RANDOM_STATE = 69  

random.seed(RANDOM_STATE)  

labeled_dir = "../datasets/rvl_cdip_kyc_labeled"
unlabeled_dir = "../datasets/rvl_cdip_kyc_unlabeled"
os.makedirs(labeled_dir, exist_ok=True)
os.makedirs(unlabeled_dir, exist_ok=True)

for class_name in os.listdir(INPUT_DIR):
    class_path = os.path.join(INPUT_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
    sampled_images = random.sample(images, min(MAX_PER_CLASS, len(images)))

    unlabeled_samples = set(random.sample(sampled_images, min(UNLABELED_PER_CLASS, len(sampled_images))))
    labeled_samples = [img for img in sampled_images if img not in unlabeled_samples]

    labeled_class_dir = os.path.join(labeled_dir, class_name)
    os.makedirs(labeled_class_dir, exist_ok=True)
    for img_name in labeled_samples:
        src = os.path.join(class_path, img_name)
        dst = os.path.join(labeled_class_dir, img_name)
        shutil.copy2(src, dst)

    for img_name in unlabeled_samples:
        src = os.path.join(class_path, img_name)
        dst = os.path.join(unlabeled_dir, f"{class_name}_{img_name}")
        shutil.copy2(src, dst)

print("✅ Done! Labeled and unlabeled subsets created.")
