"""
This script will randomly distribute data into train,
validation, and test directories, based on the split ratios.
"""
import os
import shutil
import numpy as np

# Set your source and target directories
src_dir = "saved_frames"
target_dir = "bladder_data"

# Set ratios for splitting
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Create train, validation, and test directories
os.makedirs(os.path.join(target_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "validation"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "test"), exist_ok=True)

classes = ['NMIBC', 'Ta', 'Normal']

for cls in classes:
    os.makedirs(os.path.join(target_dir, "train", cls), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "validation", cls), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "test", cls), exist_ok=True)

    # Get list of all files
    src_files = os.listdir(os.path.join(src_dir, cls))

    # Randomly shuffle the files
    np.random.shuffle(src_files)

    # Split the files into train, validation and test based on the provided ratios
    train_files, val_files, test_files = np.split(
        np.array(src_files),
        [int(len(src_files)*train_ratio), int(len(src_files)*(train_ratio + val_ratio))])

    # Copy files into respective directories
    for name in train_files:
        shutil.copy(os.path.join(src_dir, cls, name), os.path.join(target_dir, "train", cls))

    for name in val_files:
        shutil.copy(os.path.join(src_dir, cls, name), os.path.join(target_dir, "validation", cls))

    for name in test_files:
        shutil.copy(os.path.join(src_dir, cls, name), os.path.join(target_dir, "test", cls))

print("Data split and copied successfully.")
