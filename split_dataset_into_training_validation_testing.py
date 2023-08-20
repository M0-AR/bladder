import os
import shutil
import random

# Paths
src_image_dir = 'dataset-jpg/images/'
src_anno_dir = 'dataset-jpg/annotations/'

train_image_dir = 'dataset-jpg/train/images/'
train_anno_dir = 'dataset-jpg/train/annotations/'

val_image_dir = 'dataset-jpg/val/images/'
val_anno_dir = 'dataset-jpg/val/annotations/'

test_image_dir = 'dataset-jpg/test/images/'
test_anno_dir = 'dataset-jpg/test/annotations/'

# Get a list of all the image and annotation files
all_files = [f.split('.')[0] for f in os.listdir(src_image_dir) if f.endswith('.png')]

# Shuffle the files
random.shuffle(all_files)

# Split ratios
train_ratio = 0.8
val_ratio = 0.1

# Calculate split sizes
train_size = int(len(all_files) * train_ratio)
val_size = int(len(all_files) * val_ratio)

# Split the files
train_files = all_files[:train_size]
val_files = all_files[train_size: train_size+val_size]
test_files = all_files[train_size+val_size:]

# Copy files to respective directories
for file in train_files:
    shutil.copy(src_image_dir + file + '.png', train_image_dir)
    shutil.copy(src_anno_dir + file + '.txt', train_anno_dir)

for file in val_files:
    shutil.copy(src_image_dir + file + '.png', val_image_dir)
    shutil.copy(src_anno_dir + file + '.txt', val_anno_dir)

for file in test_files:
    shutil.copy(src_image_dir + file + '.png', test_image_dir)
    shutil.copy(src_anno_dir + file + '.txt', test_anno_dir)
