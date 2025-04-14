import os
import pandas as pd
from random import shuffle
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

dataset_path = 'SisFall_dataset'
label_dir = 'SisFall_temporally_annotated'
headers = ['ADXL345_x', 'ADXL345_y', 'ADXL345_z', 'ITG3200_x', 'ITG3200_y', 'ITG3200_z', 'MMA8451Q_x','MMA8451Q_y', 'MMA8451Q_z']

def create_train_test_split():
    all_files = []
    # Get all subdirectories in the dataset path
    subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    for subdir in subdirs:
        subdir_path = os.path.join(dataset_path, subdir)
        # Get all files in the subdirectory
        files = os.listdir(subdir_path)
        
        for file in files:
            full_path = os.path.join(subdir_path, file)
            if file == 'Readme.txt':
                continue
                
            if file.endswith('.txt'):
                all_files.append(full_path)

    shuffle(all_files)
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
    print(f"Num Train Files: {len(train_files)}")
    print(f"Num Test Files: {len(test_files)}")

    train_data = []
    test_data = []

    # For training files
    for file in train_files:
        dir_name = os.path.dirname(file)
        label_path = os.path.join(label_dir, os.path.basename(dir_name), os.path.basename(file))

        if not os.path.exists(label_path):
            print(f"Label file does not exist for {file}")
            continue

        data = pd.read_csv(file, sep=',')
        data.columns = headers
        data['MMA8451Q_z'] = data['MMA8451Q_z'].map(lambda x: str(x)[:-1])
        #print(data.shape)

        labels = pd.read_csv(label_path, sep=',', dtype=np.int64)
        labels.columns = ['label']
        #print(labels.shape)

        data['label'] = labels['label']
        data['file_name'] = os.path.basename(file)
        if data.isna().any().any():
            print(f"Skipping {file} due to NaN labels")
            continue
        train_data.append(data)

    # For test files
    for file in test_files:
        dir_name = os.path.dirname(file)
        label_path = os.path.join(label_dir, os.path.basename(dir_name), os.path.basename(file))

        if not os.path.exists(label_path):
            print(f"Label file does not exist for {file}")
            continue

        data = pd.read_csv(file, sep=',')
        data.columns = headers
        data['MMA8451Q_z'] = data['MMA8451Q_z'].map(lambda x: str(x)[:-1])
#        print(data.shape)

        labels = pd.read_csv(label_path, sep=',', dtype=np.int64)
        labels.columns = ['label']
#        print(labels.shape)
        data['label'] = labels['label']
        data['file_name'] = os.path.basename(file)

        if data.isna().any().any():
            print(f"Skipping {file} due to NaN labels")
            continue
        test_data.append(data)
    return train_data, test_data

train_data, test_data = create_train_test_split()
for i, df in enumerate(train_data):
    print(f"Train Data {i}: {df.shape}")
train_df = pd.concat(train_data, ignore_index=True)
test_df = pd.concat(test_data, ignore_index=True)
print(train_df.shape)
print(test_df.shape)

# Save to CSV files
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)