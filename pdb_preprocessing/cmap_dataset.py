''' 
Need to add the Label data!
'''
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class CustomContactMapDataset(Dataset):
    def __init__(self, cmap_dir, transform=None, target_transform=None):
        self.cmap_files = [os.path.join(cmap_dir, f) for f in os.listdir(cmap_dir) if f.endswith('.npy')]
        self.transform = transform
        self.target_transform = target_transform
        self.max_size = 0  # Variable to store the size of the largest matrix
        self.padded_tensors = []

        # Process data and find the size of the largest matrix
        self.tensors = []
        for cmap_path in self.cmap_files:
            data = np.load(cmap_path)
            tensor_data = torch.as_tensor(data)
            self.tensors.append(tensor_data)

        # Update max_size if a larger matrix is found
            self.max_size = max(self.max_size, max(tensor_data.size()))

        # Pad the tensors to the size of the largest matrix
        for tensor in self.tensors:
            pad_rows = self.max_size - tensor.size(0)
            pad_cols = self.max_size - tensor.size(1)
            padded_tensor = F.pad(tensor, (0, pad_cols, 0, pad_rows))
            self.padded_tensors.append(padded_tensor)

    def __len__(self):
        return len(self.cmap_files)

    def __getitem__(self, idx):
        return self.padded_tensors[idx]


'''Test zone'''
# Contact Map directory path
cmap_dir = "../seed-development-gene-prediction/pdb_preprocessing/contact_maps"

# Checking if files are loaded properly
cmap_files = [f for f in os.listdir(cmap_dir) if f.endswith('.npy')]
print(cmap_files)

# Create an instance of the ContactMapDataset
cmap_dataset = CustomContactMapDataset(cmap_dir)

# List the contact map files found
print(cmap_dataset.cmap_files)

# Define DataLoader
batch_size = 3  
cmap_dataloader = DataLoader(cmap_dataset, batch_size=batch_size, shuffle=True)

train_features = next(iter(cmap_dataloader))
print(f"Feature batch shape: {train_features.size()}")
