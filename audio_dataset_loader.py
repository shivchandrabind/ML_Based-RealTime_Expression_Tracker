import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class AudioFeatureDataset(Dataset):
    def __init__(self, feature_dir):
        self.feature_dir = feature_dir
        self.feature_files = []
        self.labels = []

        for file in os.listdir(feature_dir):
            if not file.endswith(".csv"):
                continue

            self.feature_files.append(os.path.join(feature_dir, file))

            if "lie" in file.lower():
                self.labels.append(1)  # Lie = 1
            elif "truth" in file.lower():
                self.labels.append(0)  # Truth = 0
            else:
                raise ValueError(f"Cannot determine label for file: {file}")

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        file_path = self.feature_files[idx]
        df = pd.read_csv(file_path, header=0)
        df = df.select_dtypes(include=["number"])  # Keep only numeric columns
        features = df.values.flatten().astype(float)
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label
