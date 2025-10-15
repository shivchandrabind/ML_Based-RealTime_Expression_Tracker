import os
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        # Iterate over dataset names: casme2, lpw, samm, ubfc
        for dataset_name in os.listdir(root_dir):
            dataset_path = os.path.join(root_dir, dataset_name)
            if os.path.isdir(dataset_path):
                # Inside each dataset, go through subject folders
                for subject_folder in os.listdir(dataset_path):
                    subject_path = os.path.join(dataset_path, subject_folder)
                    if os.path.isdir(subject_path):
                        for file in os.listdir(subject_path):
                            if file.endswith(('.jpg', '.png')):
                                file_path = os.path.join(subject_path, file)
                                self.samples.append((file_path, dataset_name))  # Label by dataset name

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label_idx = self._label_to_index(label)
        return image, label_idx

    def _label_to_index(self, label):
        label_map = {
            'casme2': 0,
            'lpw': 1,
            'samm': 2,
            'ubfc': 3
        }
        return label_map.get(label.lower(), -1)  # default to -1 if not found
