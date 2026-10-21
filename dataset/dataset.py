import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SemiconductorDataset(Dataset):

    def __init__(self, root_dir, file_names=None):

        self.gt_dir = os.path.join(root_dir, "GT")
        self.noisy_dir = os.path.join(root_dir, "NoisyLR")

        if not os.path.isdir(self.gt_dir):
            raise FileNotFoundError(f"GT directory not found: {self.gt_dir}")

        if not os.path.isdir(self.noisy_dir):
            raise FileNotFoundError(f"NoisyLR directory not found: {self.noisy_dir}")

        if file_names is None:
            self.files = sorted(os.listdir(self.gt_dir))
        else:
            self.files = sorted(file_names)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        filename = self.files[idx]

        gt = np.load(os.path.join(self.gt_dir, filename)).astype(np.float32)
        noisy = np.load(os.path.join(self.noisy_dir, filename)).astype(np.float32)

        gt = torch.from_numpy(gt).unsqueeze(0)
        noisy = torch.from_numpy(noisy).unsqueeze(0)

        return noisy, gt