# dataset.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset

class BlendshapeDataset(Dataset):
    """
    Load per-user txt files and split into samples by numeric-only separator lines.
    Each non-empty, non-separator line is a CSV row with 52 floats (one frame).
    Returned sample shape: (T, 52)
    """

    def __init__(self, file_paths, path_to_label):
        """
        Args:
            file_paths (List[str]): list of txt paths for this split
            path_to_label (Dict[str, int]): label mapping for paths
        """
        self.samples = []
        self.labels = []
        self.path_to_label = path_to_label

        for p in file_paths:
            file_samples = self._load_samples(p)
            self.samples.extend(file_samples)
            self.labels.extend([self.path_to_label[p]] * len(file_samples))

    @staticmethod
    def _is_numeric_separator(line: str) -> bool:
        """Return True if line is a numeric-only separator."""
        s = line.strip()
        return s.isdigit()

    def _load_samples(self, file_path):
        samples = []
        current = []

        if not os.path.exists(file_path):
            print(f"[WARN] Missing file: {file_path}")
            return samples

        with open(file_path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if self._is_numeric_separator(line):
                    if current:
                        samples.append(np.array(current, dtype=np.float32))
                        current = []
                else:
                    feats = list(map(float, line.split(",")))
                    current.append(feats)

        if current:
            samples.append(np.array(current, dtype=np.float32))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]   # (T, 52)
        label = self.labels[idx]
        return torch.tensor(sample, dtype=torch.float32), label
