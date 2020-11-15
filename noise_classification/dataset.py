import os
import torch
from torch.utils.data import Dataset
import numpy as np


def pad_mel_spectogram(mel_spectogram, length=1000):
    h, w = mel_spectogram.shape
    return np.resize(mel_spectogram, (1, length, w))


class NoiseClassificationDataset(Dataset):
    def __init__(self, root_dir):
        """
        В папке root_dir должны находиться 2 папки: clean и noisy
        """
        self.root_dir = root_dir
        self.records_list = []

        clean_path = os.path.join(root_dir, 'clean')
        noisy_path = os.path.join(root_dir, 'noisy')

        for person_idx in os.listdir(clean_path):
            person_path = os.path.join(clean_path, person_idx)
            for filename in os.listdir(person_path):
                path_to_file = os.path.join(person_path, filename)
                self.records_list.append((path_to_file, 0))

        for person_idx in os.listdir(noisy_path):
            person_path = os.path.join(noisy_path, person_idx)
            for filename in os.listdir(person_path):
                path_to_file = os.path.join(person_path, filename)
                self.records_list.append((path_to_file, 1))

    def __len__(self):
        return len(self.records_list)

    def __getitem__(self, idx):
        path = self.records_list[idx][0]
        label = self.records_list[idx][1]

        mel_spectrogram = pad_mel_spectogram(np.load(path)).astype(np.float32)
        mel_spectrogram = torch.from_numpy(mel_spectrogram)

        return mel_spectrogram, label
