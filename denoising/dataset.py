import os
import torch
from torch.utils.data import Dataset
import numpy as np


def pad_mel_spectogram(mel_spectogram, length = 1400):
    h, w = mel_spectogram.shape
    mel_spctr_new = np.zeros((length, w))
    mel_spctr_new[:h] = mel_spectogram
    mel_spctr_new.resize((1, length, w))
    return mel_spctr_new


class DenoisingDataset(Dataset):
    def __init__(self, root_dir):
        """
        В папке root_dir должны находиться 2 папки: clean и noisy
        """
        self.root_dir = root_dir
        clean_path = os.path.join(root_dir, 'clean')
        noisy_path = os.path.join(root_dir, 'noisy')

        self.records_list = []

        for person_idx in os.listdir(clean_path):
            person_clean_path = os.path.join(clean_path, person_idx)
            person_noisy_path = os.path.join(noisy_path, person_idx)
            for filename in os.listdir(person_clean_path):
                path_to_clean = os.path.join(person_clean_path, filename)
                path_to_noise = os.path.join(person_noisy_path, filename)
                if os.path.isfile(path_to_noise):
                    self.records_list.append((path_to_clean, path_to_noise))

    def __len__(self):
        return len(self.records_list)

    def __getitem__(self, idx):
        noisy_path = self.records_list[idx][0]
        clean_path = self.records_list[idx][1]

        mel_spectr_noisy = np.load(noisy_path).astype(np.float32)
        mel_spectr_clean = np.load(clean_path).astype(np.float32)

        assert mel_spectr_noisy.shape == mel_spectr_clean.shape
        len_ = mel_spectr_noisy.shape[0]

        mel_spectr_noisy = pad_mel_spectogram(mel_spectr_noisy)
        mel_spectr_clean = pad_mel_spectogram(mel_spectr_clean)

        mel_spectr_noisy = torch.tensor(mel_spectr_noisy, dtype=torch.float32)
        mel_spectr_clean = torch.tensor(mel_spectr_clean, dtype=torch.float32)

        return mel_spectr_noisy, mel_spectr_clean, len_
