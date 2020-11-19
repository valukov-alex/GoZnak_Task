import torch
from model import CRNN
from dataset import pad_mel_spectogram
import argparse
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_file", type=str)
    parser.add_argument("path_to_save", type=str)
    parser.add_argument("-m", dest="path_to_model", default="models/model.pt", type=str)
    args = parser.parse_args()
    return args


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = parse_args()

    noisy_mel = np.load(args.path_to_file)
    h, w = noisy_mel.shape
    noisy_mel = pad_mel_spectogram(noisy_mel)
    noisy_mel = torch.tensor(noisy_mel, dtype=torch.float32)
    noisy_mel = noisy_mel.unsqueeze(0)

    model = CRNN().to(device)
    model.load_state_dict(torch.load(args.path_to_model, map_location=device))
    model.eval()

    clean_mel = model(noisy_mel)
    clean_mel = clean_mel.squeeze(0)
    clean_mel = clean_mel.data.cpu().numpy()
    clean_mel = clean_mel[:h]

    save_dir = os.path.dirname(args.path_to_save)
    if save_dir and not os.path.exists(save_dir):
        os.mkdir(save_dir)
    np.save(args.path_to_save, clean_mel)


if __name__ == "__main__":
    main()
