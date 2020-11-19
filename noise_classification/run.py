import torch
from model import SmallResNet
from dataset import pad_mel_spectogram
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_file", type=str)
    parser.add_argument("-m", dest="path_to_model", default="models/model.pt", type=str)
    args = parser.parse_args()
    return args


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = parse_args()

    mel = np.load(args.path_to_file)
    h, w = mel.shape
    mel = pad_mel_spectogram(mel)
    mel = torch.tensor(mel, dtype=torch.float32)
    mel = mel.unsqueeze(0)

    model = SmallResNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(args.path_to_model, map_location=device))
    model.eval()

    pred = model(mel)
    _, answer = torch.max(pred, 1)
    print(answer.squeeze(0).item())


if __name__ == "__main__":
    main()
