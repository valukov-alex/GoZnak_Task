import torch
from torch.utils.data import DataLoader
from model import SmallResNet
from train_utils import train_model
from dataset import NoiseClassificationDataset
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", type=str)
    parser.add_argument("val_dir", type=str)
    parser.add_argument("-m", dest="model_save_path", default="./models/model.pt", type=str)
    parser.add_argument("-b", dest="batch_size", default=64, type=int)
    parser.add_argument("-l", dest="learning_rate", default=1e-4, type=float)
    parser.add_argument("-e", dest="epochs", default=20, type=int)
    args = parser.parse_args()
    return args


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = parse_args()

    train_dataset = NoiseClassificationDataset(args.train_dir)
    val_dataset = NoiseClassificationDataset(args.val_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False)

    model = SmallResNet(num_classes=2).to(device)
    train_model(model, train_loader, val_loader, args.epochs,
                args.learning_rate, device, log=True)

    model_dir = os.path.dirname(args.model_save_path)
    if not os.path.exists(model_dir) and model_dir:
        os.mkdir(model_dir)
    torch.save(model, args.model_save_path, _use_new_zipfile_serialization=False)


if __name__ == "__main__":
    main()
