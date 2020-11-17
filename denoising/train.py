import torch
from torch.utils.data import DataLoader
from model import ResNetAutoEncoder
from train_utils import train_model
from dataset import DenoisingDataset
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", type=str)
    parser.add_argument("val_dir", type=str)
    parser.add_argument("model_save_dir", type=str)
    parser.add_argument("-b", dest="batch_size", default=64, type=int)
    parser.add_argument("-l", dest="learning_rate", default=1e-4, type=float)
    parser.add_argument("-e", dest="epochs", default=20, type=int)
    args = parser.parse_args()
    return args


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = parse_args()

    train_dataset = DenoisingDataset(args.train_dir)
    val_dataset = DenoisingDataset(args.val_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False)

    model = ResNetAutoEncoder().to(device)
    train_model(model, train_loader, val_loader, args.epochs,
                args.learning_rate, device, log=True)

    torch.save(model, args.model_save_dir)


if __name__ == "__main__":
    main()
