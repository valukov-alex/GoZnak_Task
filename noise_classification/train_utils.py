from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torch


def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    ce_loss = 0
    with torch.no_grad():
        for imgs, labels in test_loader:

            imgs = imgs.to(device)
            labels = labels.to(device)

            preds = model(imgs)
            _, answers = torch.max(preds.data, 1)

            total += labels.size(0)
            correct += (answers == labels).sum().item()

            ce_loss += F.cross_entropy(preds, labels).item()
    ce_loss /= len(test_loader)
    accuracy = correct / total
    return ce_loss, accuracy


def train_model(model, train_loader, val_loader, num_epochs, lr,
                device, log=False):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    val_acc = []
    for i in range(num_epochs):
        model.train()

        if log:
            print("Epoch {}".format(i))

        running_loss = 0.0
        total = 0
        correct = 0

        for i_batch, (imgs, labels) in enumerate(train_loader):

            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            pred = model(imgs)

            _, answers = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (answers == labels).sum().item()

            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_acc = correct/total
        ce_loss, acc = test_model(model, val_loader, device)

        val_acc.append(acc)

        if log:
            print("Train: accuracy: {:.3f}, CE: {:.3f}".format(train_acc, running_loss/len(train_loader)))
            print("Validation: accuracy: {:.3f}, CE: {:.3f}".format(acc, ce_loss))

    return val_acc
