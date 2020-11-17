import torch.optim as optim
import torch


def calc_batch_mse(noisy_images, clean_images, lens):
    h, w = noisy_images.size()[2:4]
    masks = (torch.arange(h)[None, :].to(lens.device) < lens[:, None]).unsqueeze(-1).unsqueeze(1)
    mse_batch = torch.sum(((noisy_images - clean_images) * masks) ** 2)
    mse_batch /= w * lens.sum()
    return mse_batch


def test_model(model, test_loader, device):
    model.eval()
    mse_loss = 0
    with torch.no_grad():
        for imgs_noisy, imgs_clean, lens in test_loader:

            imgs_noisy = imgs_noisy.to(device)
            imgs_clean = imgs_clean.to(device)
            lens = lens.to(device)

            answers = model(imgs_noisy)

            mse_loss += calc_batch_mse(answers, imgs_clean, lens).item()

    mse_loss /= len(test_loader)
    return mse_loss


def train_model(model, train_loader, val_loader, num_epochs, lr,
                device, log=False):

    optimizer = optim.Adam(model.parameters(), lr=lr)

    val_mse = []
    for i in range(num_epochs):
        model.train()

        if log:
            print("Epoch {}".format(i))

        running_loss = 0.0

        for i_batch, (imgs_noisy, imgs_clean, lens) in enumerate(train_loader):

            imgs_noisy = imgs_noisy.to(device)
            imgs_clean = imgs_clean.to(device)
            lens = lens.to(device)

            optimizer.zero_grad()

            answers = model(imgs_noisy)

            loss = calc_batch_mse(answers, imgs_clean, lens)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        mse_loss = test_model(model, val_loader, device)

        val_mse.append(mse_loss)

        if log:
            print("Train MSE: {:.4f}".format(running_loss / len(train_loader)))
            print("Validation MSE: {:.4f}".format(mse_loss))

    return val_mse
