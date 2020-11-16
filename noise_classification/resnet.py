from torch import nn


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.Relu()

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.Relu()

        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = self.relu2(out)

        out = self.pool(out)

        return out


class SmallResNet(nn.Module):
    def __init__(self, num_classes):
        self.conv = nn.Conv2d(1, 64, (7, 7), stride=2, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.Relu()
        self.pool = nn.MaxPool2d((2, 2))

        self.block1 = ResNetBlock(64, 128)
        self.block2 = ResNetBlock(128, 256)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pool(out)

        out = self.block1(out)
        out = self.block2(out)

        out = self.avg_pool(out)
        out = self.flatten(out)

        out = self.fc(out)
        return out
