from torch import nn


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), 
                               stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), 
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1),
                                    stride=2)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(identity)

        out += identity

        out = self.relu2(out)

        return out


class DeconvResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeconvResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), 
                               stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=(3, 3), 
                                        stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(1, 1),
                                           stride=2, output_padding=1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.upsample(identity)
        out += identity

        out = self.relu2(out)

        return out


class CRNN(nn.Module):
    def __init__(self, h_in=80, h_out=80):
        super(CRNN, self).__init__()
        self.conv = nn.Conv2d(1, 32, (7, 7), stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.downblock = ResNetBlock(32, 64)
        self.upblock = DeconvResnetBlock(64, 32)
        self.upsample = nn.ConvTranspose2d(32, 16, (3, 3), stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.last_conv = nn.Conv2d(16, 1, (1, 1))
        self.fc1 = nn.Linear(h_in, 128)
        self.rnn = nn.GRU(128, 256, num_layers=1, batch_first=True, bidirectional=True)
        self.fc2 = nn.Linear(512, h_out)
    
    def forward(self, x):
        out = x.unsqueeze(1)

        out = self.conv(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.downblock(out)
        out = self.upblock(out)
        out = self.upsample(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.last_conv(out)

        out = out.squeeze(1)

        out = self.fc1(out)
        out, _ = self.rnn(out)
        out = self.fc2(out)

        return out
