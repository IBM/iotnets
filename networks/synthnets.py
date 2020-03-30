import torch.nn as nn
import torch.nn.functional as F

class SynthWide(nn.Module):
    def __init__(self, num_c=10, f=1):
        super(SynthWide, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32*f, 3, padding=1)    # 32x32
        self.conv2 = nn.Conv2d(32*f, 64*f, 3, padding=1)   # 16x16
        self.conv3 = nn.Conv2d(64*f, 128*f, 3, padding=1)   # 8x8
        self.conv4 = nn.Conv2d(128*f, 256, 3, padding=1)  # 4x4
        self.fc1 = nn.Linear(256*4*4, num_c)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 256*4*4)
        x = self.fc1(x)
        return x


class SynthWideAndLong(nn.Module):
    def __init__(self, num_c=10, f=1, l=1):
        super(SynthWideAndLong, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32*f, 3, padding=1)       # 32x32
        self.block1 = self._make_layer(32*f, l)
        self.conv2 = nn.Conv2d(32*f, 64*f, 3, padding=1)    # 16x16
        self.block2 = self._make_layer(64*f, l)
        self.conv3 = nn.Conv2d(64*f, 128*f, 3, padding=1)   # 8x8
        self.block3 = self._make_layer(128*f, l)
        self.conv4 = nn.Conv2d(128*f, 256, 3, padding=1)    # 4x4
        self.fc1 = nn.Linear(256*4*4, num_c)

    def _make_layer(self, channels, l):
        layers = []
        for i in range(l):
            layers.append(nn.Conv2d(channels, channels, [3, 3], [1, 1], 1, bias=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.block1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.block2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.block3(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 256*4*4)
        x = self.fc1(x)
        return x


class SynthWide256(nn.Module):
    def __init__(self, num_c=10, f=1):
        super(SynthWide256, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32*f, 3, padding=1)       # 256x256
        self.conv2 = nn.Conv2d(32*f, 64*f, 3, padding=1)    # 128*128
        self.conv3 = nn.Conv2d(64*f, 128*f, 3, padding=1)   # 64x64
        self.conv4 = nn.Conv2d(128*f, 256*f, 3, padding=1)  # 32x32
        self.conv5 = nn.Conv2d(256*f, 512*f, 3, padding=1)  # 16x16
        self.conv6 = nn.Conv2d(512*f, 1024*f, 3, padding=1) # 8x8
        self.conv7 = nn.Conv2d(1024*f, 256, 3, padding=1)   # 4x4
        self.fc1 = nn.Linear(256*4*4, num_c)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
        x = self.pool(F.relu(self.conv7(x)))
        x = x.view(-1, 256*4*4)
        x = self.fc1(x)
        return x
