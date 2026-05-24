import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, activation, n_hidden=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.activation = activation
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 10)

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.pool(self.activation(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x