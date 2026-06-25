import torch.nn as nn

class CNN(nn.Module):
  def __init__(self, activation, hidden=128):
    super().__init__()

    self.saved = {}
    self.activation = activation
    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
    self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)

    self.flatten = nn.Flatten()

    self.fc1 = nn.Linear(64 * 16 * 16, hidden)
    self.fc2 = nn.Linear(hidden, 10)

    for m in self.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)

  def save_act(self, name, x):
    if x.requires_grad:
      x.retain_grad()
    self.saved[name] = x
  def forward(self, x):
    x = self.conv1(x)
    x = self.activation(x)
    self.save_act('conv1', x)

    x = self.conv2(x)
    x = self.activation(x)
    self.save_act('conv2', x)

    x = self.pool(x)

    x = self.conv3(x)
    x = self.activation(x)
    self.save_act('conv3', x)

    x = self.flatten(x)

    x = self.fc1(x)
    x = self.activation(x)
    self.save_act('fc1', x)
    
    x = self.fc2(x)
    return x