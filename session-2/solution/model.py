import torch
import torch.nn.functional as F

class MyModel(torch.nn.Module):

    def __init__(self, h1=32, h2=64, h3=128, h4=128):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, h1, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(h1, h2, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(h2, h3, 3, padding=1)

        self.fc1 = torch.nn.Linear(8*8*h3, h4)
        self.fc2 = torch.nn.Linear(h4, 15)

        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        return self.fc2(x)