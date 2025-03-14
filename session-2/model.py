import torch

class MyModel(torch.nn.Module):

    def __init__(self):
        # If you need a starting point, 3 convolutional layers with MaxPools and ReLUs followed by 2 Linear layers with a ReLU is a good start.

        self.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=128,
            out_channels=512,
            kernel_size=3
        )

        self.poll = torch.nn.MaxPool2d(kernel_size=2)

        self.relu = torch.nn.ReLU()

        self.linear1 = torch.nn.Linear(in_features=, out_features=)
        self.linear2 = torch.nn.Linear(in_features=, out_features=)

        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        pass