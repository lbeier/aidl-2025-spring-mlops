import torch

class MyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # If you need a starting point, 3 convolutional layers with MaxPools and ReLUs followed by 2 Linear layers with a ReLU is a good start.

        # Output volume size: ( N + 2P – F ) / S + 1

        # For filters FxF, zero-padding with P=(F-1)/2 preserves size spatially
        self.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv2 = torch.nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv3 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.maxPool = torch.nn.MaxPool2d(kernel_size=2)

        self.relu = torch.nn.ReLU()

        self.linear1 = torch.nn.Linear(in_features=128 * 8 * 8, out_features=512) #  in_features = out_channels * (output_size * output_size) from previous conv layer
        self.linear2 = torch.nn.Linear(in_features=512, out_features=15) # For 15-class classification

    def forward(self, x):
        # Input size (1, 64, 64)
        x = self.conv1(x)
        x = self.maxPool(x) # After pooling, size (1, 32, 32)
        x = self.relu(x)

        # Input size (1, 32, 32)
        x = self.conv2(x)
        x = self.maxPool(x) # After pooling, size (1, 16, 16)
        x = self.relu(x)

        # Input size (1, 16, 16)
        x = self.conv3(x)
        x = self.maxPool(x) # After pooling, size (1, 8, 8)
        x = self.relu(x)

        x = x.flatten(start_dim=1)

        x = self.linear1(x)
        x = self.relu(x)

        return self.linear2(x) # Do not apply ReLU after linear2 because CrossEntropyLoss expects raw logits (unactivated values) for multi-class classification.