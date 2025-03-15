import torch

from dataset import MyDataset
from model import MyModel
from utils import accuracy
from typing import Tuple
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")

print("Using device: {}".format(device))

def compute_accuracy(predicted_batch: torch.Tensor, label_batch: torch.Tensor) -> int:
    pred = predicted_batch.argmax(dim=1, keepdim=True) # get the index of the max log-probability
    acum = pred.eq(label_batch.view_as(pred)).sum().item()
    return acum

def train_single_epoch(train_loader: torch.utils.data.DataLoader,
        network: torch.nn.Module,
        optimizer: torch.optim,
        criterion: torch.nn.functional,
        epoch: int) -> Tuple[float]:
    avg_loss = []
    acc = 0.
    network.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        # Move input data and labels to the device
        data, target = data.to(device), target.to(device)

        # Set network gradients to 0.
        optimizer.zero_grad()

        # Forward batch of images through the network
        output = network(data)

        # Compute loss
        loss = criterion(output, target)

        # Compute backpropagation
        loss.backward()

        # Update parameters of the network
        optimizer.step()

        # Compute metrics
        acc += compute_accuracy(output, target)
        avg_loss.append(loss.item())

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    avg_acc = 100. * acc / len(train_loader.dataset)

    return np.mean(avg_acc), avg_acc


@torch.no_grad() # decorator: avoid computing gradients
def eval_single_epoch(
        test_loader: torch.utils.data.DataLoader,
        network: torch.nn.Module,
        criterion: torch.nn.functional
    ) -> Tuple[float, float]:
    # Deactivate the train=True flag inside the model
    network.eval()

    test_loss = []
    acc = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = network(data)
        # Apply the loss criterion and accumulate the loss
        loss = criterion(output, target)

        test_loss.append(loss.item())

        # compute number of correct predictions in the batch
        acc += compute_accuracy(output, target)

    test_acc = 100. * acc / len(test_loader.dataset)
    test_loss = np.mean(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, acc, len(test_loader.dataset), test_acc,
        ))

    return test_loss, test_acc

def train_model(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]), # used to normalize pixel values to a range of [-1, 1], which helps with stable and faster training
    ])
    my_dataset = MyDataset(images_path=config["images_path"], labels_path=config["labels_path"], transform=transform)
    train_dataset, test_dataset = torch.utils.data.random_split(my_dataset, [0.8, 0.2])
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config['test_batch_size'],
        shuffle=False,
        drop_last=True,
    )

    my_model = MyModel().to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(my_model.parameters(), lr=config['learning_rate_cnn'])

    for epoch in range(config["epochs"]):
        train_single_epoch(
            train_loader=train_loader,
            network=my_model,
            optimizer=optimizer,
            criterion=loss_function,
            epoch=epoch,
        )
        eval_single_epoch(
            test_loader=test_loader,
            network=my_model,
            criterion=loss_function,
        )

    return my_model


if __name__ == "__main__":
    config = {
        "learning_rate_cnn": 1e-3,
        "batch_size": 64,
        "test_batch_size": 1000,
        "epochs": 10,
        "images_path": "data/data",
        "labels_path": "data/chinese_mnist.csv"
    }

    train_model(config)