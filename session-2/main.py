import torch

from dataset import MyDataset
from model import MyModel
from utils import accuracy
from torchvision import transforms

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train_single_epoch():
    # TODO: Implement training loop
    raise NotImplementedError


def eval_single_epoch():
    # TODO: Implement evaluation loop
    raise NotImplementedError


def train_model(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]), # used to normalize pixel values to a range of [-1, 1], which helps with stable and faster training
    ])
    my_dataset = MyDataset(images_path=config["images_path"], labels_path=config["labels_path"], transform=transform)
    my_model = MyModel(...).to(device)

    for epoch in range(config["epochs"]):
        train_single_epoch(...)
        eval_single_epoch(...)

    return my_model


if __name__ == "__main__":
    config = {
        "images_path": "data/data",
        "labels_path": "data/chinese_mnist.csv"
    }

    train_model(config)