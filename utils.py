import os
import torchvision.transforms as transforms
from torchvision import datasets


def load_data(data_type):
    if not os.path.exists(os.path.join(os.getcwd(), "data", data_type)):
        os.mkdir(os.path.join(os.getcwd(), "data"))

    if not os.path.exists(os.path.join(os.getcwd(), "data", data_type)):
        os.mkdir(os.path.join(os.getcwd(), "data", data_type))
        os.mkdir(os.path.join(os.getcwd(), "data", data_type, "train"))
        os.mkdir(os.path.join(os.getcwd(), "data", data_type, "test"))

    if data_type == "CIFAR10":
        train_dataset = datasets.CIFAR10("./data/CIFAR10/train", train=True, download=True,
                                         transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR10("./data/CIFAR10/test", train=False, download=True,
                                        transform=transforms.ToTensor())
        return train_dataset, test_dataset

    if data_type == "FashionMNIST":
        train_dataset = datasets.FashionMNIST("./data/CIFAR10/train", train=True, download=True,
                                              transform=transforms.ToTensor())
        test_dataset = datasets.FashionMNIST("./data/CIFAR10/test", train=True, download=True,
                                             transform=transforms.ToTensor())
        return train_dataset, test_dataset
