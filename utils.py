import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import os


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


def compute_gradient_loss(critic, real_image, fake_image, device):
    epsilon = torch.rand(real_image.size(0), 1, 1, 1).to(device)
    x_hat = epsilon * real_image + (1 - epsilon) * fake_image

    critic_output = critic(x_hat)
    fake_outputs = torch.ones(real_image.shape[0], 1, requires_grad=False).to(device)
    gradients = torch.autograd.grad(critic_output, x_hat, fake_outputs, create_graph=True, retain_graph=True,
                                    only_inputs=True)
    gradients = gradients[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_loss = torch.mean((torch.norm(gradients, p=2) - 1) ** 2)
    return gradient_loss


def save_images(generator, z_dim, writer, epoch, device):
    generator.eval()

    z = torch.randn(8, z_dim).to(device)
    fake_images = generator(z)
    img_grid = torchvision.utils.make_grid(fake_images)
    writer.add_image(f"Fake Images | Epoch {epoch}", img_grid)

