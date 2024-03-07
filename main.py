import argparse
import torch
from Handler import Handler

parser = argparse.ArgumentParser(
    description="A program to build and train WGAN and DCGAN models.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--model",
    type=str,
    help="The model which will be trained (WGAN / DCGAN). Default - WGAN",
    default="WGAN",
)

parser.add_argument(
    "--model-name",
    type=str,
    help="The name of the model which will be saved in ./models/model_name. Default - WGAN_CIFAR10",
    default="WGAN_CIFAR10",
)

parser.add_argument(
    "--data",
    type=str,
    help="The type of the data which will be used (CIFAR10 / FashionMNIST). Default - CIFAR10",
    default="CIFAR10",
)

parser.add_argument(
    "--epochs",
    type=int,
    help="Number of epochs for training. Default - 20",
    default=20,
)

parser.add_argument(
    "--train",
    type=bool,
    help="If set to True, model will be trained, otherwise, model will be tested. Default - True",
    default=True,
)


if __name__ == "__main__":
    args = parser.parse_args()
    model = args.model
    model_name = args.model_name
    z_dim = 100
    data_type = args.data
    n_critic = 5
    if model == "DCGAN":
        n_critic = 1
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    lr = 0.0002
    lamda = 10
    epochs = args.epochs
    beta1 = 0.5
    beta2 = 0.9
    if model == "DCGAN":
        beta2 = 0.999
    train = args.train

    handler = Handler(
        model=model,
        model_name=model_name,
        z_dim=z_dim,
        data_type=data_type,
        n_critic=n_critic,
        batch_size=batch_size,
        device=device,
        lr=lr,
        lamda=lamda,
        epochs=epochs,
        beta1=0.5,
        beta2=0.999
    )

    if train:
        handler.train()
    else:
        handler.test()
