import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import load_data
from model import Generator, Critic


class Handler:
    def __init__(self, model, model_name, z_dim, data_type, n_critic, batch_size, device, lr):
        assert model in ["WGAN", "DCGAN"], "Model must be WGAN/DCGAN."
        assert data_type in ["CIFAR10", "FashionMNIST"], "Data type must be CIFAR10/FashionMNIST."

        self.z_dim = z_dim
        self.out_dim = 3 * 32 * 32
        if data_type == "FashionMNIST":
            self.out_dim = 1 * 28 * 28

        self.generator = Generator(z_dim=self.z_dim, out_dim=self.out_dim)
        self.critic = Critic(in_dim=self.out_dim)
        self.model_name = model_name
        self.data_type = data_type
        self.n_critic = n_critic
        self.batch_size = batch_size
        self.device = device
        self.lr = lr

        self.train_dataset, self.test_dataset = load_data(self.data_type)

        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(0, 0.9))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, betas=(0, 0.9))

    def train_one_epoch(self, epoch):
        self.generator.train()
        self.critic.train()

        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        print_every = len(dataloader) // 5

        for batch_idx, (image, _) in enumerate(dataloader):
            image = image.to(self.device)
            z = torch.randn(self.batch_size, self.z_dim)

            self.critic_optimizer.zero_grad()

            fake_image = self.generator(z).detach()

            critic_loss = torch.mean(self.critic(fake_image)) - torch.mean(self.critic(image))

            critic_loss.backward()
            self.critic_optimizer.step()



if __name__ == "__main__":
    model = "WGAN"
    model_name = "WGAN"
    z_dim = 128
    data_type = "CIFAR10"
    n_critic = 5
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    lr = 0.0001

    handler = Handler(
        model=model,
        model_name=model_name,
        z_dim=z_dim,
        data_type=data_type,
        n_critic=n_critic,
        batch_size=batch_size,
        device=device,
        lr=lr
    )

    handler.train_one_epoch(0)


