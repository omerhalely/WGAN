import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import load_data, compute_gradient_loss
from model import Generator, Critic
import os


torch.manual_seed(42)


class Handler:
    def __init__(self, model, model_name, z_dim, data_type, n_critic, batch_size, device, lr, epsilon, lamda, epochs):
        assert model in ["WGAN", "DCGAN"], "Model must be WGAN/DCGAN."
        assert data_type in ["CIFAR10", "FashionMNIST"], "Data type must be CIFAR10/FashionMNIST."

        self.z_dim = z_dim
        self.out_shape = (3, 32, 32)
        if data_type == "FashionMNIST":
            self.out_shape = (1, 28, 28)

        print("Building Model...")
        self.generator = Generator(z_dim=self.z_dim, out_shape=self.out_shape)
        self.critic = Critic(in_dim=self.out_shape)

        self.model_name = model_name
        self.data_type = data_type
        self.n_critic = n_critic
        self.batch_size = batch_size
        self.device = device
        self.lr = lr
        self.epsilon = epsilon
        self.lamda = lamda
        self.epochs = epochs

        self.train_dataset, self.test_dataset = load_data(self.data_type)

        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(0, 0.9))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, betas=(0, 0.9))

        self.generator.to(self.device)
        self.critic.to(self.device)

    def train_one_epoch(self, epoch):
        self.generator.train()
        self.critic.train()

        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        critic_total_loss = 0
        generator_total_loss = 0

        print_every = len(dataloader) // 5

        for batch_idx, (image, _) in enumerate(dataloader):
            if batch_idx % print_every == 0 and batch_idx != 0:
                print(f"| Epoch {epoch + 1} | Critic Loss {critic_total_loss / batch_idx} | "
                      f"Generator Loss {generator_total_loss / batch_idx} |")

            image = image.to(self.device)
            z = torch.randn(self.batch_size, self.z_dim).to(self.device)

            self.critic_optimizer.zero_grad()

            fake_image = self.generator(z)

            loss = torch.mean(self.critic(fake_image)) - torch.mean(self.critic(image))
            gradient_loss = self.lamda * compute_gradient_loss(self.critic, image, fake_image, self.epsilon,
                                                               self.device)
            critic_loss = loss + gradient_loss
            critic_loss.backward()
            self.critic_optimizer.step()

            critic_total_loss += critic_loss.item()

            self.generator_optimizer.zero_grad()
            if batch_idx % self.n_critic == 0:
                fake_image = self.generator(z)

                generator_loss = -torch.mean(self.critic(fake_image))

                generator_loss.backward()
                self.generator_optimizer.step()

                generator_total_loss += generator_loss.item()

        average_critic_loss = critic_total_loss / len(dataloader),
        average_generator_loss = generator_total_loss / len(dataloader),
        return average_critic_loss, average_generator_loss

    def run(self):
        if not os.path.exists(os.path.join(os.getcwd(), "models")):
            os.mkdir(os.path.join(os.getcwd(), "models"))
        if not os.path.exists(os.path.join(os.getcwd(), "models", self.model_name)):
            os.mkdir(os.path.join(os.getcwd(), "models", self.model_name))

        checkpoint_filename = os.path.join(os.getcwd(), "models", self.model_name, f"{self.model_name}.pt")

        best_critic_loss = torch.inf
        best_generator_loss = torch.inf
        for epoch in range(epochs):
            print(70 * "-")
            critic_loss, generator_loss = self.train_one_epoch(epoch)

            if critic_loss < best_critic_loss or generator_loss < best_generator_loss:
                state = {
                    "generator": self.generator.state_dict(),
                    "critic": self.critic.state_dict()
                }
                torch.save(state, checkpoint_filename)
                best_critic_loss = critic_loss
                best_generator_loss = generator_loss


if __name__ == "__main__":
    model = "WGAN"
    model_name = "WGAN"
    z_dim = 128
    data_type = "CIFAR10"
    n_critic = 5
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    lr = 0.0001
    epsilon = 0.9
    lamda = 10
    epochs = 5

    handler = Handler(
        model=model,
        model_name=model_name,
        z_dim=z_dim,
        data_type=data_type,
        n_critic=n_critic,
        batch_size=batch_size,
        device=device,
        lr=lr,
        epsilon=epsilon,
        lamda=lamda,
        epochs=epochs
    )

    handler.train_one_epoch(0)


