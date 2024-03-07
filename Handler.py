import matplotlib.pyplot as plt
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import load_data, compute_gradient_loss, save_images
from model import WGANGenerator, WGANCritic, DCGANGenerator, DCGANDiscriminator
import os


torch.manual_seed(42)


class Handler:
    def __init__(self, model, model_name, z_dim, data_type, n_critic, batch_size, device, lr, lamda, epochs, beta1,
                 beta2):
        assert model in ["WGAN", "DCGAN"], "Model must be WGAN/DCGAN."
        assert data_type in ["CIFAR10", "FashionMNIST"], "Data type must be CIFAR10/FashionMNIST."

        self.model = model
        self.z_dim = z_dim
        self.out_shape = (3, 32, 32)
        if data_type == "FashionMNIST":
            self.out_shape = (1, 28, 28)

        print(f"Building Model {model}")
        if model == "WGAN":
            self.generator = WGANGenerator(z_dim=self.z_dim, out_shape=self.out_shape)
            self.discriminator = WGANCritic(in_dim=self.out_shape)
        if model == "DCGAN":
            self.generator = DCGANGenerator(z_dim=self.z_dim, img_size=self.out_shape[1],
                                            output_channels=self.out_shape[0])
            self.discriminator = DCGANDiscriminator(image_size=self.out_shape[1], in_channels=self.out_shape[0])

        self.model_name = model_name
        self.data_type = data_type
        self.n_critic = n_critic
        self.batch_size = batch_size
        self.device = device
        self.lr = lr
        self.lamda = lamda
        self.epochs = epochs
        self.writer = SummaryWriter(f"runs/{self.model_name}")

        self.train_dataset = load_data(self.data_type)

        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, beta2))
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))

        self.generator.to(self.device)
        self.discriminator.to(self.device)

    def train_one_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()

        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        discriminator_total_loss = 0
        generator_total_loss = 0
        generator_iterations = 0

        DCGAN_loss = torch.nn.BCELoss()

        print_every = len(dataloader) // 5
        discriminator_loss = None
        generator_loss = None

        for batch_idx, (image, _) in enumerate(dataloader):
            if batch_idx % print_every == 0 and batch_idx != 0:
                print(f"| Epoch {epoch + 1} | Discriminator Loss {(discriminator_total_loss / batch_idx):.2f} | "
                      f"Generator Loss {(generator_total_loss / generator_iterations):.2f} |")

            image = image.to(self.device)
            z = torch.randn(image.size(0), self.z_dim).to(self.device)

            self.discriminator_optimizer.zero_grad()

            fake_image = self.generator(z)

            if self.model == "WGAN":
                loss = torch.mean(self.discriminator(fake_image)) - torch.mean(self.discriminator(image))
                gradient_loss = self.lamda * compute_gradient_loss(self.discriminator, image, fake_image, self.device)
                discriminator_loss = loss + gradient_loss

            if self.model == "DCGAN":
                real_ground_truth = torch.ones(image.shape[0], 1, requires_grad=False).to(self.device)
                fake_ground_truth = torch.zeros(image.shape[0], 1, requires_grad=False).to(self.device)
                real_image_loss = DCGAN_loss(self.discriminator(image), real_ground_truth)
                fake_image_loss = DCGAN_loss(self.discriminator(fake_image), fake_ground_truth)
                discriminator_loss = (real_image_loss + fake_image_loss) / 2

            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            discriminator_total_loss += discriminator_loss.item()

            self.generator_optimizer.zero_grad()
            if batch_idx % self.n_critic == 0:
                fake_image = self.generator(z)

                if self.model == "WGAN":
                    generator_loss = -torch.mean(self.discriminator(fake_image))

                if self.model == "DCGAN":
                    real_ground_truth = torch.ones(image.shape[0], 1, requires_grad=False).to(self.device)
                    generator_loss = DCGAN_loss(self.discriminator(fake_image), real_ground_truth)

                generator_loss.backward()
                self.generator_optimizer.step()

                generator_total_loss += generator_loss.item()
                generator_iterations += 1

        average_discriminator_loss = discriminator_total_loss / len(dataloader)
        average_generator_loss = generator_total_loss / generator_iterations
        return average_discriminator_loss, average_generator_loss

    def train(self):
        print(f"Start Training {self.model_name}")
        print(f"Training Model on {self.device}")

        if not os.path.exists(os.path.join(os.getcwd(), "models")):
            os.mkdir(os.path.join(os.getcwd(), "models"))
        if not os.path.exists(os.path.join(os.getcwd(), "models", self.model_name)):
            os.mkdir(os.path.join(os.getcwd(), "models", self.model_name))

        checkpoint_filename = os.path.join(os.getcwd(), "models", self.model_name, f"{self.model_name}.pt")
        save_images_checkpoint = 5
        for epoch in range(self.epochs):
            print(70 * "-")
            discriminator_loss, generator_loss = self.train_one_epoch(epoch)

            self.writer.add_scalars(f"Loss/{self.model_name}", {"Discriminator": discriminator_loss,
                                                                "Generator": generator_loss}, epoch)

            if epoch % save_images_checkpoint == 0:
                state = {
                    "generator": self.generator.state_dict(),
                    "discriminator": self.discriminator.state_dict()
                }
                torch.save(state, checkpoint_filename)
                print(f"Saved model to {checkpoint_filename}")
                save_images(self.generator, self.z_dim, self.writer, epoch, self.device)

            print(70 * "-")
            print(f"| End of epoch {epoch + 1} | Discriminator Loss {discriminator_loss:.2f} | "
                  f"Generator Loss {generator_loss:.2f} | ")

        print(70 * "-")
        save_images(self.generator, self.z_dim, self.writer, self.epochs + 1, self.device)

    def load_model(self):
        print(f"Loading model {self.model_name}...")
        model_path = os.path.join(os.getcwd(), "models", f"{self.model_name}", f"{self.model_name}.pt")
        assert os.path.exists(model_path), f"Model {model_name}.pt does not exist."

        checkpoint = torch.load(model_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator"])
        print("Loaded Model Successfully.")

    def test(self):
        print(f"Testing model {self.model_name}...")
        filename = os.path.join(os.getcwd(), "models", self.model_name, f"{self.model_name}_Samples.png")
        self.load_model()
        self.generator.eval()
        z = torch.randn(self.batch_size, self.z_dim).to(self.device)

        fake_images = self.generator(z)
        img_grid = 255 * torchvision.utils.make_grid(fake_images.cpu().detach())
        img_grid = img_grid.to(torch.uint8)
        plt.imshow(torch.permute(img_grid, (1, 2, 0)))
        plt.savefig(filename)
        plt.close()
        print(f"Saved outputs to {filename}.")


if __name__ == "__main__":
    model = "DCGAN"
    model_name = "DCGAN_FashionMNIST"
    z_dim = 100
    data_type = "FashionMNIST"
    n_critic = 1
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    lr = 0.0002
    lamda = 10
    epochs = 20
    beta1 = 0.5
    beta2 = 0.999

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

    handler.train()
    handler.test()
