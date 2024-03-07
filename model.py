import torch
import torch.nn as nn


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.linear_block(x)
        return x


class WGANGenerator(nn.Module):
    def __init__(self, z_dim, out_shape):
        super().__init__()
        self.out_shape = out_shape
        self.generator = nn.Sequential(
            LinearBlock(z_dim, 128),
            LinearBlock(128, 256),
            LinearBlock(256, 512),
            LinearBlock(512, 1024),
            LinearBlock(1024, out_shape[0] * out_shape[1] * out_shape[2]),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.generator(x)
        x = torch.reshape(x, (x.size(0), self.out_shape[0], self.out_shape[1], self.out_shape[2]))
        return x


class WGANCritic(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim[0] * in_dim[1] * in_dim[2], 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.model(x)
        return x


class DCGANGenerator(nn.Module):
    def __init__(self, z_dim, img_size, output_channels):
        super().__init__()
        self.img_size = img_size
        self.linear = nn.Linear(z_dim, 128 * (img_size // 4) ** 2)

        self.conv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), 128, self.img_size // 4, self.img_size // 4)
        x = self.conv(x)
        return x


class DCGANDiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
        )
        if bn:
            self.model = nn.Sequential(
                self.model,
                nn.BatchNorm2d(out_channels, 0.8)
            )

    def forward(self, x):
        x = self.model(x)
        return x


class DCGANDiscriminator(nn.Module):
    def __init__(self, image_size, in_channels):
        super().__init__()
        self.model = nn.Sequential(
            DCGANDiscriminatorBlock(in_channels, 16, bn=False),
            DCGANDiscriminatorBlock(16, 32, bn=False),
            DCGANDiscriminatorBlock(32, 64, bn=False),
            DCGANDiscriminatorBlock(64, 128, bn=False)
        )
        self.linear = nn.Sequential(
            nn.Linear(128 * round((image_size / (2 ** 4))) ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class GeneratorResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class CriticResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class ResidualWGANGenerator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.linear = nn.Sequential(
            nn.Linear(z_dim, z_dim * 4 * 4)
        )
        self.residual = nn.Sequential(
            GeneratorResidualBlock(128, 128),
            GeneratorResidualBlock(128, 128),
            GeneratorResidualBlock(128, 128)
        )
        self.output_layer = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), self.z_dim, 4, 4)
        x = self.residual(x)
        x = self.output_layer(x)
        return x


class ResidualWGANCritic(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.residual = nn.Sequential(
            CriticResidualBlock(in_channels, 128),
            CriticResidualBlock(128, 128)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.residual(x)
        x = self.conv(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    batch_size = 64
    z_dim = 128
    out_shape = (3, 32, 32)

    z = torch.randn(batch_size, z_dim)

    generator = WGANGenerator(z_dim=z_dim, out_shape=out_shape)
    critic = WGANCritic(in_dim=out_shape)

    generator_output = generator(z)
    critic_output = critic(generator_output)

    print(f"Generator Output Shape {generator_output.shape}. Critic Output Shape {critic_output.shape}.")

    dc_gan_generator = DCGANGenerator(z_dim, out_shape[1], out_shape[0])
    dc_gan_discriminator = DCGANDiscriminator(out_shape[1], out_shape[0])

    dc_gan_generator_output = dc_gan_generator(z)
    dc_gan_discriminator_output = dc_gan_discriminator(dc_gan_generator_output)

    print(f"DCGAN Generator Output Shape {dc_gan_generator_output.shape}. "
          f"DCGAN Discriminator Output Shape {dc_gan_discriminator_output.shape}.")
