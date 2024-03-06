import torch
import torch.nn as nn


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.linear_block(x)
        return x


class Generator(nn.Module):
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


class Critic(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim[0] * in_dim[1] * in_dim[2], 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.model(x)
        return x


if __name__ == "__main__":
    batch_size = 64
    z_dim = 128
    out_shape = (3, 32, 32)

    generator = Generator(z_dim, out_shape)
    critic = Critic(out_shape)

    z = torch.randn(batch_size, z_dim)

    fake_output = generator(z)
    output = critic(fake_output)

    print(f"Generator Output Shape {fake_output.shape}. Critic Output Shape {output.shape}.")
