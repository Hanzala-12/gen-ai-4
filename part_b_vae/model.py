"""Convolutional VAE with a Bernoulli decoder for 28x28 characters."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten())
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1))

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        return mu + sigma * torch.randn_like(sigma)

    def decode(self, z):
        """Returns one logit per pixel, shape (N, 1, 28, 28)."""
        return self.dec(self.dec_fc(z).view(-1, 64, 7, 7))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n, device):
        z = torch.randn(n, self.latent_dim, device=device)
        return torch.sigmoid(self.decode(z))

    @torch.no_grad()
    def interpolate(self, x_a, x_b, steps=10):
        """Linear interpolation between the posterior means of two images."""
        mu_a, _ = self.encode(x_a)
        mu_b, _ = self.encode(x_b)
        w = torch.linspace(0, 1, steps, device=mu_a.device).unsqueeze(1)
        return torch.sigmoid(self.decode((1 - w) * mu_a + w * mu_b))


def kl_divergence(mu, logvar):
    """Closed-form KL(N(mu, sigma^2) || N(0, I)) per image, summed over latent dims."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def negative_elbo(logits, x, mu, logvar):
    """Returns (loss, reconstruction, kl), each averaged over the batch, in nats per image."""
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=(1, 2, 3))
    kl = kl_divergence(mu, logvar)
    return (rec + kl).mean(), rec.mean(), kl.mean()
