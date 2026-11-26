"""DDPM: linear noise schedule, closed-form forward process, class-conditional U-Net and sampler.
Time steps are 1-based (t = 1..T); schedule arrays are stored 0-based and read at t - 1."""
import copy
import torch
from diffusers import UNet2DModel


class LinearNoiseSchedule:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        to = lambda v: v.float().to(device)
        self.betas, self.alphas, self.alpha_bars = to(betas), to(alphas), to(alpha_bars)
        self.sqrt_alpha_bars = to(alpha_bars.sqrt())
        self.sqrt_one_minus_alpha_bars = to((1 - alpha_bars).sqrt())
        self.sigmas = to(betas.sqrt())

    def alpha_bar(self, t):
        return self.alpha_bars[t - 1]

    def q_sample(self, x0, t, noise):
        """x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps for a (N,) tensor of steps t in 1..T (t = 0 returns x0)."""
        i = (t - 1).clamp(min=0)
        a = torch.where(t > 0, self.sqrt_alpha_bars[i], torch.ones_like(self.sqrt_alpha_bars[i])).view(-1, 1, 1, 1)
        b = torch.where(t > 0, self.sqrt_one_minus_alpha_bars[i], torch.zeros_like(a.view(-1))).view(-1, 1, 1, 1)
        return a * x0 + b * noise


def build_unet():
    """Exactly the constructor given in the assignment (random initialisation)."""
    return UNet2DModel(
        sample_size=28, in_channels=1, out_channels=1, layers_per_block=2,
        block_out_channels=(32, 64, 128),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        num_class_embeds=40)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)


@torch.no_grad()
def ddpm_sample(model, sched, labels, generator=None, record_steps=(), shape=(1, 28, 28)):
    """DDPM Algorithm 2. Returns x_0 clipped to [-1, 1] and {t: x_t} for the requested t (0 = output)."""
    device = labels.device
    n = len(labels)
    x = torch.randn((n, *shape), generator=generator, device=device)
    trajectory = {}
    for t in range(sched.T, 0, -1):
        if t in record_steps:
            trajectory[t] = x.clone()
        tt = torch.full((n,), t, device=device, dtype=torch.long)
        eps = model(x, tt, class_labels=labels).sample.float()
        i = t - 1
        x = (x - (1 - sched.alphas[i]) / sched.sqrt_one_minus_alpha_bars[i] * eps) / sched.alphas[i].sqrt()
        if t > 1:
            x = x + sched.sigmas[i] * torch.randn(x.shape, generator=generator, device=device)
    x = x.clamp(-1, 1)
    if 0 in record_steps:
        trajectory[0] = x.clone()
    return x, trajectory
