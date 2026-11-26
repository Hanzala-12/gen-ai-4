"""Masked autoencoder for 64x64 EuroSAT tiles (64 patches of 8x8x3)."""
import torch
import torch.nn as nn


def _transformer(width, heads, ff, depth):
    layer = nn.TransformerEncoderLayer(width, heads, ff, dropout=0.0, activation="gelu",
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)


class MAE(nn.Module):
    def __init__(self, num_patches=64, patch_dim=192, enc_dim=192, enc_depth=6, enc_heads=3,
                 enc_ff=768, dec_dim=128, dec_depth=2, dec_heads=4, dec_ff=512, mask_ratio=0.75):
        super().__init__()
        self.num_patches = num_patches
        self.n_keep = int(round(num_patches * (1 - mask_ratio)))

        self.patch_embed = nn.Linear(patch_dim, enc_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, enc_dim))
        self.encoder = _transformer(enc_dim, enc_heads, enc_ff, enc_depth)
        self.enc_norm = nn.LayerNorm(enc_dim)

        self.dec_embed = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, num_patches, dec_dim))
        self.decoder = _transformer(dec_dim, dec_heads, dec_ff, dec_depth)
        self.dec_pred = nn.Linear(dec_dim, patch_dim)
        self._init_weights()

    def _init_weights(self):
        for p in (self.pos_embed, self.dec_pos_embed, self.mask_token):
            nn.init.trunc_normal_(p, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def random_masking(self, x):
        """x: (N, L, D). Keeps n_keep random tokens per image.
        Returns visible tokens, the binary mask (1 = masked) and the inverse permutation."""
        N, L, D = x.shape
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :self.n_keep]
        x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(N, L, device=x.device)
        mask[:, :self.n_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_visible, mask, ids_restore

    def forward_encoder(self, patches):
        x = self.patch_embed(patches) + self.pos_embed
        x, mask, ids_restore = self.random_masking(x)
        return self.enc_norm(self.encoder(x)), mask, ids_restore

    def forward_decoder(self, latent, ids_restore):
        x = self.dec_embed(latent)
        N, n_vis, C = x.shape
        mask_tokens = self.mask_token.expand(N, ids_restore.shape[1] - n_vis, -1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, C))   # original order
        x = self.decoder(x + self.dec_pos_embed)
        return self.dec_pred(x)

    @staticmethod
    def masked_mse(pred, target, mask):
        per_patch = ((pred.float() - target.float()) ** 2).mean(dim=-1)
        return (per_patch * mask).sum() / mask.sum()

    def forward(self, patches):
        latent, mask, ids_restore = self.forward_encoder(patches)
        pred = self.forward_decoder(latent, ids_restore)
        return self.masked_mse(pred, patches, mask), pred, mask

    def features(self, patches):
        """All 64 patches, no masking; average of the encoder outputs -> (N, 192)."""
        x = self.patch_embed(patches) + self.pos_embed
        return self.enc_norm(self.encoder(x)).mean(dim=1)


class LinearProbe(nn.Module):
    def __init__(self, dim=192, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, f):
        return self.fc(f)
