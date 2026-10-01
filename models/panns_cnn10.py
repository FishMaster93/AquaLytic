"""PANNs Cnn10 backbone with an optional frozen random-projection buffer.

The buffer (``random_proj``) is the high-dimensional feature expansion used
by the closed-form ridge-regression head: it is initialised once with
``kaiming_normal_`` (std = sqrt(2 / 512), i.e. N(0, 1/256) regardless of the
output dimension) and then permanently frozen, so it never breaks the
"incremental accumulation == joint training" equivalence that the closed-form
update relies on.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import ConvBlock, init_layer


class PANNS_Cnn10(nn.Module):
    def __init__(self, classes_num=4, random_proj_dim=None, random_proj_seed=42):
        super(PANNS_Cnn10, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)

        self.random_proj_dim = random_proj_dim
        if random_proj_dim is not None:
            self.random_proj = nn.Linear(512, random_proj_dim, bias=True)
            gen = torch.Generator().manual_seed(random_proj_seed)
            nn.init.kaiming_normal_(self.random_proj.weight, generator=gen)
            nn.init.zeros_(self.random_proj.bias)
            for p in self.random_proj.parameters():
                p.requires_grad = False
            head_in = random_proj_dim
            # Optional PCA whitening applied before the projection. Defaults
            # to identity; call fit_whitening() with Phase-0 raw features to
            # enable it. Ablated in the paper (slightly worse than plain
            # random projection) and off by default.
            self.register_buffer('whiten_mu', torch.zeros(512))
            self.register_buffer('whiten_W', torch.eye(512))
        else:
            head_in = 512

        self.fc_audioset = nn.Linear(head_in, classes_num, bias=True)
        init_layer(self.fc_audioset)

    def _backbone_forward(self, x):
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu_(self.fc1(x))
        return x

    def forward(self, x):
        x = self._backbone_forward(x)

        if self.random_proj_dim is not None:
            x_whitened = (x - self.whiten_mu) @ self.whiten_W.T
            embedding = F.relu(self.random_proj(x_whitened))
        else:
            embedding = x

        clipwise_output = self.fc_audioset(embedding)
        return clipwise_output, embedding

    @torch.no_grad()
    def extract_raw_feature(self, x):
        """Same as forward() up to fc1, skipping whitening/projection --
        used to fit the whitening statistics from Phase-0 real data."""
        return self._backbone_forward(x)

    @torch.no_grad()
    def fit_whitening(self, raw_embeddings, eps=1e-3):
        """PCA whitening (Sigma^{-1/2}) fit once on Phase-0 real data, then
        frozen alongside random_proj. Off by default -- see note above."""
        mu = raw_embeddings.mean(dim=0)
        centered = raw_embeddings - mu
        cov = (centered.T @ centered) / (raw_embeddings.shape[0] - 1)
        eigval, eigvec = torch.linalg.eigh(cov)
        eigval = eigval.clamp(min=eps)
        w = eigvec @ torch.diag(eigval.rsqrt()) @ eigvec.T
        self.whiten_mu.copy_(mu.to(self.whiten_mu.device))
        self.whiten_W.copy_(w.to(self.whiten_W.device))


def load_pretrained_cnn10(model, ckpt_path):
    """Load official AudioSet-pretrained PANNs Cnn10 weights (conv_block1-4 +
    fc1), skipping fc_audioset (527-way AudioSet head vs. our 3-logit CORAL
    head -- shape mismatch, and it needs retraining anyway)."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    pretrained_sd = ckpt['model'] if 'model' in ckpt else ckpt
    own_sd = model.state_dict()
    loaded = {
        k: v for k, v in pretrained_sd.items()
        if k in own_sd and v.shape == own_sd[k].shape
    }
    own_sd.update(loaded)
    model.load_state_dict(own_sd)
    return sorted(loaded.keys())
