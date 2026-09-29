import torch
from torch import nn

PATCH_SAMPLES = 8
D_MODEL = 64
NHEAD = 4
N_LAYERS = 2
DIM_FEEDFORWARD = 128


class Transformer(nn.Module):
    """Supervised encoder. No reconstruction head. 8-sample patches are 80 ms at 100 Hz."""

    def __init__(self, n_leads: int = 12, n_classes: int = 5) -> None:
        super().__init__()
        self.patch_samples = PATCH_SAMPLES
        self.embed = nn.Linear(n_leads * PATCH_SAMPLES, D_MODEL)
        n_tokens = 1000 // PATCH_SAMPLES
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, D_MODEL))
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=N_LAYERS, enable_nested_tensor=False
        )
        self.head = nn.Linear(D_MODEL, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, leads, time = x.shape
        p = self.patch_samples
        tokens = x.reshape(b, leads, time // p, p).permute(0, 2, 1, 3).reshape(b, time // p, leads * p)
        h = self.encoder(self.embed(tokens) + self.pos)
        return self.head(h.mean(dim=1))
