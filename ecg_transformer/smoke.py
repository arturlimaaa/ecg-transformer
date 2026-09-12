import torch
from torch import nn


class TinyCNN(nn.Module):
    def __init__(self, n_leads: int = 12, n_classes: int = 5) -> None:
        super().__init__()
        self.conv = nn.Conv1d(n_leads, 16, kernel_size=7, padding=3)
        self.head = nn.Linear(16, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x))
        return self.head(h.mean(dim=-1))


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_synthetic(
    steps: int = 20,
    batch_size: int = 8,
    seed: int = 0,
    device: torch.device | None = None,
) -> list[float]:
    device = device or pick_device()
    torch.manual_seed(seed)
    model = TinyCNN().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    x = torch.randn(batch_size, 12, 1000, device=device)
    y = torch.randint(0, 2, (batch_size, 5), device=device).float()
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses
