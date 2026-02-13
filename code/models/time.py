import torch
import torch.nn as nn


class TimeDeltaProjection(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, deltas: torch.Tensor) -> torch.Tensor:
        scaled = torch.log1p(deltas).unsqueeze(-1)
        return self.proj(scaled)
