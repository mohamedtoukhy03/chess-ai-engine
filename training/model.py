"""
Chess Evaluation CNN — Residual Network Architecture

Takes a (18, 8, 8) encoded board as input:
  - 12 planes: one per piece type per color (P,N,B,R,Q,K × white,black)
  - 6 planes: side-to-move, castling rights, en-passant square
  - Each plane is 8×8 (the board)

Outputs a single scalar evaluation in [-1, 1]:
  - +1.0 = winning for white
  -  0.0 = equal
  - -1.0 = winning for black
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_RESIDUAL_BLOCKS, NUM_FILTERS, FC_HIDDEN, INPUT_CHANNELS


class ResBlock(nn.Module):
    """Residual block with two 3×3 convolutions and batch norm."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Global average pooling
        y = x.view(b, c, -1).mean(dim=2)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1)
        return x * y


class SEResBlock(nn.Module):
    """Residual block with Squeeze-and-Excitation."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.se    = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + residual)
        return out


class ChessEvalNet(nn.Module):
    """
    CNN for chess position evaluation.

    Architecture:
      1. Input conv: 12 → NUM_FILTERS channels (3×3)
      2. N SE-Residual blocks
      3. Global average pooling
      4. FC layers → single tanh output
    """

    def __init__(self):
        super().__init__()

        # Input convolution
        self.input_conv = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, NUM_FILTERS, 3, padding=1, bias=False),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.ReLU(inplace=True),
        )

        # Residual tower
        self.res_tower = nn.Sequential(
            *[SEResBlock(NUM_FILTERS) for _ in range(NUM_RESIDUAL_BLOCKS)]
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, 1, 1, bias=False),  # 1×1 conv to reduce channels
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 8, FC_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(FC_HIDDEN, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 18, 8, 8) encoded board state

        Returns:
            (batch, 1) evaluation score in [-1, 1]
        """
        x = self.input_conv(x)
        x = self.res_tower(x)
        x = self.value_head(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ChessEvalNet()
    print(f"ChessEvalNet — {count_parameters(model):,} trainable parameters")
    print(model)

    # Test forward pass
    dummy = torch.randn(1, 18, 8, 8)
    out = model(dummy)
    print(f"\nInput shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output value: {out.item():.4f}")
