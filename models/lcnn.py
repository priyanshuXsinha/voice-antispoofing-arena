"""models/lcnn.py — Light CNN (LCNN) with Max-Feature-Map activation for anti-spoofing.

Run 1 model. Takes LFCC features (1, 3*n_lfcc, T) → binary logits.
Reference: Wu et al., "Light CNN for Deep Face Representation with Noisy Labels" (adapted for speech).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Max Feature Map (MFM) activation
# ─────────────────────────────────────────────────────────────

class MFM(nn.Module):
    """
    Max-Feature-Map activation: splits channel dim in half and takes
    element-wise max. Halves channel count, acts as a competitive activation.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, conv_type: str = "conv2d"):
        super().__init__()
        if conv_type == "conv2d":
            self.conv = nn.Conv2d(
                in_channels, 2 * out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding, bias=False
            )
        elif conv_type == "linear":
            self.conv = nn.Linear(in_channels, 2 * out_channels, bias=False)
        self.conv_type = conv_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.conv_type == "conv2d":
            # Split along channel dim
            a, b = torch.split(x, x.size(1) // 2, dim=1)
        else:
            # Split along last dim for linear layers
            a, b = torch.split(x, x.size(-1) // 2, dim=-1)
        return torch.max(a, b)


# ─────────────────────────────────────────────────────────────
# LCNN
# ─────────────────────────────────────────────────────────────

class LCNN(nn.Module):
    """
    Light CNN for audio anti-spoofing (binary classifier).
    Input:  (batch, 1, n_feat, time)  — typically (B, 1, 180, 400) for 60*3 LFCC
    Output: (batch, 2) logits
    """

    def __init__(
        self,
        input_channels: int = 1,
        dropout: float = 0.5,
        nb_classes: int = 2,
    ):
        super().__init__()

        # Block 1
        self.block1 = nn.Sequential(
            MFM(input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2
        self.block2 = nn.Sequential(
            MFM(32, 48, kernel_size=1, stride=1, padding=0),
            MFM(48, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
        )

        # Block 3
        self.block3 = nn.Sequential(
            MFM(64, 64, kernel_size=1, stride=1, padding=0),
            MFM(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
        )

        # Block 4
        self.block4 = nn.Sequential(
            MFM(64, 64, kernel_size=1, stride=1, padding=0),
            MFM(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )

        # Block 5
        self.block5 = nn.Sequential(
            MFM(64, 32, kernel_size=1, stride=1, padding=0),
            MFM(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
        )

        # Classifier head (adaptive pool + FC)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout        = nn.Dropout(p=dropout)

        # MFM FC layers
        self.fc1 = nn.Linear(64 * 4 * 4, 2 * 160)
        self.fc2 = nn.Linear(160, nb_classes)

    def _mfm_fc(self, x: torch.Tensor) -> torch.Tensor:
        """Max-Feature-Map for fully-connected layer."""
        a, b = torch.split(x, x.size(-1) // 2, dim=-1)
        return torch.max(a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, n_feat, time)
        returns: (B, 2) logits
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)         # flatten
        x = self.dropout(x)

        x = self.fc1(x)
        x = self._mfm_fc(x)                # MFM on FC1
        x = self.dropout(x)

        x = self.fc2(x)
        return x   # (B, 2) logits

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate-layer embedding (before final FC) for analysis."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self._mfm_fc(x)   # (B, 160)
        return x


if __name__ == "__main__":
    model = LCNN()
    dummy = torch.randn(4, 1, 180, 400)   # B=4, 60*3 LFCC, 400 time frames
    out = model(dummy)
    print(f"LCNN output shape: {out.shape}")   # (4, 2)
    emb = model.get_embedding(dummy)
    print(f"LCNN embedding shape: {emb.shape}") # (4, 160)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LCNN trainable params: {n_params:,}")
