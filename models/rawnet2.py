"""models/rawnet2.py — RawNet2 end-to-end anti-spoofing classifier.

Run 2 model. Operates directly on raw waveform → binary logits.
Reference: Tak et al., "End-to-End anti-spoofing with RawNet2" (ICASSP 2021).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Sinc convolution (learnable band-pass filter bank)
# ─────────────────────────────────────────────────────────────

class SincConv(nn.Module):
    """
    Sinc-based bandpass filter bank applied to raw waveform.
    Each filter is parameterised by (f1, bandwidth) ensuring physical
    interpretability: f2 = f1 + |bw|, filter = sinc(2πf2t) - sinc(2πf1t).
    """

    def __init__(
        self,
        out_channels: int,      # number of filters
        kernel_size: int,       # must be odd
        sample_rate: int = 16000,
        in_channels: int = 1,
        stride: int = 1,
        padding: int = 0,
        min_low_hz: float = 50.0,
        min_band_hz: float = 50.0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size  = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.sample_rate  = sample_rate
        self.stride       = stride
        self.padding      = padding
        self.min_low_hz   = min_low_hz
        self.min_band_hz  = min_band_hz

        # Mel-spaced initialisation of filter frequencies
        low_hz   = 30.0
        high_hz  = sample_rate / 2.0 - (min_low_hz + min_band_hz)
        mel      = torch.linspace(
            self._hz_to_mel(low_hz), self._hz_to_mel(high_hz), out_channels + 1
        )
        hz       = self._mel_to_hz(mel)

        self.low_hz_  = nn.Parameter(hz[:-1].unsqueeze(1))   # (out, 1)
        self.band_hz_ = nn.Parameter((hz[1:] - hz[:-1]).unsqueeze(1))  # (out, 1)

        # Hamming window
        n = torch.arange(1, (self.kernel_size - 1) // 2 + 1, dtype=torch.float32)
        self.register_buffer("window_",
            0.54 - 0.46 * torch.cos(2 * math.pi * n / (self.kernel_size - 1))
        )
        self.register_buffer("n_",
            2 * math.pi * torch.arange(
                -(self.kernel_size - 1) // 2,
                0, dtype=torch.float32
            ).unsqueeze(0)
        )   # (1, half_size)

    @staticmethod
    def _hz_to_mel(hz): return 2595.0 * math.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (B, 1, T) → (B, out_channels, T')"""
        low  = self.min_low_hz  + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),
                           self.min_low_hz, self.sample_rate / 2.0)
        band = (high - low)[:, 0]  # (out,)

        # Normalised frequencies
        f_times_t = self.n_ / self.sample_rate   # (1, half)
        low_pass1  = 2 * low  * torch.sinc(2 * low  * f_times_t)   # (out, half)
        low_pass2  = 2 * high * torch.sinc(2 * high * f_times_t)   # (out, half)
        band_pass  = (low_pass2 - low_pass1) * self.window_        # (out, half)

        # Normalise by band energy
        band_pass  = band_pass / (2.0 * band.unsqueeze(1))

        # Build symmetric filter (half + zero + half reversed)
        filters = torch.cat([
            band_pass.flip(1),
            torch.zeros(self.out_channels, 1, device=band_pass.device),
            band_pass,
        ], dim=1).unsqueeze(1)   # (out, 1, kernel_size)

        return F.conv1d(waveform, filters,
                        stride=self.stride,
                        padding=self.padding,
                        bias=None,
                        groups=1)


# ─────────────────────────────────────────────────────────────
# Residual block
# ─────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.bn1   = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.mp    = nn.MaxPool1d(3)
        self.fms   = FMS(out_channels)

        # Shortcut
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.leaky_relu(self.bn1(x), negative_slope=0.3)
        out = self.conv1(out)
        out = F.leaky_relu(self.bn2(out), negative_slope=0.3)
        out = self.conv2(out)

        if self.downsample:
            identity = self.downsample(x)

        out = out + identity
        out = self.mp(out)
        out = self.fms(out)
        return out


class FMS(nn.Module):
    """Feature Map Scaling: channel-wise sigmoid-gated attention."""
    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        s = x.mean(dim=-1)              # (B, C)
        s = torch.sigmoid(self.fc(s))   # (B, C)
        return x * s.unsqueeze(-1)


# ─────────────────────────────────────────────────────────────
# RawNet2
# ─────────────────────────────────────────────────────────────

class RawNet2(nn.Module):
    """
    RawNet2 end-to-end anti-spoofing model.
    Input:  (batch, 1, T) raw waveform  (T ≈ 64000 for 4 sec @ 16kHz)
    Output: (batch, 2) logits
    """

    def __init__(
        self,
        sinc_filters:       int = 128,
        sinc_filter_length: int = 1024,
        filts:              list = None,
        gru_node:           int = 1024,
        nb_gru_layer:       int = 3,
        nb_classes:         int = 2,
        dropout:            float = 0.5,
        sample_rate:        int = 16000,
    ):
        super().__init__()

        if filts is None:
            filts = [20, [20, 20], [20, 128], [128, 128]]

        # Sinc filter layer
        self.sinc_conv = SincConv(
            out_channels=sinc_filters,
            kernel_size=sinc_filter_length,
            sample_rate=sample_rate,
            padding=sinc_filter_length // 2,
        )
        self.bn_sinc    = nn.BatchNorm1d(sinc_filters)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResBlock(sinc_filters, filts[1][0]),
            ResBlock(filts[1][0],  filts[1][1]),
            ResBlock(filts[2][0],  filts[2][1]),
            ResBlock(filts[3][0],  filts[3][1]),
        )

        # GRU
        self.gru = nn.GRU(
            input_size=filts[3][1],
            hidden_size=gru_node,
            num_layers=nb_gru_layer,
            batch_first=True,
            dropout=dropout if nb_gru_layer > 1 else 0.0,
        )

        self.fc    = nn.Linear(gru_node, gru_node)
        self.bn_fc = nn.BatchNorm1d(gru_node)
        self.sig   = nn.Sigmoid()

        self.classifier = nn.Linear(gru_node, nb_classes)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T) → logits (B, 2)"""
        # SincConv + abs + BN
        x = torch.abs(self.sinc_conv(x))
        x = F.max_pool1d(x, kernel_size=3)
        x = self.bn_sinc(x)
        x = F.leaky_relu(x, negative_slope=0.3)

        # Residual blocks
        x = self.res_blocks(x)

        # GRU: (B, C, T) → (B, T, C)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]   # last time step: (B, gru_node)

        # FC
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn_fc(x)
        x = self.sig(x)

        x = self.dropout(x)
        return self.classifier(x)   # (B, 2)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return GRU output embedding for speaker analysis."""
        x = torch.abs(self.sinc_conv(x))
        x = F.max_pool1d(x, kernel_size=3)
        x = self.bn_sinc(x)
        x = F.leaky_relu(x, negative_slope=0.3)
        x = self.res_blocks(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        return x[:, -1, :]   # (B, gru_node)


if __name__ == "__main__":
    model = RawNet2()
    # 4 sec @ 16kHz = 64000 samples
    dummy = torch.randn(2, 1, 64000)
    out   = model(dummy)
    print(f"RawNet2 output shape:    {out.shape}")     # (2, 2)
    emb   = model.get_embedding(dummy)
    print(f"RawNet2 embedding shape: {emb.shape}")     # (2, 1024)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"RawNet2 trainable params: {n_p:,}")
