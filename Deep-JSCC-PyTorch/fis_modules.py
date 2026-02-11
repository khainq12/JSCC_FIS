"""
Differentiable FIS Modules for Deep JSCC
----------------------------------------
- Layer 1: Importance Assessment (Soft Fuzzy)
- Layer 2: Bit Allocation (Soft Fuzzy + Target Rate)
- STE Quantizer

Fully vectorized
Fully differentiable
No pixel-wise loops
Trainable end-to-end
"""

import torch
import torch.nn as nn


# ============================================================
# SOFT FUZZY MEMBERSHIP FUNCTIONS
# ============================================================

class SoftMembership(nn.Module):
    """
    Differentiable triangular membership
    """

    def __init__(self, a, b, c):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    def forward(self, x):
        left = (x - self.a) / (self.b - self.a + 1e-8)
        right = (self.c - x) / (self.c - self.b + 1e-8)
        return torch.clamp(torch.min(left, right), 0.0, 1.0)


# ============================================================
# FIS LAYER 1: IMPORTANCE ASSESSMENT
# ============================================================

class FIS_ImportanceAssessment(nn.Module):
    """
    Input:  (B, C, H, W)
    Output: (B, 1, H, W) importance in [0,1]
    """

    def __init__(self, channels):
        super().__init__()

        self.compress = nn.Conv2d(channels, 1, 1)

        self.low = SoftMembership(0.0, 0.0, 0.5)
        self.medium = SoftMembership(0.3, 0.5, 0.7)
        self.high = SoftMembership(0.5, 1.0, 1.0)

        self.w_low = nn.Parameter(torch.tensor(0.2))
        self.w_medium = nn.Parameter(torch.tensor(0.5))
        self.w_high = nn.Parameter(torch.tensor(0.8))

    def forward(self, x):

        mag = torch.abs(self.compress(x))
        mag = torch.sigmoid(mag)

        mu_low = self.low(mag)
        mu_med = self.medium(mag)
        mu_high = self.high(mag)

        numerator = (
            mu_low * self.w_low +
            mu_med * self.w_medium +
            mu_high * self.w_high
        )

        denominator = mu_low + mu_med + mu_high + 1e-8

        importance = numerator / denominator

        return importance


# ============================================================
# FIS LAYER 2: BIT ALLOCATION
# ============================================================

class FIS_BitAllocation(nn.Module):
    """
    Input:
        importance (B,1,H,W)
        SNR_dB (scalar or tensor)
        target_rate (optional)

    Output:
        bits_map (B,1,H,W)
    """

    def __init__(self, min_bits=4, max_bits=12):
        super().__init__()

        self.min_bits = min_bits
        self.max_bits = max_bits

        self.low = SoftMembership(0.0, 0.0, 0.5)
        self.medium = SoftMembership(0.3, 0.5, 0.7)
        self.high = SoftMembership(0.5, 1.0, 1.0)

        self.b_low = nn.Parameter(torch.tensor(4.0))
        self.b_medium = nn.Parameter(torch.tensor(8.0))
        self.b_high = nn.Parameter(torch.tensor(12.0))

    def forward(self, importance, SNR_dB, target_rate=None):

        device = importance.device

        # -----------------------
        # Handle SNR
        # -----------------------
        if not torch.is_tensor(SNR_dB):
            SNR_dB = torch.tensor(SNR_dB, device=device)

        snr_norm = torch.clamp(SNR_dB / 30.0, 0.0, 1.0)

        # broadcast to match spatial dims
        snr_norm = snr_norm.view(-1, 1, 1, 1)

        importance = importance * snr_norm

        # -----------------------
        # Fuzzy inference
        # -----------------------
        mu_low = self.low(importance)
        mu_med = self.medium(importance)
        mu_high = self.high(importance)

        numerator = (
            mu_low * self.b_low +
            mu_med * self.b_medium +
            mu_high * self.b_high
        )

        denominator = mu_low + mu_med + mu_high + 1e-8

        bits = numerator / denominator

        # -----------------------
        # Global Target Rate
        # -----------------------
        if target_rate is not None:

            if not torch.is_tensor(target_rate):
                target_rate = torch.tensor(target_rate, device=device)

            avg_bits = bits.mean()

            scale = target_rate / (avg_bits + 1e-8)

            bits = bits * scale

        bits = torch.clamp(bits, self.min_bits, self.max_bits)

        return bits


# ============================================================
# STE QUANTIZER
# ============================================================

class STEQuantizer(nn.Module):
    """
    Straight Through Estimator Quantizer
    """

    def forward(self, x, bits):

        levels = torch.pow(2.0, bits)

        x_min = x.amin(dim=1, keepdim=True)
        x_max = x.amax(dim=1, keepdim=True)

        x_norm = (x - x_min) / (x_max - x_min + 1e-8)

        x_q = torch.round(x_norm * (levels - 1)) / (levels - 1)

        # Straight Through Estimator
        x_q = x_norm + (x_q - x_norm).detach()

        x_deq = x_q * (x_max - x_min) + x_min

        return x_deq
