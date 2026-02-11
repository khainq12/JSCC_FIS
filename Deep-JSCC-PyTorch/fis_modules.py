"""
Differentiable FIS Modules for Deep JSCC
Author: Modified for research-grade integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================================================
# 1️⃣ Importance Assessment (Differentiable FIS Style)
# ==========================================================

class FIS_ImportanceAssessment(nn.Module):
    """
    Differentiable fuzzy-style importance estimator

    Input:  (B, C, H, W)
    Output: (B, H, W) importance ∈ [0,1]
    """

    def __init__(self, channels):
        super().__init__()

        self.channels = channels

        # Learnable fuzzy fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, features):
        B, C, H, W = features.shape

        # ----- Descriptors -----
        mag = torch.norm(features, dim=1, keepdim=True) / math.sqrt(C)
        var = torch.var(features, dim=1, keepdim=True)
        std = torch.std(features, dim=1, keepdim=True)

        # Normalize
        var = var / (torch.mean(features ** 2, dim=1, keepdim=True) + 1e-6)
        std = std / (torch.mean(torch.abs(features), dim=1, keepdim=True) + 1e-6)

        mag = torch.clamp(mag, 0, 1)
        var = torch.clamp(var, 0, 1)
        std = torch.clamp(std, 0, 1)

        fuzzy_input = torch.cat([mag, var, std], dim=1)

        importance = self.fusion(fuzzy_input)

        return importance.squeeze(1)


# ==========================================================
# 2️⃣ Bit Allocation Module
# ==========================================================

class FIS_BitAllocation(nn.Module):
    """
    Differentiable bit allocation

    Input:
        importance_map (B, H, W)
        SNR_dB (float)
        target_rate (float)

    Output:
        bits (B, H, W)  ∈ [min_bits, max_bits]
    """

    def __init__(self, min_bits=4, max_bits=12):
        super().__init__()

        self.min_bits = min_bits
        self.max_bits = max_bits

        self.mapping = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, importance_map, SNR_dB, target_rate=0.5):
        B, H, W = importance_map.shape
        device = importance_map.device

        snr_norm = torch.tensor(SNR_dB / 30.0, device=device)
        rate_norm = torch.tensor(target_rate, device=device)

        snr_tensor = snr_norm.expand_as(importance_map)
        rate_tensor = rate_norm.expand_as(importance_map)

        stacked = torch.stack(
            [importance_map, snr_tensor, rate_tensor], dim=-1
        )  # (B,H,W,3)

        bits = self.mapping(stacked).squeeze(-1)

        bits = self.min_bits + bits * (self.max_bits - self.min_bits)

        return bits


# ==========================================================
# 3️⃣ Adaptive Quantizer (Straight-Through Estimator)
# ==========================================================

class AdaptiveQuantizer(nn.Module):
    """
    Differentiable uniform quantizer with STE

    Input:
        features (B,C,H,W)
        bit_allocation (B,H,W)

    Output:
        quantized features (B,C,H,W)
    """

    def __init__(self):
        super().__init__()

    def forward(self, features, bit_allocation):

        B, C, H, W = features.shape

        bits = bit_allocation.unsqueeze(1)  # (B,1,H,W)

        levels = torch.pow(2.0, bits)

        f_min = features.amin(dim=1, keepdim=True)
        f_max = features.amax(dim=1, keepdim=True)

        denom = (f_max - f_min) + 1e-6

        f_norm = (features - f_min) / denom

        # ----- Quantization -----
        f_quant = torch.round(f_norm * (levels - 1)) / (levels - 1)

        # Straight-Through Estimator
        f_quant = f_norm + (f_quant - f_norm).detach()

        f_dequant = f_quant * denom + f_min

        return f_dequant


# ==========================================================
# 4️⃣ Complete HA-FIS Module (Ready to Plug into Encoder)
# ==========================================================

class HAFIS_Module(nn.Module):
    """
    Complete hierarchical adaptive FIS block

    Pipeline:
        features → importance → bit allocation → quantization
    """

    def __init__(self, channels, min_bits=4, max_bits=12):
        super().__init__()

        self.importance_net = FIS_ImportanceAssessment(channels)
        self.bit_allocator = FIS_BitAllocation(min_bits, max_bits)
        self.quantizer = AdaptiveQuantizer()

    def forward(self, features, snr_dB, target_rate=0.5):

        importance = self.importance_net(features)

        bits = self.bit_allocator(importance, snr_dB, target_rate)

        quantized = self.quantizer(features, bits)

        return quantized, importance, bits
