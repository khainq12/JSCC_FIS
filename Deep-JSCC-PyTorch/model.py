# -*- coding: utf-8 -*-
"""
model.py
Original JSCC + FIS-Enhanced JSCC version
"""

import torch
import torch.nn as nn

# ====== Import FIS modules (NEW) ======
from fis_modules import (
    FIS_ImportanceAssessment,
    FIS_BitAllocation,
    AdaptiveQuantizer
)


# ============================================================
# ====================== BASELINE JSCC ========================
# ============================================================

class JSCC(nn.Module):
    """
    Original JSCC model (kept unchanged for fair comparison)
    """

    def __init__(self, C=16, channel_num=16):
        super(JSCC, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(C, C, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(C, C, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(C, C, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(C, channel_num, kernel_size=5, stride=1, padding=2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channel_num, C, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(C, C, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(C, C, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(C, C, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(C, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# ============================================================
# ====================== FIS-ENHANCED JSCC ===================
# ============================================================

class JSCC_FIS(nn.Module):
    """
    FIS-Enhanced JSCC model
    Adds:
        - Importance Assessment
        - Adaptive Bit Allocation
        - Adaptive Quantization
    """

    def __init__(self, C=16, channel_num=16):
        super(JSCC_FIS, self).__init__()

        # ===== Same Encoder as baseline =====
        self.encoder = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(C, C, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(C, C, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(C, C, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(C, channel_num, kernel_size=5, stride=1, padding=2),
        )

        # ===== Same Decoder as baseline =====
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channel_num, C, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(C, C, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(C, C, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(C, C, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(C, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

        # ===== FIS Modules (NEW) =====
        self.fis_importance = FIS_ImportanceAssessment()
        self.fis_allocation = FIS_BitAllocation()
        self.quantizer = AdaptiveQuantizer()

    def forward(self, x, snr=10.0, target_rate=0.5, return_info=False):
        """
        Forward pass with FIS adaptive encoding

        Args:
            x: (B, 3, H, W)
            snr: channel SNR in dB
            target_rate: rate budget [0, 1]
            return_info: return intermediate maps

        Returns:
            encoded_quantized
            decoded
            info (optional)
        """

        # ===== 1. Encode =====
        encoded = self.encoder(x)  # (B, C, H', W')

        # ===== 2. Importance Assessment =====
        importance_map = self.fis_importance(encoded)  # (B, H', W')

        # ===== 3. Bit Allocation =====
        bit_allocation = self.fis_allocation(
            importance_map,
            snr,
            target_rate
        )  # (B, H', W')

        # ===== 4. Adaptive Quantization =====
        encoded_quantized = self.quantizer(
            encoded,
            bit_allocation
        )  # (B, C, H', W')

        # ===== 5. Decode =====
        decoded = self.decoder(encoded_quantized)

        if return_info:
            info = {
                "importance_map": importance_map,
                "bit_allocation": bit_allocation,
                "avg_bits": bit_allocation.float().mean().item()
            }
            return encoded_quantized, decoded, info

        return encoded_quantized, decoded
