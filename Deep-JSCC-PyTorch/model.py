"""
model.py
Baseline JSCC + FIS-Enhanced JSCC
"""

import torch
import torch.nn as nn

from fis_modules import (
    FIS_ImportanceAssessment,
    FIS_BitAllocation,
    STEQuantizer
)


# ============================================================
# BASELINE MODEL (UNCHANGED)
# ============================================================

class JSCC(nn.Module):
    """Original JSCC model (baseline)"""

    def __init__(self, C=16, channel_num=16):
        super(JSCC, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, C, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, channel_num, 5, stride=1, padding=2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channel_num, C, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C, C, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C, C, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C, C, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C, 3, 5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# ============================================================
# FIS-ENHANCED MODEL
# ============================================================

class JSCC_FIS(nn.Module):
    """FIS-Enhanced JSCC model"""

    def __init__(self, C=16, channel_num=16,
                 min_bits=4, max_bits=12):

        super(JSCC_FIS, self).__init__()

        # Encoder (same as baseline)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, C, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, channel_num, 5, stride=1, padding=2),
        )

        # Decoder (same as baseline)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channel_num, C, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C, C, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C, C, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C, C, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C, 3, 5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

        # ===== FIS MODULES =====
        self.fis_importance = FIS_ImportanceAssessment(channel_num)
        self.fis_allocation = FIS_BitAllocation(
            min_bits=min_bits,
            max_bits=max_bits
        )

        self.quantizer = STEQuantizer()

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self,
                x,
                snr=10.0,
                return_info=False):

        """
        Args:
            x: (B,3,H,W)
            snr: float or tensor
            return_info: return debug info

        Returns:
            encoded_q, decoded, (optional info)
        """

        device = x.device

        if not torch.is_tensor(snr):
            snr = torch.tensor(snr, device=device)

        # -----------------------
        # Encode
        # -----------------------
        encoded = self.encoder(x)  # (B, channel_num, H', W')

        # -----------------------
        # FIS Importance
        # -----------------------
        importance_map = self.fis_importance(encoded)  # (B,1,H',W')

        # -----------------------
        # Bit Allocation
        # -----------------------
        bit_allocation = self.fis_allocation(
            importance_map,
            snr
        )  # (B,1,H',W')

        # -----------------------
        # Quantization
        # -----------------------
        encoded_q = self.quantizer(
            encoded,
            bit_allocation
        )

        # -----------------------
        # Decode
        # -----------------------
        decoded = self.decoder(encoded_q)

        if return_info:
            info = {
                "importance_map": importance_map,
                "bit_allocation": bit_allocation,
                "avg_bits": bit_allocation.mean().item()
            }
            return encoded_q, decoded, info

        return encoded_q, decoded
