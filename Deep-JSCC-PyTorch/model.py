"""
Modified model.py to add FIS-enhanced version (FIXED VERSION)
"""

import torch
import torch.nn as nn
from fis_modules import FIS_ImportanceAssessment, FIS_BitAllocation, AdaptiveQuantizer


# ============================================================
# Baseline JSCC
# ============================================================

class JSCC(nn.Module):
    """Original JSCC model (baseline)"""

    def __init__(self, C=16, channel_num=16):
        super(JSCC, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, C, 5, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 5, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 5, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 5, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, channel_num, 5, 1, 2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channel_num, C, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C, C, 5, 2, 2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C, C, 5, 2, 2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C, C, 5, 2, 2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C, 3, 5, 2, 2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# ============================================================
# FIS-Enhanced JSCC
# ============================================================

class JSCC_FIS(nn.Module):
    """FIS-Enhanced JSCC model"""

    def __init__(self, C=16, channel_num=16):
        super(JSCC_FIS, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, C, 5, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 5, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 5, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 5, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, channel_num, 5, 1, 2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channel_num, C, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C, C, 5, 2, 2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C, C, 5, 2, 2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C, C, 5, 2, 2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C, 3, 5, 2, 2, output_padding=1),
            nn.Sigmoid()
        )

        # FIS modules
        self.fis_importance = FIS_ImportanceAssessment()
        self.fis_allocation = FIS_BitAllocation()
        self.quantizer = AdaptiveQuantizer()

    # --------------------------------------------------------

    def forward(self, x, snr=10.0, target_rate=0.5, return_info=False):
        """
        Args:
            x: (B, 3, H, W)
            snr: float or tensor
            target_rate: float in [0,1]
        """

        device = x.device

        # Convert scalars to tensors safely
        if not torch.is_tensor(snr):
            snr = torch.tensor(snr, dtype=torch.float32, device=device)

        if not torch.is_tensor(target_rate):
            target_rate = torch.tensor(target_rate, dtype=torch.float32, device=device)

        target_rate = torch.clamp(target_rate, 0.0, 1.0)

        # -----------------------
        # Encode
        # -----------------------
        encoded = self.encoder(x)  # (B, C, H', W')

        # -----------------------
        # FIS Importance
        # -----------------------
        importance_map = self.fis_importance(encoded)  # (B, H', W')

        # -----------------------
        # Bit Allocation
        # -----------------------
        bit_allocation = self.fis_allocation(
            importance_map,
            snr,
            target_rate
        )  # (B, H', W')

        # Ensure broadcast compatibility
        if bit_allocation.dim() == 3:
            bit_allocation = bit_allocation.unsqueeze(1)  # (B,1,H',W')

        # -----------------------
        # Quantization
        # -----------------------
        encoded_quantized = self.quantizer(
            encoded,
            bit_allocation
        )  # (B,C,H',W')

        # -----------------------
        # Decode
        # -----------------------
        decoded = self.decoder(encoded_quantized)

        # -----------------------
        # Return
        # -----------------------
        if return_info:
            info = {
                "importance_map": importance_map,
                "bit_allocation": bit_allocation,
                "avg_bits": bit_allocation.float().mean().detach().item()
            }
            return encoded_quantized, decoded, info

        return encoded_quantized, decoded
