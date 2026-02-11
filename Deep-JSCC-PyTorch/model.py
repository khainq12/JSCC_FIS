"""
model.py
FIS-Enhanced Deep JSCC
Clean pipeline:
encode → quantize → (return encoded)
Decoder is called outside after channel
"""

import torch
import torch.nn as nn

from fis_modules import (
    FIS_ImportanceAssessment,
    FIS_BitAllocation,
    STEQuantizer
)


class JSCC_FIS(nn.Module):

    def __init__(self,
                 C=16,
                 channel_num=16,
                 min_bits=4,
                 max_bits=12):

        super(JSCC_FIS, self).__init__()

        # =======================
        # Encoder
        # =======================
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

        # =======================
        # Decoder
        # =======================
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

        # =======================
        # FIS
        # =======================
        self.fis_importance = FIS_ImportanceAssessment(channel_num)

        self.fis_allocation = FIS_BitAllocation(
            min_bits=min_bits,
            max_bits=max_bits
        )

        self.quantizer = STEQuantizer()

    # ========================================================
    # Forward (NO decode here)
    # ========================================================
    def forward(self,
                x,
                snr=10.0,
                target_rate=None,
                return_info=False):

        device = x.device

        if not torch.is_tensor(snr):
            snr = torch.tensor(snr, device=device)

        # Encode
        encoded = self.encoder(x)

        # Importance
        importance_map = self.fis_importance(encoded)

        # Bit allocation
        bit_allocation = self.fis_allocation(
            importance_map,
            snr,
            target_rate
        )

        # Quantization
        encoded_q = self.quantizer(encoded, bit_allocation)

        if return_info:
            info = {
                "importance_map": importance_map,
                "bit_allocation": bit_allocation,
                "avg_bits": bit_allocation.mean().item()
            }
            return encoded_q, None, info

        return encoded_q
