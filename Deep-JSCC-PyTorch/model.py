"""
Modified model.py to add FIS-enhanced version
"""

import torch
import torch.nn as nn
from fis_modules import FIS_ImportanceAssessment, FIS_BitAllocation, AdaptiveQuantizer


class JSCC(nn.Module):
    """Original JSCC model (unchanged for baseline comparison)"""
    def __init__(self, C=16, channel_num=16):
        super(JSCC, self).__init__()
        
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


class JSCC_FIS(nn.Module):
    """FIS-Enhanced JSCC model"""
    def __init__(self, C=16, channel_num=16):
        super(JSCC_FIS, self).__init__()
        
        # Same encoder/decoder as baseline
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
        
        # FIS modules (NEW)
        self.fis_importance = FIS_ImportanceAssessment()
        self.fis_allocation = FIS_BitAllocation()
        self.quantizer = AdaptiveQuantizer()
    
    def forward(self, x, snr=10.0, target_rate=0.5, return_info=False):
        """
        Forward with FIS-based adaptive encoding
        
        Args:
            x: (B, 3, H, W) input images
            snr: channel SNR in dB
            target_rate: rate budget [0, 1]
            return_info: return intermediate info
        
        Returns:
            encoded: (B, channel_num, H', W') encoded features
            decoded: (B, 3, H, W) decoded images
            info: dict (if return_info=True)
        """
        # Encode
        encoded = self.encoder(x)  # (B, channel_num, H', W')
        
        # FIS processing
        importance_map = self.fis_importance(encoded)  # (B, H', W')
        bit_allocation = self.fis_allocation(importance_map, snr, target_rate)  # (B, H', W')
        encoded_quantized = self.quantizer(encoded, bit_allocation)  # (B, channel_num, H', W')
        
        # Decode
        decoded = self.decoder(encoded_quantized)  # (B, 3, H, W)
        
        if return_info:
            info = {
                'importance_map': importance_map,
                'bit_allocation': bit_allocation,
                'avg_bits': bit_allocation.float().mean().item()
            }
            return encoded_quantized, decoded, info
        else:
            return encoded_quantized, decoded
