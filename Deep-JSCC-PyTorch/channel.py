# -*- coding: utf-8 -*-
"""
Channel models used by Deep-JSCC.

Fixes vs. previous version
- Correct AWGN noise variance (no accidental /2).
- Rayleigh fading applied consistently to I/Q pairs (complex baseband).
- Support both (C,H,W) and (B,C,H,W) inputs, and preserve original shape.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Channel(nn.Module):
    def __init__(self, channel_type: str = 'AWGN', snr: float = 20.0):
        """
        Args:
            channel_type: 'AWGN' or 'Rayleigh'
            snr: SNR in dB
        """
        super().__init__()
        self.channel_type = channel_type
        self.snr = float(snr)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: latent tensor with shape (B,C,H,W) or (C,H,W).
               For Rayleigh, we assume I/Q representation across channels:
               first C/2 channels = I, last C/2 channels = Q.

        Returns:
            z_tilde: same shape as input.
        """
        squeeze_back = False
        if z.dim() == 3:
            z = z.unsqueeze(0)
            squeeze_back = True

        if z.dim() != 4:
            raise ValueError(f"Channel expects 3D or 4D tensor, got shape {tuple(z.shape)}")

        # Mean power per real dimension
        sig_pwr = z.pow(2).mean(dim=(1, 2, 3), keepdim=True)  # (B,1,1,1)

        # Convert SNR(dB) -> linear
        snr_lin = 10.0 ** (self.snr / 10.0)

        if self.channel_type.upper() == 'AWGN':
            # Noise variance per real dim: sig_pwr / snr_lin
            noise_var = sig_pwr / snr_lin
            noise = torch.randn_like(z) * torch.sqrt(noise_var + 1e-12)
            out = z + noise

        elif self.channel_type.upper() == 'RAYLEIGH':
            B, C, H, W = z.shape

            # Generate complex fading coefficient h = h_r + j h_i, E[|h|^2]=1
            h_r = torch.randn(B, 1, 1, 1, device=z.device, dtype=z.dtype) / (2.0 ** 0.5)
            h_i = torch.randn(B, 1, 1, 1, device=z.device, dtype=z.dtype) / (2.0 ** 0.5)

            if C % 2 == 0:
                z_I, z_Q = torch.chunk(z, chunks=2, dim=1)
                # (h_r + j h_i) * (z_I + j z_Q)
                y_I = h_r * z_I - h_i * z_Q
                y_Q = h_i * z_I + h_r * z_Q
                z_faded = torch.cat([y_I, y_Q], dim=1)
            else:
                # Fallback: real-valued fading if channels are not paired
                z_faded = h_r * z

            # AWGN after fading
            sig_pwr_faded = z_faded.pow(2).mean(dim=(1, 2, 3), keepdim=True)
            noise_var = sig_pwr_faded / snr_lin
            noise = torch.randn_like(z_faded) * torch.sqrt(noise_var + 1e-12)
            out = z_faded + noise

        else:
            raise ValueError(f"Unknown channel type: {self.channel_type}")

        if squeeze_back:
            out = out.squeeze(0)
        return out

    def change_channel(self, channel_type: str = 'AWGN', snr: float | None = None):
        self.channel_type = channel_type
        if snr is not None:
            self.snr = float(snr)

    def get_channel(self):
        return self.channel_type, self.snr
