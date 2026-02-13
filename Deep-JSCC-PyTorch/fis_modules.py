# -*- coding: utf-8 -*-
"""
Fuzzy modules for FIS-enhanced Deep-JSCC.

This version focuses on the intended role of FIS in this project:
- FIS does NOT replace the Deep-JSCC encoder/decoder.
- FIS acts as an interpretable controller that produces an allocation map A(i,j)
  (mask/power) over spatial positions of the latent feature map.

Two lightweight (but convincing) rule-bases are implemented:

(1) Importance assessment (6 rules): from latent features -> importance I in [0,1]
(2) Power/mask allocation (7 rules): from (I, SNR, rate_budget) -> A > 0

Implementation choices
- Vectorized torch implementation (fast, GPU-friendly), no scikit-fuzzy dependency.
- Sugeno-style inference (weighted average of singleton consequents) for efficiency,
  while keeping rules fully interpretable.

Notation
- Input latent features: F in R^{B x C x H x W}
- Importance map:       I in [0,1]^{B x H x W}
- Allocation map:       A in (0, +inf)^{B x H x W}

Typical usage
    imp = FIS_ImportanceAssessment()
    alloc = FIS_PowerMask()
    I, info_I = imp(F)
    A, info_A = alloc(I, snr_db=10.0, rate_budget=0.7)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn


# -----------------------------
# Helpers: membership functions
# -----------------------------
def _mf_low(x: torch.Tensor) -> torch.Tensor:
    """Low on [0,1]: 1 at 0, 0 at 0.5."""
    return torch.clamp((0.5 - x) / 0.5, 0.0, 1.0)


def _mf_high(x: torch.Tensor) -> torch.Tensor:
    """High on [0,1]: 0 at 0.5, 1 at 1."""
    return torch.clamp((x - 0.5) / 0.5, 0.0, 1.0)


def _mf_med(x: torch.Tensor) -> torch.Tensor:
    """Medium on [0,1]: triangle with peak at 0.5, support [0.25,0.75]."""
    return torch.clamp(1.0 - torch.abs(x - 0.5) / 0.25, 0.0, 1.0)


def _safe_norm01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize each sample map to [0,1] using per-sample max."""
    # x: (B,H,W)
    mx = x.amax(dim=(1, 2), keepdim=True)
    return x / (mx + eps)


def _spatial_grad_energy(F: torch.Tensor) -> torch.Tensor:
    """
    Simple edge/structure proxy from latent features:
    average absolute spatial gradients over channels.
    F: (B,C,H,W) -> g: (B,H,W)
    """
    dx = torch.abs(F[:, :, :, 1:] - F[:, :, :, :-1])  # (B,C,H,W-1)
    dy = torch.abs(F[:, :, 1:, :] - F[:, :, :-1, :])  # (B,C,H-1,W)
    # pad to (H,W)
    dx = torch.nn.functional.pad(dx, (0, 1, 0, 0))
    dy = torch.nn.functional.pad(dy, (0, 0, 0, 1))
    g = (dx + dy).mean(dim=1)  # (B,H,W)
    return g


# -----------------------------
# FIS 1: Importance assessment
# -----------------------------
class FIS_ImportanceAssessment(nn.Module):
    """
    Importance assessment with a compact, interpretable rule-base (6 rules).

    Inputs (normalized to [0,1] per image):
    - m: feature magnitude map
    - v: feature variance map (channel-wise)
    - g: gradient/edge energy map

    Output:
    - I: importance in [0,1]
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

        # Singleton consequents for Sugeno inference (interpretable levels)
        self.I_low = 0.20
        self.I_med = 0.55
        self.I_high = 0.85

    def forward(self, F: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            F: latent feature map, (B,C,H,W)

        Returns:
            I: importance map, (B,H,W)
            info: diagnostics (rule activation means)
        """
        if F.dim() != 4:
            raise ValueError(f"Expected F with shape (B,C,H,W), got {tuple(F.shape)}")

        B, C, H, W = F.shape

        # --- Features ---
        # Magnitude (L2 over channels)
        m = torch.sqrt((F ** 2).sum(dim=1) + self.eps) / (C ** 0.5)  # (B,H,W)
        m = _safe_norm01(m, self.eps)

        # Channel variance proxy
        v = F.var(dim=1, unbiased=False)  # (B,H,W)
        v = _safe_norm01(v, self.eps)

        # Edge/structure proxy
        g = _spatial_grad_energy(F)  # (B,H,W)
        g = _safe_norm01(g, self.eps)

        # --- Memberships ---
        mL, mM, mH = _mf_low(m), _mf_med(m), _mf_high(m)
        vL, vM, vH = _mf_low(v), _mf_med(v), _mf_high(v)
        gL, gM, gH = _mf_low(g), _mf_med(g), _mf_high(g)

        # --- Rule base (6 rules) ---
        # R1: IF g is High          THEN I is High
        w1 = gH

        # R2: IF m is High AND v is High THEN I is High
        w2 = torch.minimum(mH, vH)

        # R3: IF m is High AND v is Low  THEN I is Medium
        w3 = torch.minimum(mH, vL)

        # R4: IF m is Low  AND v is High THEN I is Medium
        w4 = torch.minimum(mL, vH)

        # R5: IF m is Med  AND v is Med  THEN I is Medium
        w5 = torch.minimum(mM, vM)

        # R6: IF m is Low  AND v is Low  AND g is Low THEN I is Low
        w6 = torch.minimum(torch.minimum(mL, vL), gL)

        # Sugeno aggregation
        num = (w1 + w2) * self.I_high + (w3 + w4 + w5) * self.I_med + w6 * self.I_low
        den = (w1 + w2 + w3 + w4 + w5 + w6) + self.eps
        I = torch.clamp(num / den, 0.0, 1.0)

        info = {
            "feat_mag_mean": m.mean().item(),
            "feat_var_mean": v.mean().item(),
            "feat_grad_mean": g.mean().item(),
            "rule_mean": {
                "R1_g_high->I_high": w1.mean().item(),
                "R2_mH_vH->I_high": w2.mean().item(),
                "R3_mH_vL->I_med": w3.mean().item(),
                "R4_mL_vH->I_med": w4.mean().item(),
                "R5_mM_vM->I_med": w5.mean().item(),
                "R6_mL_vL_gL->I_low": w6.mean().item(),
            },
        }
        return I, info


# -----------------------------
# FIS 2: Power/mask allocation
# -----------------------------
class FIS_PowerMask(nn.Module):
    """
    Allocate an interpretable spatial power/mask map A(i,j) with 7 rules.

    Inputs:
    - I: importance map in [0,1], (B,H,W)
    - snr_db: float or tensor, scalar per batch (we support scalar)
    - rate_budget: float in [0,1] (0 = tight budget, 1 = relaxed budget)

    Output:
    - A: allocation map, positive, (B,H,W)
         later used as z_masked = z * sqrt(A)
    """

    def __init__(
        self,
        snr_min_db: float = 0.0,
        snr_max_db: float = 20.0,
        A_low: float = 0.60,
        A_med: float = 1.00,
        A_high: float = 1.40,
        A_min: float = 0.30,
        A_max: float = 3.00,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.snr_min_db = snr_min_db
        self.snr_max_db = snr_max_db
        self.A_low = A_low
        self.A_med = A_med
        self.A_high = A_high
        self.A_min = A_min
        self.A_max = A_max
        self.eps = eps

    def _snr_norm01(self, snr_db: float, device, dtype) -> torch.Tensor:
        snr = torch.tensor(float(snr_db), device=device, dtype=dtype)
        snr_u = (snr - self.snr_min_db) / (self.snr_max_db - self.snr_min_db)
        return torch.clamp(snr_u, 0.0, 1.0)

    def forward(
        self,
        I: torch.Tensor,
        snr_db: float,
        rate_budget: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            A: (B,H,W)
            info: rule activation means
        """
        if I.dim() != 3:
            raise ValueError(f"Expected I with shape (B,H,W), got {tuple(I.shape)}")

        device, dtype = I.device, I.dtype
        B, H, W = I.shape

        # --- Normalize inputs ---
        # SNR scalar -> broadcast
        snr_u = self._snr_norm01(snr_db, device, dtype)  # scalar tensor
        snr_u_map = snr_u.view(1, 1, 1).expand(B, H, W)

        # Rate budget scalar -> broadcast
        R = torch.tensor(float(rate_budget), device=device, dtype=dtype)
        R = torch.clamp(R, 0.0, 1.0).view(1, 1, 1).expand(B, H, W)

        # Memberships
        I_L, I_M, I_H = _mf_low(I), _mf_med(I), _mf_high(I)
        S_L, S_M, S_H = _mf_low(snr_u_map), _mf_med(snr_u_map), _mf_high(snr_u_map)
        R_L, R_M, R_H = _mf_low(R), _mf_med(R), _mf_high(R)

        # --- Rule base (7 rules) ---
        # R1: IF R is Low                 THEN A is Low   (tight budget)
        w1 = R_L

        # R2: IF I is Low                 THEN A is Low
        w2 = I_L

        # R3: IF I is High AND SNR is Low THEN A is High  (protect important under bad channel)
        w3 = torch.minimum(I_H, S_L)

        # R4: IF I is High AND SNR is Med THEN A is Med
        w4 = torch.minimum(I_H, S_M)

        # R5: IF I is High AND SNR is High THEN A is Med  (don't waste power)
        w5 = torch.minimum(I_H, S_H)

        # R6: IF I is Med AND SNR is Low  THEN A is Med
        w6 = torch.minimum(I_M, S_L)

        # R7: IF I is Med AND SNR is High THEN A is Low
        w7 = torch.minimum(I_M, S_H)

        # Small default weight to avoid degenerate denominator (interpretable as "baseline A=1")
        w0 = torch.full_like(I, 0.05)

        num = (
            w0 * self.A_med
            + (w1 + w2 + w7) * self.A_low
            + (w4 + w5 + w6) * self.A_med
            + w3 * self.A_high
        )
        den = (w0 + w1 + w2 + w3 + w4 + w5 + w6 + w7) + self.eps
        A = num / den

        # Normalize so average A per image is 1 (keeps total power comparable)
        A = A / (A.mean(dim=(1, 2), keepdim=True) + self.eps)

        # Clamp to avoid extreme scaling
        A = torch.clamp(A, self.A_min, self.A_max)

        info = {
            "snr_u": float(snr_u.item()),
            "rate_budget": float(rate_budget),
            "A_mean": A.mean().item(),
            "A_std": A.std().item(),
            "rule_mean": {
                "R1_R_low->A_low": w1.mean().item(),
                "R2_I_low->A_low": w2.mean().item(),
                "R3_I_high_S_low->A_high": w3.mean().item(),
                "R4_I_high_S_med->A_med": w4.mean().item(),
                "R5_I_high_S_high->A_med": w5.mean().item(),
                "R6_I_med_S_low->A_med": w6.mean().item(),
                "R7_I_med_S_high->A_low": w7.mean().item(),
            },
        }
        return A, info


# -----------------------------
# Utility: apply allocation
# -----------------------------
def apply_power_mask(z: torch.Tensor, A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Apply allocation map A to latent z by scaling amplitude: z' = z * sqrt(A).
    z: (B,C,H,W), A: (B,H,W)
    """
    if z.dim() != 4 or A.dim() != 3:
        raise ValueError("z must be (B,C,H,W) and A must be (B,H,W)")
    scale = torch.sqrt(torch.clamp(A, min=eps)).unsqueeze(1)  # (B,1,H,W)
    return z * scale


def power_normalize(z: torch.Tensor, P: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize each sample to satisfy E[||z||^2] = P*k (same style as baseline code).
    """
    if z.dim() == 3:
        z = z.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False

    if z.dim() != 4:
        raise ValueError(f"Expected 3D or 4D tensor, got {tuple(z.shape)}")

    B = z.size(0)
    k = z[0].numel()
    z_flat = z.view(B, -1)
    denom = torch.sqrt((z_flat * z_flat).sum(dim=1, keepdim=True) + eps)  # (B,1)
    z_norm = z * (torch.sqrt(torch.tensor(P * k, device=z.device, dtype=z.dtype)).view(1, 1, 1, 1) / denom.view(B, 1, 1, 1))

    if squeeze_back:
        z_norm = z_norm.squeeze(0)
    return z_norm
