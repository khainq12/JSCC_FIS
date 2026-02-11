"""
FIS Modules for Deep JSCC
STABLE VERSION (NaN-safe, overflow-safe)
"""

import torch
import torch.nn as nn
import numpy as np


# ============================================================
# FIS Layer 1: Importance Assessment
# ============================================================

class FIS_ImportanceAssessment(nn.Module):

    def __init__(self):
        super().__init__()

        try:
            import skfuzzy as fuzz
            from skfuzzy import control as ctrl
            self.use_fuzzy = True
            self._setup_fuzzy_system()
            print("FIS Layer 1: Using fuzzy inference system")
        except ImportError:
            print("Warning: scikit-fuzzy not installed. Using NN approximation.")
            self.use_fuzzy = False
            self._setup_nn_approximation()

    def _setup_fuzzy_system(self):
        import skfuzzy as fuzz
        from skfuzzy import control as ctrl

        self.magnitude = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'magnitude')
        self.variance = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'variance')
        self.gradient = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'gradient')
        self.importance = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'importance')

        for var in [self.magnitude, self.variance, self.gradient]:
            var['low'] = fuzz.trimf(var.universe, [0, 0, 0.4])
            var['medium'] = fuzz.trimf(var.universe, [0.3, 0.5, 0.7])
            var['high'] = fuzz.trimf(var.universe, [0.6, 1, 1])

        self.importance['very_low'] = fuzz.trimf(self.importance.universe, [0, 0, 0.25])
        self.importance['low'] = fuzz.trimf(self.importance.universe, [0.15, 0.35, 0.5])
        self.importance['medium'] = fuzz.trimf(self.importance.universe, [0.4, 0.5, 0.6])
        self.importance['high'] = fuzz.trimf(self.importance.universe, [0.5, 0.7, 0.85])
        self.importance['very_high'] = fuzz.trimf(self.importance.universe, [0.75, 1, 1])

        self.rules = [
            ctrl.Rule(self.magnitude['high'] & self.variance['high'],
                      self.importance['very_high']),
            ctrl.Rule(self.gradient['high'],
                      self.importance['high']),
            ctrl.Rule(self.magnitude['low'] & self.variance['low'],
                      self.importance['very_low']),
            ctrl.Rule(self.magnitude['medium'] & self.variance['medium'],
                      self.importance['medium']),
        ]

        self.ctrl_system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.ctrl_system)

    def _setup_nn_approximation(self):
        self.nn_approx = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, features):

        B, C, H, W = features.shape
        importance_map = torch.zeros(B, H, W, device=features.device)

        if self.use_fuzzy:
            features_np = features.detach().cpu().numpy()

            for b in range(B):
                for i in range(H):
                    for j in range(W):

                        f_ij = features_np[b, :, i, j]

                        mag = np.linalg.norm(f_ij) / np.sqrt(C)
                        var = np.var(f_ij)
                        grad = np.std(f_ij)

                        mag = np.clip(mag, 0, 1)
                        var = np.clip(var, 0, 1)
                        grad = np.clip(grad, 0, 1)

                        try:
                            self.simulation.input['magnitude'] = float(mag)
                            self.simulation.input['variance'] = float(var)
                            self.simulation.input['gradient'] = float(grad)
                            self.simulation.compute()
                            importance = self.simulation.output['importance']
                        except:
                            importance = 0.4 * mag + 0.3 * var + 0.3 * grad

                        importance_map[b, i, j] = importance
        else:
            for b in range(B):
                for i in range(H):
                    for j in range(W):

                        f_ij = features[b, :, i, j]

                        mag = torch.norm(f_ij) / np.sqrt(C)
                        var = torch.var(f_ij)
                        grad = torch.std(f_ij)

                        mag = torch.clamp(mag, 0, 1)
                        var = torch.clamp(var, 0, 1)
                        grad = torch.clamp(grad, 0, 1)

                        input_vec = torch.stack([mag, var, grad]).unsqueeze(0)
                        importance = self.nn_approx(input_vec).squeeze()

                        importance_map[b, i, j] = importance

        # 🔥 SAFETY
        importance_map = torch.clamp(importance_map, 0.0, 1.0)
        importance_map = torch.nan_to_num(importance_map, nan=0.5)

        return importance_map


# ============================================================
# FIS Layer 2: Bit Allocation
# ============================================================

class FIS_BitAllocation(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, importance_map, SNR_dB, target_rate=0.5):

        # Safe linear mapping
        bit_allocation = 1 + importance_map * 7  # Range [1, 8]

        bit_allocation = torch.clamp(bit_allocation, 1.0, 8.0)
        bit_allocation = torch.nan_to_num(bit_allocation, nan=4.0)

        return bit_allocation.long()


# ============================================================
# Adaptive Quantizer (STABLE)
# ============================================================

class AdaptiveQuantizer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, features, bit_allocation):

        B, C, H, W = features.shape
        features_quantized = features.clone()

        for b in range(B):
            for i in range(H):
                for j in range(W):

                    bits = int(bit_allocation[b, i, j].item())

                    # 🔥 HARD CLAMP
                    bits = max(1, min(bits, 8))

                    f_ij = features[b, :, i, j]

                    levels = 2 ** bits

                    f_min = f_ij.min()
                    f_max = f_ij.max()

                    if (f_max - f_min).abs() > 1e-8:

                        f_norm = (f_ij - f_min) / (f_max - f_min)
                        f_norm = torch.clamp(f_norm, 0.0, 1.0)

                        f_quant = torch.round(f_norm * (levels - 1)) / (levels - 1)
                        f_dequant = f_quant * (f_max - f_min) + f_min

                        features_quantized[b, :, i, j] = f_dequant

        features_quantized = torch.nan_to_num(
            features_quantized,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        )

        return features_quantized
