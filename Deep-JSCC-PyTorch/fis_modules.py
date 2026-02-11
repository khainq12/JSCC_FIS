"""
FIS Modules for Deep JSCC
File: fis_modules.py (NEW FILE - add to root directory)
"""

import torch
import torch.nn as nn
import numpy as np


class FIS_ImportanceAssessment(nn.Module):
    """
    FIS Layer 1: Assess importance of each spatial location
    
    Input: Feature maps (B, C, H, W)
    Output: Importance map (B, H, W) with values in [0, 1]
    """
    def __init__(self):
        super().__init__()
        
        # Try to use scikit-fuzzy, fallback to neural network
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
        """Setup fuzzy inference system"""
        import skfuzzy as fuzz
        from skfuzzy import control as ctrl
        
        # Input variables
        self.magnitude = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'magnitude')
        self.variance = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'variance')
        self.gradient = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'gradient')
        
        # Output variable
        self.importance = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'importance')
        
        # Membership functions
        for var in [self.magnitude, self.variance, self.gradient]:
            var['low'] = fuzz.trimf(var.universe, [0, 0, 0.4])
            var['medium'] = fuzz.trimf(var.universe, [0.3, 0.5, 0.7])
            var['high'] = fuzz.trimf(var.universe, [0.6, 1, 1])
        
        self.importance['very_low'] = fuzz.trimf(self.importance.universe, [0, 0, 0.25])
        self.importance['low'] = fuzz.trimf(self.importance.universe, [0.15, 0.35, 0.5])
        self.importance['medium'] = fuzz.trimf(self.importance.universe, [0.4, 0.5, 0.6])
        self.importance['high'] = fuzz.trimf(self.importance.universe, [0.5, 0.7, 0.85])
        self.importance['very_high'] = fuzz.trimf(self.importance.universe, [0.75, 1, 1])
        
        # Fuzzy rules
        self.rules = [
            ctrl.Rule(self.magnitude['high'] & self.variance['high'], 
                     self.importance['very_high']),
            ctrl.Rule(self.magnitude['high'] & self.variance['medium'], 
                     self.importance['high']),
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
        """Fallback: NN to approximate FIS"""
        self.nn_approx = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        """
        features: (B, C, H, W) from encoder
        Returns: (B, H, W) importance map
        """
        B, C, H, W = features.shape
        importance_map = torch.zeros(B, H, W, device=features.device)
        
        if self.use_fuzzy:
            # Use fuzzy inference
            features_np = features.cpu().detach().numpy()
            
            for b in range(B):
                for i in range(H):
                    for j in range(W):
                        f_ij = features_np[b, :, i, j]
                        
                        # Compute features
                        mag = np.linalg.norm(f_ij) / np.sqrt(C)
                        var = np.var(f_ij)
                        grad = np.std(f_ij)
                        
                        # Normalize
                        mag = np.clip(mag, 0, 1)
                        var = np.clip(var / (np.mean(f_ij**2) + 1e-8), 0, 1)
                        grad = np.clip(grad / (np.mean(np.abs(f_ij)) + 1e-8), 0, 1)
                        
                        # FIS inference
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
            # Use NN approximation
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
        
        return importance_map


class FIS_BitAllocation(nn.Module):
    """
    FIS Layer 2: Allocate bits based on importance and channel
    
    Input: Importance map (B, H, W), SNR, rate budget
    Output: Bit allocation (B, H, W) with values in [4, 12]
    """
    def __init__(self):
        super().__init__()
        
        try:
            import skfuzzy as fuzz
            from skfuzzy import control as ctrl
            self.use_fuzzy = True
            self._setup_fuzzy_system()
            print("FIS Layer 2: Using fuzzy inference system")
        except ImportError:
            print("Warning: Using linear bit allocation")
            self.use_fuzzy = False
    
    def _setup_fuzzy_system(self):
        """Setup fuzzy system for bit allocation"""
        import skfuzzy as fuzz
        from skfuzzy import control as ctrl
        
        # Input variables
        self.importance = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'importance')
        self.snr = ctrl.Antecedent(np.arange(0, 30.1, 0.1), 'snr')
        self.rate_budget = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'rate_budget')
        
        # Output
        self.bits = ctrl.Consequent(np.arange(4, 12.1, 1), 'bits')
        
        # Membership functions
        self.importance['low'] = fuzz.trimf(self.importance.universe, [0, 0, 0.4])
        self.importance['medium'] = fuzz.trimf(self.importance.universe, [0.3, 0.5, 0.7])
        self.importance['high'] = fuzz.trimf(self.importance.universe, [0.6, 1, 1])
        
        self.snr['low'] = fuzz.trimf(self.snr.universe, [0, 0, 10])
        self.snr['medium'] = fuzz.trimf(self.snr.universe, [5, 15, 20])
        self.snr['high'] = fuzz.trimf(self.snr.universe, [15, 30, 30])
        
        self.rate_budget['low'] = fuzz.trimf(self.rate_budget.universe, [0, 0, 0.4])
        self.rate_budget['medium'] = fuzz.trimf(self.rate_budget.universe, [0.3, 0.5, 0.7])
        self.rate_budget['high'] = fuzz.trimf(self.rate_budget.universe, [0.6, 1, 1])
        
        self.bits['low'] = fuzz.trimf(self.bits.universe, [4, 4, 6])
        self.bits['medium'] = fuzz.trimf(self.bits.universe, [6, 8, 10])
        self.bits['high'] = fuzz.trimf(self.bits.universe, [10, 12, 12])
        
        # Rules
        self.rules = [
            ctrl.Rule(self.importance['high'] & self.snr['high'] & self.rate_budget['high'],
                     self.bits['high']),
            ctrl.Rule(self.importance['high'] & self.snr['medium'],
                     self.bits['medium']),
            ctrl.Rule(self.importance['medium'], self.bits['medium']),
            ctrl.Rule(self.importance['low'], self.bits['low']),
            ctrl.Rule(self.rate_budget['low'], self.bits['low']),
        ]
        
        self.ctrl_system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.ctrl_system)
    
    def forward(self, importance_map, SNR_dB, target_rate=0.5):
        """
        importance_map: (B, H, W)
        SNR_dB: scalar
        target_rate: scalar [0, 1]
        
        Returns: (B, H, W) bit allocation
        """
        B, H, W = importance_map.shape
        bit_allocation = torch.zeros(B, H, W, dtype=torch.long, device=importance_map.device)
        
        if self.use_fuzzy:
            importance_np = importance_map.cpu().detach().numpy()
            
            for b in range(B):
                for i in range(H):
                    for j in range(W):
                        importance = float(importance_np[b, i, j])
                        
                        try:
                            self.simulation.input['importance'] = importance
                            self.simulation.input['snr'] = float(SNR_dB)
                            self.simulation.input['rate_budget'] = float(target_rate)
                            self.simulation.compute()
                            bits = int(np.round(self.simulation.output['bits']))
                        except:
                            bits = int(4 + importance * 8)
                        
                        bit_allocation[b, i, j] = bits
        else:
            # Linear allocation
            bit_allocation = (4 + importance_map * 8).long()
        
        return bit_allocation


class AdaptiveQuantizer(nn.Module):
    """
    Adaptive quantization based on bit allocation
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, features, bit_allocation):
        """
        features: (B, C, H, W)
        bit_allocation: (B, H, W)
        
        Returns: (B, C, H, W) quantized features
        """
        B, C, H, W = features.shape
        features_quantized = features.clone()
        
        for b in range(B):
            for i in range(H):
                for j in range(W):
                    bits = bit_allocation[b, i, j].item()
                    
                    # Quantize all channels at this location
                    f_ij = features[b, :, i, j]
                    
                    # Uniform quantization
                    levels = 2 ** bits
                    f_min = f_ij.min()
                    f_max = f_ij.max()
                    
                    if f_max - f_min > 1e-8:
                        f_norm = (f_ij - f_min) / (f_max - f_min)
                        f_quant = torch.round(f_norm * (levels - 1)) / (levels - 1)
                        f_dequant = f_quant * (f_max - f_min) + f_min
                        features_quantized[b, :, i, j] = f_dequant
        
        return features_quantized
