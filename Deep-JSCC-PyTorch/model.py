# -*- coding: utf-8 -*-
"""
Compatibility wrapper.

The original repository contained two different model implementations:
- `model_baseline.py`: Deep-JSCC-PyTorch-style baseline (recommended for the paper)
- `model.py`: an alternative CNN autoencoder variant

To keep the project runnable and avoid import errors, this file re-exports the
baseline models and the new FIS-enhanced model that uses the SAME encoder/decoder
for fair comparison.

Use:
    from model import DeepJSCC, DeepJSCC_FIS, ratio2filtersize

Legacy alias:
    JSCC_FIS  -> DeepJSCC_FIS
"""

from __future__ import annotations

from model_baseline import DeepJSCC, DeepJSCC_FIS, ratio2filtersize

# Legacy name used by older scripts
JSCC_FIS = DeepJSCC_FIS
