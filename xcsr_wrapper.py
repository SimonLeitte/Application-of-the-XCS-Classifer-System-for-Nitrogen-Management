# xcsr_wrapper.py
from __future__ import annotations

from typing import Dict, Any, List
import numpy as np
import pandas as pd

from xcsr_environment import WofostXcsEnv

FEATURE_ORDER = [
    "DVS","LAI","RFTRA","WSO","Nsurp_sofar","last_dose_kg_ha","last_app_gap_days",
    "cum_N_applied_kg_ha","IRRAD_7","TMEAN_7","RAIN_7","Ndep_sum","SM_mean",
]

def feature_names() -> list[str]:
    return FEATURE_ORDER

def feature_dim() -> int:
    return len(FEATURE_ORDER)

class XcsObsWrapper:
    def __init__(self, env: WofostXcsEnv):
        self.env = env

    def feature_names(self) -> List[str]:
        # Now just the scalar order above
        return FEATURE_ORDER[:]

    def reset(self) -> np.ndarray:
        d = self.env.reset()
        return self._flatten(d)

    def step(self, action_index: int):
        d, r, done, info = self.env.step(action_index)
        return self._flatten(d), r, done, info

    def _flatten(self, obs: Dict[str, Any]) -> np.ndarray:
        D = len(FEATURE_ORDER)
        out = np.empty(D, dtype=float)
        missing = []

        for i, k in enumerate(FEATURE_ORDER):
            if k not in obs:
                missing.append(k)
                continue

            v = obs[k]

            # Accept scalars; reject vectors (unless size==1)
            if isinstance(v, (list, tuple, np.ndarray)):
                arr = np.asarray(v)
                if arr.size != 1:
                    raise ValueError(f"Feature '{k}' must be scalar; got shape {arr.shape}.")
                v = float(arr.reshape(-1)[0])
            elif isinstance(v, (bool, np.bool_)):
                v = 1.0 if v else 0.0
            else:
                try:
                    v = float(v)
                except Exception as e:
                    raise TypeError(f"Feature '{k}' not convertible to float: {e}")

            if not np.isfinite(v):
                raise ValueError(f"Feature '{k}' must be finite; got {v}.")

            out[i] = v

        if missing:
            raise KeyError(f"Missing required features: {missing}")

        # Final sanity check: exactly D, in order.
        assert out.shape == (D,), f"Flatten produced {out.shape}, expected ({D},)"
        return out
