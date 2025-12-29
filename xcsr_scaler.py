# xcsr_scaler.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from typing import Iterable, Dict, Tuple, List

# ---- 1) Fixed per-feature ranges (edit to your crop/site if needed) ----
# These cover your FEATURE_ORDER:
# ["DVS","LAI","RFTRA","WSO","Nsurp_sofar","last_dose_kg_ha","last_app_gap_days",
#  "cum_N_applied_kg_ha","IRRAD_7","TMEAN_7","RAIN_7","Ndep_sum","SM_mean"]
RANGES: Dict[str, Tuple[float, float]] = {
    # Phase, canopy, water-stress, yield proxy
    "DVS": (-0.10, 2.00),      # kept RAW (min-max ok too; raw is more interpretable for rules)
    "LAI": (0.0, 8.0),
    "RFTRA": (0.0, 1.0),
    "WSO": (0.0, 15000.0),

    # Running nitrogen signals
    "Nsurp_sofar": (-50.0, 150.0),        # can be <0 (soil mining) or >0 (surplus)
    "last_dose_kg_ha": (0.0, 90.0),       # matches your ACTION_SET max
    "last_app_gap_days": (0.0, 365.0),    # covers daily stride and the 999→365 sentinel collapse
    "cum_N_applied_kg_ha": (0.0, 360.0),  # typical total N across season

    # Weather aggregates (7-day)
    "IRRAD_7": (0.0, 210.0),   # MJ/m^2 over 7 days (~30 per day * 7)
    "TMEAN_7": (-10.0, 30.0),  # °C average over 7 days
    "RAIN_7": (0.0, 200.0),    # mm sum over 7 days

    # Soil mineral N & moisture
    "Ndep_sum": (0.0, 300.0),  # NO3 + NH4 pool (site-dependent)
    "SM_mean": (0.0, 1.0),     # volumetric soil moisture (0..1)
}

# ---- 2) Features to keep unnormalized (raw) ----
# Keep these raw for crisp, human-interpretable rule intervals in XCSR:
KEEP_RAW = {
    "DVS",
    "last_app_gap_days",
    "last_dose_kg_ha",
    "cum_N_applied_kg_ha",
    "Nsurp_sofar",
    "IRRAD_7",
    "TMEAN_7",
    "RAIN_7",
    "SM_mean",
    "Ndep_sum"
}

@dataclass
class MinMaxScalerSelective:
    lo: np.ndarray
    hi: np.ndarray
    keep_mask: np.ndarray
    eps: float = 1e-9

    def __post_init__(self):
        self.lo = np.asarray(self.lo, dtype=float)
        self.hi = np.asarray(self.hi, dtype=float)
        self.keep_mask = np.asarray(self.keep_mask, dtype=bool)
        if not (self.lo.shape == self.hi.shape == self.keep_mask.shape):
            raise ValueError("lo/hi/keep_mask shapes must match")

    # ---- batch API ----
    def fwd(self, X: np.ndarray) -> np.ndarray:
        """Raw -> normalized; preserves kept-raw features."""
        X = np.asarray(X, dtype=float).copy()
        idx = ~self.keep_mask
        denom = np.maximum(self.hi[idx] - self.lo[idx], self.eps)
        X[..., idx] = (X[..., idx] - self.lo[idx]) / denom
        return X

    def inv(self, Z: np.ndarray) -> np.ndarray:
        """Normalized -> raw; preserves kept-raw features."""
        Z = np.asarray(Z, dtype=float).copy()
        idx = ~self.keep_mask
        Z[..., idx] = Z[..., idx] * (self.hi[idx] - self.lo[idx]) + self.lo[idx]
        return Z

    # ---- single-vector helpers ----
    def fwd_vec(self, x: Iterable[float]) -> np.ndarray:
        x = np.asarray(list(x), dtype=float)
        return self.fwd(x)

    def inv_vec(self, z: Iterable[float]) -> np.ndarray:
        z = np.asarray(list(z), dtype=float)
        return self.inv(z)

    # --- quality-of-life for logging / single-dim denorm ---
    def denorm_i(self, i: int, z: float) -> float:
        """Denormalize a single dimension i. If keep_raw, returns z unchanged."""
        i = int(i)
        if self.keep_mask[i]:
            return float(z)
        lo, hi = float(self.lo[i]), float(self.hi[i])
        return float(z) * (hi - lo) + lo

    # ---- persistence ----
    def to_json(self, path: str | Path, feature_order: List[str]) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "feature_order": list(feature_order),
                    "lo": self.lo.tolist(),
                    "hi": self.hi.tolist(),
                    "keep_mask": self.keep_mask.astype(int).tolist(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @staticmethod
    def from_json(path: str | Path) -> tuple["MinMaxScalerSelective", List[str]]:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        lo = np.array(d["lo"], dtype=float)
        hi = np.array(d["hi"], dtype=float)
        keep_mask = np.array(d["keep_mask"], dtype=int).astype(bool)
        feats = list(d.get("feature_order", []))
        return MinMaxScalerSelective(lo, hi, keep_mask), feats


# ---- 3) Builder for your wrapper's feature order ----
def make_scaler_for_features(feature_order: List[str]) -> MinMaxScalerSelective:
    """Create a selective scaler aligned to `feature_order` using RANGES and KEEP_RAW."""
    lo = []
    hi = []
    keep = []
    missing = []
    for f in feature_order:
        if f not in RANGES:
            missing.append(f)
            # safe fallback if missing: symmetric [-1,1] range
            lo.append(-1.0)
            hi.append(1.0)
        else:
            lo_v, hi_v = RANGES[f]
            lo.append(lo_v)
            hi.append(hi_v)
        keep.append(f in KEEP_RAW)
    if missing:
        print(f"[xcsr_scaler] Warning: features missing in RANGES: {missing} (using fallback [-1,1])")
    return MinMaxScalerSelective(np.array(lo), np.array(hi), np.array(keep, dtype=bool))


# ---- 4) Pretty summary ----
def scaler_summary(scaler: MinMaxScalerSelective, feature_order: List[str]) -> str:
    lines = ["feature, lo, hi, keep_raw"]
    for f, lo, hi, k in zip(feature_order, scaler.lo, scaler.hi, scaler.keep_mask):
        lines.append(f"{f}, {lo:.6g}, {hi:.6g}, {bool(k)}")
    return "\n".join(lines)
