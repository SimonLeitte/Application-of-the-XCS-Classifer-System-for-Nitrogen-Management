# xcsr_environment.py
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from datetime import date
from typing import Dict, Any, Optional, Tuple, Deque, Iterable, List
from collections import deque

import yaml
import numpy as np
import pandas as pd

import pcse
from pcse import signals
from pcse.engine import Engine as PcseEngine
from pcse.base import ParameterProvider
from pcse.input import (
    YAMLCropDataProvider,
    CSVWeatherDataProvider,
)

# --------------------------
# Paths configuration
# --------------------------
@dataclass
class EnvPaths:
    base_dir: Path
    data_dir: Path
    input_dir: Path
    output_dir: Path
    crop_dir: Path
    soil_yaml: Path
    site_yaml: Path
    conf_path: Path
    weather_csv: Path

def default_paths(base: Optional[Path] = None) -> EnvPaths:
    base_dir = (base or Path(__file__).resolve().parent)
    data_dir = base_dir / "data" / "wofost81"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return EnvPaths(
        base_dir=base_dir,
        data_dir=data_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        crop_dir=input_dir / "crop",
        soil_yaml=input_dir / "soil" / "arminda_soil.yaml",
        site_yaml=input_dir / "site" / "arminda_site.yaml",
        conf_path=base_dir / "Wofost81_NWLP_MLWB_SNOMIN.conf",
        weather_csv=input_dir / "weather" / "random_weather_csv.csv",
    )

# --------------------------
# Utilities
# --------------------------
def _safe_getattr(obj, name: str, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default

def _is_prim(x):
    return isinstance(x, (int, float, str, bool, bytes, bytearray, type(None)))

def _find_snomin_component(model_root) -> Any:
    """Find the SNOMIN component (states expose NO3/NH4 arrays)."""
    wanted_any = ("NO3", "NH4")
    seen = set()
    q: Deque[Tuple[object, int]] = deque([(model_root, 0)])
    while q:
        obj, depth = q.popleft()
        if id(obj) in seen or depth > 8:
            continue
        seen.add(id(obj))

        st = _safe_getattr(obj, "states")
        if st is not None and any(hasattr(st, n) for n in wanted_any):
            return obj

        if obj.__class__.__name__ == "VariableKiosk":
            continue
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            for v in d.values():
                if not _is_prim(v):
                    q.append((v, depth + 1))
        if isinstance(obj, (list, tuple, set)):
            for v in obj:
                if not _is_prim(v):
                    q.append((v, depth + 1))
        elif isinstance(obj, dict):
            for v in obj.values():
                if not _is_prim(v):
                    q.append((v, depth + 1))
    return None

def _safe_sum(x):
    if x is None:
        return float("nan")
    a = np.asarray(x, dtype=float).ravel()
    return float(np.nansum(a)) if a.size else float("nan")

def _safe_mean(x):
    if x is None:
        return float("nan")
    a = np.asarray(x, dtype=float).ravel()
    return float(np.nanmean(a)) if a.size else float("nan")

def _isfinite(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def build_agro_for_year(year: int) -> list:
    """
    Minimal AgroManagement in-memory:
      - Sowing Oct 1 (Y-1)
      - Harvest Sep 1 (Y)
    """
    sow = date(year - 1, 10, 1)
    har = date(year, 9, 1)
    campaign = sow
    return [{
        campaign: {
            "CropCalendar": {
                "crop_name": "winterwheat",
                "variety_name": "Arminda",
                "crop_start_date": sow,
                "crop_start_type": "sowing",
                "crop_end_date": har,
                "crop_end_type": "harvest",
                "max_duration": 400,
            },
            "TimedEvents": None,
            "StateEvents": None,
        }
    }]

# --------------------------
# Engine with nitrogen injection
# --------------------------
class EngineWithN(PcseEngine):
    """
    Akzeptiert eine 'action' (kg N/ha) und injiziert sie via SNOMIN
    (50/50 NH4/NO3), danach normale PCSE-Taktung.
    """

    def _run(self, action_kgN_ha: float = 0.0):
        self.day, delt = self.timer()
        self.integrate(self.day, delt)
        self.drv = self._get_driving_variables(self.day)
        self.agromanager(self.day, self.drv)

        if action_kgN_ha and action_kgN_ha > 0.0:
            # SNOMIN: mineral N split 50/50
            self._send_signal(
                signal=signals.apply_n_snomin,
                amount=float(action_kgN_ha),
                application_depth=10.0,
                cnratio=0.0,
                f_NH4N=0.5,
                f_NO3N=0.5,
                f_orgmat=0.0,
                initial_age=0.0
            )
            # (Kompatibilität) generischer apply_n
            self._send_signal(
                signal=signals.apply_n,
                amount=float(action_kgN_ha),
                recovery=0.7,
                N_amount=float(action_kgN_ha),
                N_recovery=0.7,
            )

        self.calc_rates(self.day, self.drv)
        if self.flag_terminate is True:
            self._terminate_simulation(self.day)

    def run(self, days: int = 1, action: float = 0.0):
        days_left = int(days)
        while (days_left > 0) and (self.flag_terminate is False):
            days_left -= 1
            do_action = (days_left == 0)
            self._run(action if do_action else 0.0)

# --------------------------
# The environment class
# --------------------------
class WofostXcsEnv:
    """
    Beobachtung (neu, XCSR-ready):
      DVS, LAI, RFTRA, WSO,
      Nsurp_sofar,
      last_dose_kg_ha, last_app_gap_days, cum_N_applied_kg_ha,
      IRRAD_7, TMEAN_7, RAIN_7,
      Ndep_sum,
      SM_mean

    Action space (index → kg N/ha): [0, 10, 20, 30, 50, 60, 70, 80, 90]
    Max 4 Anwendungen/Episode.
    """

    ACTION_SET = [0, 10, 20, 30, 40, 50, 60, 70, 80]

    def __init__(self, paths, mode="train", layer_count=7,
                 seed: int | None = None,
                 fixed_year: int | None = None,
                 train_years: list[int] | None = None,
                 stride_days: int = 7):
        self.paths = paths
        self.mode = mode
        self.layer_count = int(layer_count)

        # deterministic sampling / schedules
        import numpy as _np
        self.rng = _np.random.default_rng(seed if seed is not None else 0)
        self.fixed_year = int(fixed_year) if fixed_year is not None else None
        self.train_years = list(train_years) if train_years else []
        self._year_ptr = 0  # pointer into train_years

        # Episode control
        self.max_actions = 4
        self.stride = int(stride_days)

        # PCSE
        self.model: Optional[EngineWithN] = None
        self.snomin = None
        self.season_start: Optional[date] = None

        # Episode trackers
        self.n_actions: int = 0
        self.n_applied_kg: float = 0.0

        # DVS window
        self.dvs_min_allowed: float = 0.2
        self.dvs_max_allowed: float = 1.4

        # Derived-feature trackers
        self.cum_N_applied_kg_ha: float = 0.0
        self.last_app_gap_days: int = 999
        self.last_dose_kg_ha: float = 0.0

        # Fallback rolling windows for *_7
        from collections import deque
        self._irr_q: Deque[float] = deque(maxlen=7)
        self._t_q: Deque[float]   = deque(maxlen=7)
        self._r_q: Deque[float]   = deque(maxlen=7)

        # Static inputs
        self.crop_dp = YAMLCropDataProvider(fpath=self.paths.crop_dir)
        with open(self.paths.soil_yaml, "r", encoding="utf-8") as f:
            self.soil_dict = yaml.safe_load(f)
        with open(self.paths.site_yaml, "r", encoding="utf-8") as f:
            self.site_dict = yaml.safe_load(f)
        self.weather_dp = CSVWeatherDataProvider(str(self.paths.weather_csv))

    def _choose_episode_year(self) -> int:
        # 1) hard-fixed year?
        if getattr(self, "fixed_year", None) is not None:
            return int(self.fixed_year)

        # 2) deterministic cycle over a set of years?
        if getattr(self, "train_years", None):
            y = int(self.train_years[self._year_ptr % len(self.train_years)])
            self._year_ptr += 1
            return y

        # 3) fallback: seeded randomness (by mode)
        if self.mode == "test":
            return int(self.rng.integers(1983, 2023))  # 1983..2022
        return int(self.rng.integers(3002, 6000))  # 3002..3006 (upper bound exclusive)

    # ---- episode control ----
    def reset(self) -> Dict[str, Any]:
        self.n_actions = 0
        self.n_applied_kg = 0.0

        # Tracker zurücksetzen
        self.cum_N_applied_kg_ha = 0.0
        self.last_app_gap_days = 999
        self.last_dose_kg_ha = 0.0
        self._irr_q.clear(); self._t_q.clear(); self._r_q.clear()

        year = self._choose_episode_year()
        self.episode_year = int(year)

        agro_struct = build_agro_for_year(year)

        params = ParameterProvider(sitedata=self.site_dict, soildata=self.soil_dict, cropdata=self.crop_dp)
        self.model = EngineWithN(params, self.weather_dp, agro_struct, config=self.paths.conf_path)

        self.snomin = _find_snomin_component(self.model)
        if self.snomin is None:
            raise RuntimeError("SNOMIN component not found; cannot access NO3/NH4 soil pools.")

        campaign_key = list(agro_struct[0].keys())[0]
        self.season_start = agro_struct[0][campaign_key]["CropCalendar"]["crop_start_date"]

        return self._observe()

    def step(self, action_index: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Call reset() before step().")
        if not (0 <= action_index < len(self.ACTION_SET)):
            raise ValueError(f"Action index {action_index} out of range 0..{len(self.ACTION_SET)-1}.")

        kgN = float(self.ACTION_SET[action_index])

        # --- HARD CONSTRAINTS: DVS window + max applications ---
        cur = self._observe()
        dvs = float(cur.get("DVS") or 0.0)
        blocked_reasons = []
        allowed_dvs = (self.dvs_min_allowed <= dvs <= self.dvs_max_allowed)
        if not allowed_dvs:
            blocked_reasons.append("dvs")
        allowed_max = (self.n_actions < self.max_actions)
        if not allowed_max:
            blocked_reasons.append("max_apps")

        effective = 0.0
        blocked = False
        if kgN > 0.0:
            if allowed_dvs and allowed_max:
                effective = kgN
                self.n_actions += 1
                self.n_applied_kg += effective
            else:
                blocked = True

        # Model um 'stride' Tage vorwärts; Action am letzten Tag injizieren
        self.model.run(days=self.stride, action=effective)

        # --- Tracker updaten (neu)
        self.last_dose_kg_ha = float(effective)
        if self.last_dose_kg_ha > 0.0:
            self.cum_N_applied_kg_ha += self.last_dose_kg_ha
            self.last_app_gap_days = 0
        else:
            self.last_app_gap_days = min(365, self.last_app_gap_days + self.stride)

        obs = self._observe()
        done = bool(self.model.flag_terminate)
        info = {
            "date": self.model.day,
            "effective_action": effective,
            "n_actions": self.n_actions,
            "blocked": blocked,
            "blocked_reasons": blocked_reasons,
            "DVS_at_action": dvs,
            # Tracker für Logs
            "last_dose_kg_ha": float(self.last_dose_kg_ha),
            "cum_N_applied_kg_ha": float(self.cum_N_applied_kg_ha),
            "last_app_gap_days": int(self.last_app_gap_days),
            "episode_year": int(getattr(self, "episode_year", -1)),
        }
        if done:
            metrics = self.episode_metrics()
            info.update({
                "NUE_EUNEP": metrics.get("NUE_EUNEP"),
                "NUE_FERT": metrics.get("NUE_FERT"),
                "Nsurp": metrics.get("Nsurp"),
                "Nsurp_FERT": metrics.get("Nsurp_FERT"),
                "Yield": metrics.get("Yield"),
                "N_applied": metrics.get("N_applied"),
                "N_uptake": metrics.get("N_uptake"),
                "N_apps": metrics.get("N_apps"),
            })

        reward = 0.0  # reale Rewards berechnet der Trainer via xcsr_rewards.py
        return obs, reward, done, info

    # ---- observation builder (NEUES Feature-Set) ----
    def _observe(self) -> Dict[str, Any]:
        out = self.model.get_output()
        last = out[-1] if out else {}
        today = self.model.day

        # Wetter: bevorzugt *_7 direkt aus CSV; Fallback via Rolling
        try:
            w = self.weather_dp(today)
        except Exception:
            w = None

        def _wattr(name, default=None):
            try:
                return getattr(w, name)
            except Exception:
                return default

        irrad_7 = _wattr("IRRAD_7", None)
        tmean_7 = _wattr("TMEAN_7", None)
        rain_7  = _wattr("RAIN_7",  None)

        if irrad_7 is None or tmean_7 is None or rain_7 is None:
            irr = float(_wattr("IRRAD", 0.0) or 0.0)
            self._irr_q.append(irr)
            irrad_7 = float(np.sum(self._irr_q))

            if _wattr("TMEAN", None) is not None:
                t_now = float(_wattr("TMEAN", 0.0) or 0.0)
            else:
                tmin = float(_wattr("TMIN", 0.0) or 0.0)
                tmax = float(_wattr("TMAX", tmin) or tmin)
                t_now = 0.5 * (tmin + tmax)
            self._t_q.append(t_now)
            tmean_7 = float(np.mean(self._t_q))

            r = float(_wattr("RAIN", 0.0) or 0.0)
            self._r_q.append(r)
            rain_7 = float(np.sum(self._r_q))
        else:
            irrad_7 = float(irrad_7)
            tmean_7 = float(tmean_7)
            rain_7  = float(rain_7)

        # Boden/Bestand für abgeleitete N-Features
        st = self.snomin.states if self.snomin is not None else None
        NO3_sum = _safe_sum(_safe_getattr(st, "NO3", None))
        NH4_sum = _safe_sum(_safe_getattr(st, "NH4", None))
        SM_mean = _safe_mean(last.get("SM", None))

        Ndep_sum = float(NO3_sum + NH4_sum)  # Boden-Mineral-N (NO3+NH4)
        NuptakeTotal = float(last.get("NuptakeTotal", 0.0) or 0.0)
        NLOSSCUM     = float(last.get("NLOSSCUM", 0.0) or 0.0)
        Nsurp_sofar  = float(self.cum_N_applied_kg_ha - NuptakeTotal - NLOSSCUM)

        obs = {
            "DVS":        float(last.get("DVS", 0.0) or 0.0),
            "LAI":        float(last.get("LAI", 0.0) or 0.0),
            "RFTRA":      float(last.get("RFTRA", 0.0) or 0.0),
            "WSO":        float(last.get("WSO", 0.0) or 0.0),
            "Nsurp_sofar":           Nsurp_sofar,
            "last_dose_kg_ha":       float(self.last_dose_kg_ha),
            "last_app_gap_days":     float(self.last_app_gap_days),
            "cum_N_applied_kg_ha":   float(self.cum_N_applied_kg_ha),
            "IRRAD_7":    irrad_7,
            "TMEAN_7":    tmean_7,
            "RAIN_7":     rain_7,
            "Ndep_sum":   float(Ndep_sum),
            "SM_mean":    float(SM_mean),
        }

        # Sanity: alles finite
        for k, v in obs.items():
            if not _isfinite(v):
                raise ValueError(f"Non-finite value in obs[{k}] = {v!r}")

        return obs

    # ---- Episode-Metriken (Deposition optional) ----
    def episode_metrics(self) -> Dict[str, Any]:
        """
        Harvest-Metriken (robust, deposition optional):
        - NUE_EUNEP (Seed + Depo + Fert) wenn Depo-Spalten vorhanden, sonst Seed+Fert
        - NUE_FERT, Nsurp, Nsurp_FERT, Yield, N_applied, N_uptake, N_apps
        """
        df = pd.DataFrame(self.model.get_output())
        eps = 1e-9

        # Outputs (harvest)
        yield_kg_ha = float(df["WSO"].iloc[-1]) if "WSO" in df.columns else 0.0
        n_out = float(df["NamountSO"].iloc[-1]) if "NamountSO" in df.columns else 0.0
        n_uptake = float(df["NuptakeTotal"].iloc[-1]) if "NuptakeTotal" in df.columns else 0.0

        # Inputs
        n_applied = float(self.cum_N_applied_kg_ha)
        N_SEED = 3.5  # fixed (winter wheat)
        # deposition optional
        if "RNO3DEPOSTT" in df.columns and "RNH4DEPOSTT" in df.columns:
            n_depo = float(df["RNO3DEPOSTT"].iloc[-1]) + float(df["RNH4DEPOSTT"].iloc[-1])
        else:
            n_depo = 0.0

        n_in_eunep = N_SEED + n_depo + n_applied
        n_in_fert = n_applied

        nue_eunep = (n_out / (n_in_eunep + eps)) if n_in_eunep > 0 else float("nan")
        nue_fert = (n_out / (n_in_fert + eps)) if n_in_fert > 0 else float("nan")

        nsurp_eunep = n_in_eunep - n_out
        nsurp_fert = n_in_fert - n_out

        # Decomposition (optional): NUE = RE * IE
        re_eunep = (n_uptake / (n_in_eunep + eps)) if n_in_eunep > 0 else float("nan")
        ie = (n_out / (n_uptake + eps)) if n_uptake > 0 else float("nan")

        return {
            "Yield": yield_kg_ha,
            "N_applied": n_applied,
            "N_uptake": n_uptake,
            "NUE_EUNEP": nue_eunep,
            "NUE_FERT": nue_fert,
            "Nsurp": nsurp_eunep,
            "Nsurp_FERT": nsurp_fert,
            "NUE_RE": re_eunep,
            "NUE_IE": ie,
            "N_apps": int(self.n_actions),
        }
