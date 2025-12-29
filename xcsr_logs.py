# xcsr_logs.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Any
import csv

@dataclass
class StepRewardBreakdown:
    r_step_total: float = 0.0
    r_step_shaping: float = 0.0
    r_step_app_cost: float = 0.0
    r_step_spacing_pen: float = 0.0
    r_step_timing_pen: float = 0.0
    @classmethod
    def from_any(cls, obj: Any) -> "StepRewardBreakdown":
        if obj is None: return cls()
        if isinstance(obj, StepRewardBreakdown): return obj
        if isinstance(obj, Mapping):
            fields = cls.__dataclass_fields__.keys()
            return cls(**{k: float(obj.get(k, 0.0)) for k in fields})
        return cls(r_step_total=float(obj))

@dataclass
class TerminalRewardBreakdown:
    r_term_total: float = 0.0
    r_term_yield: float = 0.0
    r_term_nue_band: float = 0.0
    r_term_nsurp_pen: float = 0.0
    r_term_other: float = 0.0
    NUE: Optional[float] = None
    EUNEP: Optional[float] = None
    Nsurp: Optional[float] = None
    WSO: Optional[float] = None
    Napplied: Optional[float] = None
    Napps: Optional[int] = None
    phi_nue: Optional[float] = None
    phi_nsurp: Optional[float] = None
    prod: Optional[float] = None
    ycond: Optional[float] = None
    harvest_flag: Optional[bool] = None

class RewardCSVLogger:
    def __init__(self, csv_path: Path, feature_order: Iterable[str]):
        self.csv_path = Path(csv_path)
        self.feature_order: List[str] = list(feature_order)
        self.fieldnames: List[str] = self._build_header(self.feature_order)
        self._ensure_header()

    def log_step(
        self, *, run_id: str, reward_name: str, episode: int, step: int,
        exploration_mode: str, epsilon: float, action_id: int, action_effective: float,
        blocked: bool, blocked_reason: str,
        raw_obs: Mapping[str, float], scaled_obs: Mapping[str, float],
        step_breakdown: StepRewardBreakdown | Mapping[str, Any] | float | None,
        G_cumulative_after_step: float,
    ) -> None:
        br = StepRewardBreakdown.from_any(step_breakdown)
        row = {
            "run_id": run_id, "reward_name": reward_name,
            "episode": int(episode), "step": int(step), "is_terminal": False,
            "exploration_mode": str(exploration_mode), "epsilon": float(epsilon),
            "action_id": int(action_id), "action_effective": float(action_effective),
            "blocked": bool(blocked), "blocked_reason": str(blocked_reason or ""),
            **self._flatten_obs(raw_obs, prefix="RAW_"),
            **self._flatten_obs(scaled_obs, prefix="SCALED_"),
            **asdict(br),
            "G_cumulative_after_step": float(G_cumulative_after_step),
            "r_term_total": "", "r_term_yield": "", "r_term_nue_band": "", "r_term_nsurp_pen": "", "r_term_other": "",
            "NUE": "", "EUNEP": "", "Nsurp": "", "WSO": "", "Napplied": "", "Napps": "",
            "phi_nue": "", "phi_nsurp": "", "prod": "", "ycond": "", "harvest_flag": "",
        }
        self._append_row(row)

    def log_terminal(
        self, *, run_id: str, reward_name: str, episode: int, step: int,
        exploration_mode: str, epsilon: float, action_id: int, action_effective: float,
        blocked: bool, blocked_reason: str,
        raw_obs: Mapping[str, float], scaled_obs: Mapping[str, float],
        terminal_breakdown: TerminalRewardBreakdown | Mapping[str, Any] | float | None,
        G_cumulative_after_step: float,
    ) -> None:
        tbr = self._coerce_terminal_breakdown(terminal_breakdown)
        row = {
            "run_id": run_id, "reward_name": reward_name,
            "episode": int(episode), "step": int(step), "is_terminal": True,
            "exploration_mode": str(exploration_mode), "epsilon": float(epsilon),
            "action_id": int(action_id), "action_effective": float(action_effective),
            "blocked": bool(blocked), "blocked_reason": str(blocked_reason or ""),
            **self._flatten_obs(raw_obs, prefix="RAW_"),
            **self._flatten_obs(scaled_obs, prefix="SCALED_"),
            "r_step_total": "", "r_step_shaping": "", "r_step_app_cost": "", "r_step_spacing_pen": "", "r_step_timing_pen": "",
            "G_cumulative_after_step": float(G_cumulative_after_step),
            **tbr,
        }
        self._append_row(row)

    @staticmethod
    def _build_header(feature_order: Iterable[str]) -> List[str]:
        base = ["run_id","reward_name","episode","step","is_terminal","exploration_mode","epsilon","action_id","action_effective","blocked","blocked_reason"]
        feats = list(feature_order)
        obs_cols = [f"RAW_{k}" for k in feats] + [f"SCALED_{k}" for k in feats]
        step_parts = ["r_step_total","r_step_shaping","r_step_app_cost","r_step_spacing_pen","r_step_timing_pen"]
        cumu = ["G_cumulative_after_step"]
        term_parts_common = ["r_term_total","r_term_yield","r_term_nue_band","r_term_nsurp_pen","r_term_other","NUE","EUNEP","Nsurp","WSO","Napplied","Napps"]
        term_parts_nue_paper = ["phi_nue","phi_nsurp","prod","ycond","harvest_flag"]
        return base + obs_cols + step_parts + cumu + term_parts_common + term_parts_nue_paper

    def _ensure_header(self) -> None:
        if not self.csv_path.exists():
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with self.csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def _flatten_obs(self, obs: Mapping[str, float], prefix: str) -> Dict[str, float]:
        out: Dict[str, float | str] = {}
        for k in self.feature_order:
            col = f"{prefix}{k}"
            v = obs.get(k) if obs is not None else None
            out[col] = float(v) if v is not None else ""
        return out  # type: ignore[return-value]

    def _append_row(self, row: Mapping[str, Any]) -> None:
        safe_row = {k: row.get(k, "") for k in self.fieldnames}
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(safe_row)

    @staticmethod
    def _coerce_terminal_breakdown(obj: TerminalRewardBreakdown | Mapping[str, Any] | float | None) -> Dict[str, Any]:
        out = {k: "" for k in [
            "r_term_total","r_term_yield","r_term_nue_band","r_term_nsurp_pen","r_term_other",
            "NUE","EUNEP","Nsurp","WSO","Napplied","Napps","phi_nue","phi_nsurp","prod","ycond","harvest_flag"
        ]}
        if obj is None: return out
        if isinstance(obj, TerminalRewardBreakdown):
            data = asdict(obj)
        elif isinstance(obj, Mapping):
            data = dict(obj)
        else:
            data = {"r_term_total": float(obj)}
        for k in out.keys():
            if k in data and data[k] is not None:
                out[k] = data[k]
        return out
