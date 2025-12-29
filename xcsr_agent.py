# xcsr_agent.py — XCSR (real-valued) agent with centralized CONFIG + decision telemetry
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import json, time, hashlib

from xcsr_scaler import MinMaxScalerSelective

# ---- Subsumption maturity threshold ----
THETA_SUB = 50


# =======================
# Telemetrie / JSONL-Logger
# =======================

def _hash_feature_order(feature_order: List[str]) -> str:
    s = "|".join(map(str, feature_order))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


class AgentEventLogger:
    """
    JSONL-Logger für Agenten-Interna:
      - decision.jsonl   : Zustand (scaled+raw), p(a), Match-Set [M], Action-Set [A], gewählte Aktion
      - updates.jsonl    : Regel-Updates in [A] (vor/nach), Δp/ΔF/Δeps, target_type (TD/terminal)
      - covering.jsonl   : neu erzeugte Regeln + Ausgangszustand
      - ga.jsonl         : GA-Ereignisse (Parents/Offspring/Deletions)
      - population.jsonl : Snapshot der Population (vor/nach Episode)
    """
    def __init__(self, base_dir: Path, feature_order: List[str], scaler: Optional[MinMaxScalerSelective], run_tag: str = "xcsr"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.fo = list(feature_order)
        self.fo_hash = _hash_feature_order(self.fo)
        self.scaler = scaler
        self.run_tag = str(run_tag)

    # ---------- helpers ----------

    def _append_jsonl(self, name: str, obj: Dict[str, Any]) -> None:
        fp = self.base_dir / name
        with fp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")

    def _denorm_bounds(self, lower: List[float], upper: List[float]) -> Tuple[List[float], List[float]]:
        lo_raw, up_raw = [], []
        for i, (lo, up) in enumerate(zip(lower, upper)):
            try:
                lo_raw.append(float(self.scaler.denorm_i(i, float(lo))) if self.scaler is not None else float(lo))
                up_raw.append(float(self.scaler.denorm_i(i, float(up))) if self.scaler is not None else float(up))
            except Exception:
                lo_raw.append(float(lo)); up_raw.append(float(up))
        return lo_raw, up_raw

    def serialize_classifier(self, cl, cid: int) -> Dict[str, Any]:
        lower = list(map(float, cl.lower))
        upper = list(map(float, cl.upper))
        mask  = [bool(x) for x in cl.mask]
        lr_raw, ur_raw = self._denorm_bounds(lower, upper)
        # Beachte: unsere Felder heißen p (Prediction), eps (Error), F (Fitness)
        return {
            "id": int(cid),
            "action": int(getattr(cl, "action", -1)),
            "lower": lower,
            "upper": upper,
            "mask":  mask,
            "lower_raw": lr_raw,
            "upper_raw": ur_raw,
            "p": float(getattr(cl, "p", 0.0)),
            "eps": float(getattr(cl, "eps", 0.0)),
            "F": float(getattr(cl, "F", 0.0)),
            "exp": int(getattr(cl, "exp", 0)),
            "num": int(getattr(cl, "num", 1)),
            "as_size": float(getattr(cl, "as_size", 1.0)),
            "ts_ga": int(getattr(cl, "ts_ga", 0)),
        }

    # ---------- public writers ----------

    def log_decision(
        self,
        *,
        episode: int,
        step: int,
        state_scaled: List[float],
        state_raw: Optional[List[float]],
        p_array: List[float],
        action_chosen: int,
        exploration_mode: str,
        epsilon: float,
        match_ids: List[int],
        action_ids: List[int],
        top_rules: List[Dict[str, Any]],
        timestamp: Optional[float] = None
    ) -> None:
        row = {
            "ts": timestamp or time.time(),
            "run": self.run_tag,
            "fo_hash": self.fo_hash,
            "feature_order": self.fo,
            "episode": int(episode),
            "step": int(step),
            "exploration_mode": str(exploration_mode),
            "epsilon": float(epsilon),
            "state_scaled": list(map(float, state_scaled)),
            "state_raw": list(map(float, state_raw)) if state_raw is not None else None,
            "p_array": list(map(float, p_array)),
            "action_chosen": int(action_chosen),
            "match_set_ids": list(map(int, match_ids)),
            "action_set_ids": list(map(int, action_ids)),
            "top_rules": top_rules,
        }
        self._append_jsonl("decision.jsonl", row)

    def log_updates(self, *, episode: int, step: int, updates: List[Dict[str, Any]], timestamp: Optional[float] = None) -> None:
        row = {"ts": timestamp or time.time(), "run": self.run_tag, "episode": int(episode), "step": int(step), "updates": updates}
        self._append_jsonl("updates.jsonl", row)

    def log_covering(self, *, episode: int, step: int, trigger_state_scaled: List[float], new_rule: Dict[str, Any], timestamp: Optional[float] = None) -> None:
        row = {
            "ts": timestamp or time.time(), "run": self.run_tag,
            "episode": int(episode), "step": int(step),
            "trigger_state_scaled": list(map(float, trigger_state_scaled)),
            "new_rule": new_rule,
        }
        self._append_jsonl("covering.jsonl", row)

    def log_ga(
        self, *, episode: int, step: int, niche_action: int,
        parents: List[Dict[str, Any]], offspring: List[Dict[str, Any]], deletions: List[int],
        rates: Dict[str, float], timestamp: Optional[float] = None
    ) -> None:
        row = {
            "ts": timestamp or time.time(), "run": self.run_tag,
            "episode": int(episode), "step": int(step),
            "niche_action": int(niche_action),
            "parents": parents, "offspring": offspring, "deletions": deletions,
            "rates": {k: float(v) for k, v in rates.items()},
        }
        self._append_jsonl("ga.jsonl", row)

    def log_population_snapshot(self, *, episode: int, when: str, population: List[Dict[str, Any]], timestamp: Optional[float] = None) -> None:
        row = {"ts": timestamp or time.time(), "run": self.run_tag, "episode": int(episode), "when": when, "population": population}
        self._append_jsonl("population.jsonl", row)


# =======================
# XCSR-Config & Core
# =======================

@dataclass(frozen=True)
class XCSRConfig:
    # Exploration
    eps_start: float = 0.45
    eps_end: float = 0.02
    eps_decay: float = 0.992
    novelty_prob: float = 0.03

    # Learning
    beta: float = 0.4
    gamma: float = 0.99
    p_init: float = 0.0
    eps_init: float = 0.0
    F_init: float = 0.01

    # Accuracy function κ
    eps0: float = 0.25
    alpha: float = 0.1
    nu: float = 5.0

    # GA
    theta_GA: int = 500
    chi: float = 0.8
    mu: float = 0.04
    p_dontcare: float = 0.35
    cov_width: float = 0.15
    ga_exp_gate: int = 60

    # Deletion / population
    N: int = 800
    delta: float = 0.1

    # Initial observed ranges (updated online)
    init_min: float = -1.0
    init_max: float = 1.0

    # Toggles
    ga_enabled: bool = True
    covering_enabled: bool = True
    deletion_enabled: bool = True
    updates_enabled: bool = True


DEFAULT_XCSR_CONFIG = XCSRConfig()


@dataclass
class Classifier:
    """Single XCSR rule with interval conditions + don't-care mask."""
    lower: np.ndarray
    upper: np.ndarray
    mask:  np.ndarray
    action: int

    # XCS parameters
    p: float = 0.0       # prediction
    eps: float = 0.0     # absolute error
    F: float = 0.01      # fitness
    exp: int = 0         # experience
    num: int = 1         # numerosity
    as_size: float = 1.0
    ts_ga: int = 0

@dataclass
class ActionTrace:
    selected_action: int
    match_ids: list
    action_ids: list
    pred_array: list  # prediction array used for action selection (optional)


class XCSRAgent:
    """
    XCSR agent (real-valued XCS) for discrete actions and continuous observations.

    Public API:
        select_action(obs_vec: np.ndarray) -> int
        learn(obs_vec, action, reward, next_obs_vec, done) -> None
        end_of_episode_backup(R_terminal: float) -> None
        end_episode() -> None
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        seed: int = 0,
        config: XCSRConfig = DEFAULT_XCSR_CONFIG,
        scaler: Optional[MinMaxScalerSelective] = None,
        # NEW: optional telemetry
        feature_order: Optional[List[str]] = None,
        log_dir: Optional[Path] = None,
    ):
        self.cfg = config
        self.D = int(n_features)
        self.A = int(n_actions)
        self.rng = np.random.default_rng(seed)

        self.scaler = scaler

        # If a scaler is provided, we operate in a mixed space:
        # - normalized features (keep_mask == False) in [0,1]
        # - kept-raw features (keep_mask == True) in their raw ranges
        if self.scaler is not None:
            self.fixed_domain = True
            self.keep_mask = self.scaler.keep_mask
            # domain bounds in the space we operate in
            self.dom_lo = self.scaler.lo.copy()
            self.dom_hi = self.scaler.hi.copy()
            idx = ~self.keep_mask  # normalized dims
            self.dom_lo[idx] = 0.0
            self.dom_hi[idx] = 1.0
            # use fixed domain for min/max (no online updates)
            self.min = self.dom_lo.copy()
            self.max = self.dom_hi.copy()
        else:
            self.fixed_domain = False
            # original behavior (online ranges)
            self.min = np.full(self.D, self.cfg.init_min, dtype=float)
            self.max = np.full(self.D, self.cfg.init_max, dtype=float)
            # for clipping
            self.dom_lo = self.min.copy()
            self.dom_hi = self.max.copy()

        # exploration
        self.eps = self.cfg.eps_start

        # time
        self.t = 0

        # population
        self.pop: List[Classifier] = []

        # episode action sets for Monte-Carlo terminal backup
        self.ep_sets: List[List[Classifier]] = []

        # indices for logging context (episode/step set by trainer)
        self._episode_idx: int = -1
        self._step_idx: int = -1
        self._last_exploration: str = "policy"  # "novelty"|"eps"|"greedy"/"policy"

        # ----- telemetry logger -----
        self.logger: Optional[AgentEventLogger] = None

        # ===== MUST-CARE: store feature names & settings =====
        self.feature_order = list(feature_order) if feature_order is not None else [f"f{i}" for i in range(self.D)]
        # names we always want to "care" about
        self.must_care_features = set(["DVS", "last_app_gap_days", "cum_N_applied_kg_ha"])
        # per-feature dontcare override (lower -> care more often)
        self.p_dontcare_default = self.cfg.p_dontcare
        self.p_dontcare_per_feat = {name: 0.05 for name in self.must_care_features}
        # when caring at covering/repair time → small interval around x
        self.cover_width_mustcare = max(0.05, self.cfg.cov_width * 0.8)

        try:
            if feature_order is not None:
                base = Path(log_dir) if log_dir is not None else Path("agent_logs")
                self.logger = AgentEventLogger(base, feature_order, scaler=self.scaler, run_tag="xcsr")
        except Exception as e:
            print(f"[warn] AgentEventLogger init failed: {e}")
            self.logger = None

        # ---- cache feature indices for special handling (available everywhere) ----
        try:
            self.feature_index = {name: i for i, name in enumerate(self.feature_order)}
            self.i_dvs  = self.feature_index.get("DVS")
            self.i_gap  = self.feature_index.get("last_app_gap_days")
            self.i_dose = self.feature_index.get("last_dose_kg_ha")
        except Exception:
            self.feature_index = {}
            self.i_dvs = self.i_gap = self.i_dose = None

    # ------------------------------
    # public API
    # ------------------------------

    def _prep_x(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=float)
        if self.scaler is not None:
            x = self.scaler.fwd_vec(x)
        return x

    def _check_x(self, x: np.ndarray):
        if x.ndim != 1 or x.shape[0] != self.D:
            raise ValueError(f"Feature dimension mismatch: got {x.shape}, expected ({self.D},)")

    def _least_experienced_action(self, match_indices: List[int]) -> int:
        best_a, best = 0, float("inf")
        for a in range(self.A):
            exps = [self.pop[i].exp for i in match_indices if self.pop[i].action == a]
            mean_exp = -1.0 if len(exps) == 0 else float(np.mean(exps))  # prefer empty niches
            if mean_exp < best:
                best, best_a = mean_exp, a
        return int(best_a)

    def _raw_vec(self, x: np.ndarray) -> np.ndarray:
        """Convert mixed/scaled x back to raw units (kept-raw dims stay unchanged)."""
        try:
            return self.scaler.inv_vec(x) if self.scaler is not None else x
        except Exception:
            return x

    def _in_noop_region(self, x: np.ndarray) -> bool:
        """True when no positive action can be legal (e.g., DVS outside [0.2, 1.4])."""
        try:
            if self.i_dvs is None:
                return False
            dvs_raw = float(self._raw_vec(x)[self.i_dvs])
            return (dvs_raw < 0.2) or (dvs_raw > 1.4)
        except Exception:
            return False

    def select_action(self, obs: np.ndarray) -> int:
        x = self._prep_x(obs)
        self._check_x(x)
        if not self.fixed_domain:
            self._update_ranges(x)
        M = self._match_set(x)

        # novelty-biased exploration
        if self.rng.random() < self.cfg.novelty_prob:
            self._last_exploration = "novelty"
            a = self._least_experienced_action(M)
            # ensure niche exists now (avoid deferred cover in learn)
            if self.cfg.covering_enabled:
                A_idx = [i for i in M if self.pop[i].action == a]
                if (len(A_idx) == 0) and (not self._in_noop_region(x)):
                    self._cover(x, a)
                    M = self._match_set(x)
        else:
            if len(M) == 0:
                if self.cfg.covering_enabled and not self._in_noop_region(x):
                    a_cov = int(self.rng.integers(0, self.A))
                    self._cover(x, a_cov)  # logs in _cover
                    M = self._match_set(x)
                else:
                    self._last_exploration = "policy"
                    return 0  # don't cover in non-actionable regions

            preds = self._prediction_array(M)
            if self.rng.random() < self.eps:
                self._last_exploration = "eps"
                valid = np.where(np.isfinite(preds))[0]
                if valid.size == 0:
                    a = int(self.rng.integers(0, self.A))
                else:
                    a = int(self.rng.choice(valid))
            else:
                self._last_exploration = "greedy"
                a = int(np.nanargmax(preds))

        # Build action-set for logging
        A_idx = [i for i in M if self.pop[i].action == a]
        preds = self._prediction_array(M) if 'preds' not in locals() else preds

        # Decision telemetry
        if self.logger is not None:
            try:
                # reconstruct raw from scaled for readability
                raw_from_scaled = []
                for i, val in enumerate(x):
                    try:
                        raw_from_scaled.append(float(self.scaler.denorm_i(i, float(val))) if self.scaler is not None else float(val))
                    except Exception:
                        raw_from_scaled.append(float(val))

                match_ids  = [id(self.pop[i]) for i in M]
                action_ids = [id(self.pop[i]) for i in A_idx]
                topK = sorted(A_idx, key=lambda j: self.pop[j].F, reverse=True)[:5]
                top_rules = [self.logger.serialize_classifier(self.pop[j], cid=id(self.pop[j])) for j in topK]

                self.logger.log_decision(
                    episode=self._episode_idx,
                    step=self._step_idx,
                    state_scaled=[float(xx) for xx in x],
                    state_raw=raw_from_scaled,
                    p_array=[float(z) for z in np.asarray(preds).ravel().tolist()] if preds is not None else [],
                    action_chosen=int(a),
                    exploration_mode=self._last_exploration,
                    epsilon=float(self.eps),
                    match_ids=match_ids,
                    action_ids=action_ids,
                    top_rules=top_rules,
                )
            except Exception as e:
                print(f"[warn] decision log failed: {e}")

        return int(a)

    def learn(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        if not self.cfg.updates_enabled:
            return

        x = self._prep_x(obs)
        self._check_x(x)

        if not done:
            xn = self._prep_x(next_obs)
            self._check_x(xn)
        else:
            xn = x

        if not self.fixed_domain:
            self._update_ranges(x)
            if not done:
                self._update_ranges(xn)

        M = self._match_set(x)

        # Ensure selected action exists in [M]; if not, cover it (but not in no-op regions)
        A_idx = [i for i in M if self.pop[i].action == action]
        if len(A_idx) == 0:
            if self.cfg.covering_enabled and not self._in_noop_region(x):
                self._cover(x, action)  # logs in _cover
                M = self._match_set(x)
                A_idx = [i for i in M if self.pop[i].action == action]
            else:
                A_idx = []

        # store the niche members for terminal Monte-Carlo backup
        self.ep_sets.append([self.pop[i] for i in A_idx])

        # TD target
        if done:
            P = float(reward)
        else:
            Mn = self._match_set(xn)
            if len(Mn) == 0:
                maxPn = 0.0
            else:
                preds_next = self._prediction_array(Mn)
                maxPn = float(np.nanmax(np.where(np.isfinite(preds_next), preds_next, -np.inf)))
                if not np.isfinite(maxPn):
                    maxPn = 0.0
            P = float(reward) + self.cfg.gamma * maxPn

        # Snapshot BEFORE update (for telemetry)
        before = [{
            "id": id(self.pop[i]),
            "p": float(self.pop[i].p), "eps": float(self.pop[i].eps), "F": float(self.pop[i].F),
            "exp": int(self.pop[i].exp), "num": int(self.pop[i].num)
        } for i in A_idx]

        # Update action set parameters
        self._update_action_set(A_idx, P)

        # Snapshot AFTER update
        after = [{
            "id": id(self.pop[i]),
            "p": float(self.pop[i].p), "eps": float(self.pop[i].eps), "F": float(self.pop[i].F),
            "exp": int(self.pop[i].exp), "num": int(self.pop[i].num)
        } for i in A_idx]

        # Log Δ
        if self.logger is not None and A_idx:
            try:
                by_id = {d["id"]: d for d in after}
                updates = []
                for b in before:
                    a2 = by_id.get(b["id"])
                    if a2 is None:
                        continue
                    updates.append({
                        "id": b["id"],
                        "p_before": b["p"],   "p_after": a2["p"],   "d_p":   a2["p"]   - b["p"],
                        "F_before": b["F"],   "F_after": a2["F"],   "d_F":   a2["F"]   - b["F"],
                        "eps_before": b["eps"], "eps_after": a2["eps"], "d_eps": a2["eps"] - b["eps"],
                        "exp_before": b["exp"], "exp_after": a2["exp"], "d_exp": a2["exp"] - b["exp"],
                        "num_before": b["num"], "num_after": a2["num"],
                        "target_type": "TD" if not done else "terminal",
                        "P_target": float(P),
                        "reward_step": float(reward),
                    })
                self.logger.log_updates(episode=self._episode_idx, step=self._step_idx, updates=updates)
            except Exception as e:
                print(f"[warn] updates log failed: {e}")

        # Action-set subsumption (compress [A] before GA)
        if A_idx:
            self._action_set_subsumption(A_idx)
            A_idx = [i for i in M if (self.pop[i].action == action and self.pop[i].num > 0)]

        # GA in action set
        if self.cfg.ga_enabled:
            self._run_GA(A_idx, x)

        # Deletion if overpopulated
        if self.cfg.deletion_enabled:
            self._deletion()

        self.t += 1

    def end_episode(self):
        """Decay exploration after each episode."""
        self.eps = max(self.cfg.eps_end, self.eps * self.cfg.eps_decay)

    def end_of_episode_backup(self, R_terminal: float, gamma: float = 1.0):
        """Monte-Carlo backup for sparse terminal rewards across stored action sets."""
        if not self.ep_sets:
            return

        G = float(R_terminal)
        for A in reversed(self.ep_sets):
            if not A: continue

            as_size = sum(cl.num for cl in A)
            for cl in A:
                cl.exp += 1
                cl.as_size += self.cfg.beta * (as_size - cl.as_size)

                p_old = cl.p
                cl.eps += self.cfg.beta * (abs(G - p_old) - cl.eps)
                cl.p   += self.cfg.beta * (G - p_old)

            # fitness toward relative accuracy
            kappa = []
            for cl in A:
                if cl.eps < self.cfg.eps0:
                    k = 1.0
                else:
                    k = self.cfg.alpha * (cl.eps / self.cfg.eps0) ** (-self.cfg.nu)
                kappa.append(k * cl.num)

            denom = sum(kappa) if kappa else 1.0
            for cl, acc in zip(A, kappa):
                k_rel = acc / denom if denom > 0 else 0.0
                cl.F += self.cfg.beta * (k_rel - cl.F)

        self.ep_sets.clear()

    # ------------------------------
    # MUST-CARE helpers
    # ------------------------------
    def _feat_idx(self, name: str) -> int:
        try:
            return self.feature_order.index(name)
        except Exception:
            return -1

    def _is_must_care_idx(self, i: int) -> bool:
        if i < 0 or i >= self.D:
            return False
        try:
            return self.feature_order[i] in self.must_care_features
        except Exception:
            return False

    def _p_dontcare_for_idx(self, i: int) -> float:
        try:
            name = self.feature_order[i]
            return self.p_dontcare_per_feat.get(name, self.p_dontcare_default)
        except Exception:
            return self.p_dontcare_default

    def _enforce_mustcare_after_ga(self, c: Classifier, x: np.ndarray):
        """Ensure must-care dims are cared and have a small interval around current x."""
        for i in range(self.D):
            if not self._is_must_care_idx(i):
                continue
            if c.mask[i]:
                r = self._range_width(i)
                span = max(self.cover_width_mustcare * r, 1e-9)
                c.lower[i] = x[i] - span
                c.upper[i] = x[i] + span
                c.mask[i]  = False
        self._clip_bounds(c)

    # ------------------------------
    # core XCSR helpers
    # ------------------------------
    def _match_set(self, x: np.ndarray) -> List[int]:
        return [i for i, cl in enumerate(self.pop) if self._matches(cl, x)]

    def _matches(self, cl: Classifier, x: np.ndarray) -> bool:
        if cl.mask.all():  # fully general
            return True
        ok_low = x >= cl.lower
        ok_up  = x <= cl.upper
        ok = np.logical_or(cl.mask, np.logical_and(ok_low, ok_up))
        return bool(np.all(ok))

    def _prediction_array(self, match_indices: List[int]) -> np.ndarray:
        preds = np.full(self.A, -np.inf, dtype=float)
        for a in range(self.A):
            num = 0.0
            den = 0.0
            for i in match_indices:
                cl = self.pop[i]
                if cl.action != a:
                    continue
                num += cl.F * cl.p * cl.num
                den += cl.F * cl.num
            preds[a] = (num / den) if den > 0.0 else -np.inf
        return preds

    def _update_action_set(self, A_idx: List[int], P: float):
        if not A_idx:
            return
        as_size = sum(self.pop[i].num for i in A_idx)

        for i in A_idx:
            cl = self.pop[i]
            cl.exp += 1
            cl.as_size += self.cfg.beta * (as_size - cl.as_size)
            cl.eps += self.cfg.beta * (abs(P - cl.p) - cl.eps)
            cl.p   += self.cfg.beta * (P - cl.p)

        # compute accuracy κ_i
        kappa = []
        for i in A_idx:
            cl = self.pop[i]
            if cl.eps < self.cfg.eps0:
                k = 1.0
            else:
                k = self.cfg.alpha * (cl.eps / self.cfg.eps0) ** (-self.cfg.nu)
            kappa.append(k * cl.num)

        denom = sum(kappa) if kappa else 1.0
        for i, rel in zip(A_idx, kappa):
            cl = self.pop[i]
            k_rel = rel / denom if denom > 0 else 0.0
            cl.F += self.cfg.beta * (k_rel - cl.F)

    # ------------------------------
    # GA: selection, crossover, mutation, insertion, subsumption
    # ------------------------------
    def _run_GA(self, A_idx: List[int], x: np.ndarray):
        if not A_idx:
            return

        avg_time_since = np.mean([self.t - self.pop[i].ts_ga for i in A_idx])
        if avg_time_since < self.cfg.theta_GA:
            return

        avg_exp = float(np.mean([self.pop[i].exp for i in A_idx])) if A_idx else 0.0
        if avg_exp < self.cfg.ga_exp_gate:
            return

        for i in A_idx:
            self.pop[i].ts_ga = self.t

        # Fitness-proportionate selection within niche
        def select() -> Classifier:
            Fvals = np.array([self.pop[i].F for i in A_idx], dtype=float)
            Fsum = float(Fvals.sum()) or 1.0
            probs = Fvals / Fsum
            parent = self.pop[int(np.random.choice(A_idx, p=probs))]
            return parent

        p1 = select()
        p2 = select()

        c1 = self._clone_classifier(p1)
        c2 = self._clone_classifier(p2)

        # Crossover (uniform)
        if self.rng.random() < self.cfg.chi:
            self._uniform_crossover(c1, c2)

        # Mutation (must-care aware)
        self._mutate(c1, x)
        self._mutate(c2, x)

        # Enforce must-care after GA
        self._enforce_mustcare_after_ga(c1, x)
        self._enforce_mustcare_after_ga(c2, x)

        self.feature_index = {name: i for i, name in enumerate(self.feature_order)}
        self.i_dvs = self.feature_index.get("DVS")
        self.i_gap = self.feature_index.get("last_app_gap_days")
        self.i_dose = self.feature_index.get("last_dose_kg_ha")

        # Offspring param init (parents' averages)
        for c in (c1, c2):
            c.p = (p1.p + p2.p) / 2.0
            c.eps = (p1.eps + p2.eps) / 2.0
            c.F = (p1.F + p2.F) / 2.0
            c.exp = 0
            c.num = 1
            c.as_size = (p1.as_size + p2.as_size) / 2.0
            c.ts_ga = self.t

        # Subsumption / insertion
        inserted = 0
        for child in (c1, c2):
            if self._try_subsume(p1, child) or self._try_subsume(p2, child):
                continue
            if self._try_offspring_subsume_into_niche(child, A_idx):
                continue
            self.pop.append(child)
            inserted += 1

        if inserted == 0:
            self.pop.append(c1)

        # GA telemetry
        if self.logger is not None:
            try:
                parents = [self.logger.serialize_classifier(p1, id(p1)), self.logger.serialize_classifier(p2, id(p2))]
                offspring = [self.logger.serialize_classifier(c1, id(c1)), self.logger.serialize_classifier(c2, id(c2))]
                self.logger.log_ga(
                    episode=self._episode_idx, step=self._step_idx,
                    niche_action=int(self.pop[A_idx[0]].action) if A_idx else -1,
                    parents=parents, offspring=offspring, deletions=[],
                    rates={"chi": self.cfg.chi, "mu": self.cfg.mu, "theta_GA": float(self.cfg.theta_GA)}
                )
            except Exception as e:
                print(f"[warn] ga log failed: {e}")

    def _clone_classifier(self, cl: Classifier) -> Classifier:
        return Classifier(
            lower=cl.lower.copy(),
            upper=cl.upper.copy(),
            mask=cl.mask.copy(),
            action=int(cl.action),
            p=float(cl.p),
            eps=float(cl.eps),
            F=float(cl.F),
            exp=int(cl.exp),
            num=int(cl.num),
            as_size=float(cl.as_size),
            ts_ga=int(cl.ts_ga),
        )

    def _uniform_crossover(self, c1: Classifier, c2: Classifier):
        for i in range(self.D):
            if self.rng.random() < 0.5:
                c1.lower[i], c2.lower[i] = c2.lower[i], c1.lower[i]
                c1.upper[i], c2.upper[i] = c2.upper[i], c1.upper[i]
                c1.mask[i],  c2.mask[i]  = c2.mask[i],  c1.mask[i]
        if self.rng.random() < 0.5:
            c1.action, c2.action = c2.action, c1.action
        for c in (c1, c2):
            lo = np.minimum(c.lower, c.upper)
            up = np.maximum(c.lower, c.upper)
            c.lower, c.upper = lo, up

    def _mutate(self, c: Classifier, x: np.ndarray):
        for i in range(self.D):
            must = self._is_must_care_idx(i)

            # mask toggle: never wildcard must-care
            if self.rng.random() < self.cfg.mu:
                if must:
                    # if by chance it was wildcarded, force caring small interval
                    if c.mask[i]:
                        r = self._range_width(i)
                        span = max(self.cover_width_mustcare * r, 1e-9)
                        c.lower[i] = x[i] - span
                        c.upper[i] = x[i] + span
                        c.mask[i]  = False
                else:
                    c.mask[i] = ~c.mask[i]

            # bounds jitter/retarget if caring
            if not c.mask[i]:
                r = self._range_width(i)
                if self.rng.random() < self.cfg.mu:
                    # retarget around current x (helps niche tracking)
                    span = max(self.cfg.cov_width * r, 1e-9)
                    center = x[i]
                    c.lower[i] = center - span
                    c.upper[i] = center + span
                else:
                    width = c.upper[i] - c.lower[i]
                    jitter = 0.2 * width if width > 0 else 0.02 * r
                    c.lower[i] += self.rng.uniform(-jitter, jitter)
                    c.upper[i] += self.rng.uniform(-jitter, jitter)

        # occasionally mutate action
        if self.rng.random() < self.cfg.mu:
            c.action = int(self.rng.integers(0, self.A))

        self._clip_bounds(c)

    # -------- Subsumption helpers --------
    def _generality(self, cl: Classifier) -> int:
        return int(np.sum(cl.mask))

    def _more_general_than(self, a: Classifier, b: Classifier) -> bool:
        if np.sum(a.mask) <= np.sum(b.mask):
            return False
        if np.any((~a.mask) & b.mask):
            return False
        tight = (~a.mask) & (~b.mask)
        if np.any(a.lower[tight] > b.lower[tight]): return False
        if np.any(a.upper[tight] < b.upper[tight]): return False
        return True

    def _try_subsume(self, parent: Classifier, child: Classifier) -> bool:
        if parent.action != child.action: return False
        if not (parent.exp >= THETA_SUB and parent.eps <= self.cfg.eps0): return False
        # must-care: parent must care on all must-care dims
        for i in range(self.D):
            if self._is_must_care_idx(i) and parent.mask[i]:
                return False
        if not self._more_general_than(parent, child): return False
        parent.num += child.num
        return True

    def _try_offspring_subsume_into_niche(self, child: Classifier, A_idx: List[int]) -> bool:
        for i in A_idx:
            cl = self.pop[i]
            if cl.action != child.action: continue
            if not (cl.exp >= THETA_SUB and cl.eps <= self.cfg.eps0): continue
            # must-care: candidate parent must care on all must-care dims
            if any(self._is_must_care_idx(k) and cl.mask[k] for k in range(self.D)):
                continue
            if self._more_general_than(cl, child):
                cl.num += child.num
                return True
        return False

    def _action_set_subsumption(self, A_idx: List[int]) -> None:
        candidates = [
            i for i in A_idx
            if (self.pop[i].exp >= THETA_SUB and self.pop[i].eps <= self.cfg.eps0)
               and not any(self._is_must_care_idx(k) and self.pop[i].mask[k] for k in range(self.D))
        ]
        if not candidates:
            return
        subsumer_idx = max(candidates, key=lambda i: (self._generality(self.pop[i]), self.pop[i].F))
        subsumer = self.pop[subsumer_idx]
        for j in A_idx:
            if j == subsumer_idx: continue
            cl = self.pop[j]
            if cl.action != subsumer.action: continue
            if self._more_general_than(subsumer, cl):
                subsumer.num += cl.num
                cl.num = 0

    # ------------------------------
    # covering & deletion
    # ------------------------------
    def _ensure_matches_now(self, c: Classifier, x: np.ndarray) -> None:
        """Post-condition: for all cared dims, current x lies inside [lower, upper]."""
        for i in range(self.D):
            if c.mask[i]:
                continue
            if x[i] < c.lower[i] or x[i] > c.upper[i]:
                r = self._range_width(i)
                span = max(self.cover_width_mustcare * r, 1e-9)
                lo = x[i] - span
                up = x[i] + span
                # clip to domain
                lo = max(self.dom_lo[i], min(lo, up))
                up = min(self.dom_hi[i], max(lo, up))
                c.lower[i], c.upper[i] = lo, up
        # final rectify
        lo = np.minimum(c.lower, c.upper)
        up = np.maximum(c.lower, c.upper)
        c.lower, c.upper = lo, up

    def _cover(self, x: np.ndarray, action: int):
        lower = np.empty(self.D, dtype=float)
        upper = np.empty(self.D, dtype=float)
        mask  = np.zeros(self.D, dtype=bool)

        for i in range(self.D):
            must = self._is_must_care_idx(i)
            p_dc = self._p_dontcare_for_idx(i)
            dontcare = (self.rng.random() < p_dc) and (not must)

            if dontcare:
                mask[i] = True
                lower[i] = self.min[i]
                upper[i] = self.max[i]
            else:
                r = self._range_width(i)
                span = max(self.cover_width_mustcare * r, 1e-9)
                lower[i] = x[i] - span
                upper[i] = x[i] + span
                mask[i]  = False

        cl = Classifier(
            lower=lower, upper=upper, mask=mask, action=int(action),
            p=self.cfg.p_init, eps=self.cfg.eps_init, F=self.cfg.F_init,
            exp=0, num=1, as_size=1.0, ts_ga=self.t
        )
        self._clip_bounds(cl)
        self._ensure_matches_now(cl, x)  # invariant: covered rule must match now
        self.pop.append(cl)
        self._deletion()

        # Covering-Log
        if self.logger is not None:
            try:
                self.logger.log_covering(
                    episode=self._episode_idx,
                    step=self._step_idx,
                    trigger_state_scaled=[float(xx) for xx in x],
                    new_rule=self.logger.serialize_classifier(cl, id(cl))
                )
            except Exception as e:
                print(f"[warn] covering log failed: {e}")

    def _deletion(self):
        def pop_numerosity() -> int:
            return int(sum(cl.num for cl in self.pop))

        if self.pop:
            self.pop = [cl for cl in self.pop if cl.num > 0]

        while self.pop and pop_numerosity() > self.cfg.N:
            avgF = np.mean([cl.F for cl in self.pop]) or 1e-9
            votes = []
            for cl in self.pop:
                vote = cl.as_size * cl.num
                if (cl.F / max(cl.num, 1)) < (self.cfg.delta * avgF):
                    vote *= (avgF / max(cl.F / max(cl.num, 1), 1e-9))
                votes.append(vote)
            idx = int(np.argmax(votes))
            if self.pop[idx].num > 1:
                self.pop[idx].num -= 1
            else:
                self.pop.pop(idx)

    # ------------------------------
    # ranges & bounds
    # ------------------------------
    def _update_ranges(self, x: np.ndarray):
        self.min = np.minimum(self.min, x)
        self.max = np.maximum(self.max, x)

    def _range_width(self, i: int) -> float:
        return float(max(self.max[i] - self.min[i], 1e-6))

    def _clip_bounds(self, c):
        # Rectify swaps first
        lo = np.minimum(c.lower, c.upper)
        up = np.maximum(c.lower, c.upper)

        # Per-dimension margin (10% of domain width); ensure it’s a vector
        margin = 0.1 * (self.dom_hi - self.dom_lo)
        if np.isscalar(margin):
            margin = np.full(self.D, margin, dtype=float)

        # Zero margin for specific features to avoid drift beyond domain
        for i in (getattr(self, "i_dvs", None),
                  getattr(self, "i_gap", None),
                  getattr(self, "i_dose", None)):
            if i is not None:
                margin[i] = 0.0

        # Clamp to expanded domain
        c.lower = np.maximum(lo, self.dom_lo - margin)
        c.upper = np.minimum(up, self.dom_hi + margin)

        # Defensive: rectify again after clamping
        lo2 = np.minimum(c.lower, c.upper)
        up2 = np.maximum(c.lower, c.upper)
        c.lower, c.upper = lo2, up2


    @staticmethod
    def _rule_id(cl):
        for name in ("id", "clid", "uid", "rule_id", "rid", "label"):
            if hasattr(cl, name):
                return getattr(cl, name)
        return id(cl)  # fallback: Python object id

    def select_action_with_trace(self, obs: np.ndarray):
        """
        Select action exactly like select_action, but also return a per-step trace.

        Returns:
            (action_index, ActionTrace)
                - action_index: int, chosen action
                - ActionTrace:
                    selected_action : int
                    match_ids       : List[int]  (ids of rules in match set [M])
                    action_ids      : List[int]  (ids of rules in action set [A])
                    pred_array      : List[float] (prediction array used to pick the action)
        Notes:
            - May perform covering if enabled and needed (same as select_action).
            - Does not perform learning/updates/GA (those happen in learn()).
        """
        # Prep and domain maintenance (identical to select_action)
        x = self._prep_x(obs)
        self._check_x(x)
        if not self.fixed_domain:
            self._update_ranges(x)

        # Build match set
        M = self._match_set(x)
        preds = None  # will be filled when needed

        # Novelty-biased exploration branch
        if self.rng.random() < self.cfg.novelty_prob:
            self._last_exploration = "novelty"
            a = self._least_experienced_action(M)

            # Ensure niche exists now (avoid deferred cover in learn), as in select_action
            if self.cfg.covering_enabled:
                A_idx_tmp = [i for i in M if self.pop[i].action == a]
                if (len(A_idx_tmp) == 0) and (not self._in_noop_region(x)):
                    self._cover(x, a)
                    M = self._match_set(x)  # recompute with the covered rule present
        else:
            # If nothing matches: cover (when allowed) or return policy default (0)
            if len(M) == 0:
                if self.cfg.covering_enabled and not self._in_noop_region(x):
                    a_cov = int(self.rng.integers(0, self.A))
                    self._cover(x, a_cov)
                    M = self._match_set(x)
                else:
                    self._last_exploration = "policy"
                    # No matching rules and no covering → deterministic no-op action 0
                    # Produce an empty trace for completeness.
                    trace = ActionTrace(
                        selected_action=0,
                        match_ids=[],
                        action_ids=[],
                        pred_array=[],
                    )
                    return 0, trace

            # Compute prediction array over actions for ε/greedy choice
            preds = self._prediction_array(M)

            # ε-greedy exploration
            if self.rng.random() < self.eps:
                self._last_exploration = "eps"
                valid = np.where(np.isfinite(preds))[0]
                if valid.size == 0:
                    a = int(self.rng.integers(0, self.A))
                else:
                    a = int(self.rng.choice(valid))
            else:
                self._last_exploration = "greedy"
                a = int(np.nanargmax(preds))

        # Build action set for the chosen action; ensure preds is available
        A_idx = [i for i in M if self.pop[i].action == a]
        if preds is None:
            preds = self._prediction_array(M)

        # Assemble trace using the same id() scheme as your logger
        trace = ActionTrace(
            selected_action=int(a),
            match_ids=[id(self.pop[i]) for i in M],
            action_ids=[id(self.pop[i]) for i in A_idx],
            pred_array=[float(z) for z in np.asarray(preds).ravel().tolist()] if preds is not None else [],
        )
        return int(a), trace
