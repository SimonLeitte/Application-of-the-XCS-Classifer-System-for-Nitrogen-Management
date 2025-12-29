# xcs_rewards.py — XCSR-friendly reward with potential shaping
from __future__ import annotations
from typing import Optional, Dict, Any
import math
import numpy as np

# -----------------------------
# CONFIG (tune here centrally)
# -----------------------------

# terminal paper-style normalization (for WSO only used at terminal)
WSO_MIN = 5000.0
WSO_MAX = 15000.0

# widths for clipped-linear φ terms
OMEGA_NUE   = 1.0     # width around NUE center
OMEGA_NSURP = 20.0   # width around Nsurp center

# target centers as in paper-style shaping
TARGET_NUE_CENTER   = 0.70
TARGET_NUE_HALFSPAN = 0.20          # band [0.5, 0.9]

TARGET_NSURP_CENTER   = 20.0
TARGET_NSURP_HALFSPAN = 20.0         # (center ± 20)

# DVS agronomic window
DVS_BOUNDS = (0.2, 1.4)

# weights inside potential Φ(s)
W_PHI_PRODUCT = 0.6    # weight for φ_NUE * φ_Nsurp (running uses φ_NUE≈1.0)
W_WSO_NORM    = 0.25
W_DVS_WIN     = 0.15

# shaping gamma (should roughly match agent's gamma)
#GAMMA_SHAPING = 0.99
KAPPA_SHAPING = 1.0

# application costs/penalties (only when effective_action > 0)
APPLY_COST = -0.1       # was -10 — makes “spam early” unattractive but not impossible
TARGET_GAP_DAYS = 21.0   # was 14 — aim for ~3 weeks if you step weekly
LAM_GAP = 0.5       # was 0.02 — make too-short gaps clearly bad when you DO apply
LAM_DVS_APP = 0.05       # keep

LAMBDA_NEG_SURPLUS = 0.2  # penalty per kg when Nsurp < 0
LAMBDA_POS_SURPLUS = 0.2  # penalty per kg when Nsurp > 40

# tiny nudge when a positive application was ATTEMPTED but the env blocked it
ATTEMPT_BLOCKED_COST = 0.0

# --------------------------------------------------------------------
# numeric helpers
# --------------------------------------------------------------------
def _f(x, default=0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default

def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))

def _norm01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return _clip01((x - lo) / (hi - lo))

# --------------------------------------------------------------------
# terminal φ terms (ground truth; use info[] at harvest)
# --------------------------------------------------------------------
def _phi_nue_final(nue: float, width: float = OMEGA_NUE) -> float:
    """Clipped linear around center 0.7 with halfspan 0.2, scaled by width."""
    v = 1.0 - ((abs(_f(nue) - TARGET_NUE_CENTER) - TARGET_NUE_HALFSPAN) / max(1e-9, width))
    return _clip01(v)

def _phi_nsurp_final(nsurp: float, width: float = OMEGA_NSURP) -> float:
    """Clipped linear around center 20 with halfspan 20, scaled by width."""
    v = 1.0 - ((abs(_f(nsurp) - TARGET_NSURP_CENTER) - TARGET_NSURP_HALFSPAN) / max(1e-9, width))
    return _clip01(v)

def _wso_norm_terminal(wso: float) -> float:
    return _norm01(_f(wso), WSO_MIN, WSO_MAX)

# --------------------------------------------------------------------
# running (dense) state-only ingredients for Φ(s)
# --------------------------------------------------------------------
def _phi_nsurp_running(nsurp_sofar: float, width: float = OMEGA_NSURP) -> float:
    """
    Running proxy for 'being near good surplus' using Nsurp_sofar.
    We avoid NUE mid-episode (not well-defined), so φ_NUE≈1 for shaping.
    """
    return _phi_nsurp_final(nsurp_sofar, width)

def _dvs_window(dvs: float, lo: float = DVS_BOUNDS[0], hi: float = DVS_BOUNDS[1]) -> float:
    """
    Triangular membership: 1.0 inside [lo, hi], decays to 0 outside.
    For simplicity, we return 1 inside window, and 1 - normalized distance outside.
    """
    d = _f(dvs)
    if lo <= d <= hi:
        return 1.0
    # distance to nearest bound normalized by window width
    win = max(1e-9, hi - lo)
    if d < lo:
        return _clip01(1.0 - (lo - d) / win)
    else:
        return _clip01(1.0 - (d - hi) / win)

def _wso_norm_running(wso: float) -> float:
    # use same bounds as terminal; it's a progress proxy in Φ(s)
    return _wso_norm_terminal(wso)

def _Phi(state: Dict[str, Any]) -> float:
    """
    Potential function Φ(s) = 0.6 * ( φ_NUE_running * φ_NSURP_running ) +
                              0.3 *  WSO_norm +
                              0.1 *  DVS_window
    For running φ_NUE we use 1.0 (omit), φ_NSURP uses Nsurp_sofar proxy.
    """
    dvs = _f(state.get("DVS"))
    wso = _f(state.get("WSO"))
    nsf = _f(state.get("Nsurp_sofar"))

    phi_nsurp = _phi_nsurp_running(nsf)
    phi_nue_running = 1.0  # no reliable NUE mid-episode

    prod = phi_nue_running * phi_nsurp
    wso_n = _wso_norm_running(wso)
    dvs_m = _dvs_window(dvs)

    Phi = (W_PHI_PRODUCT * prod) + (W_WSO_NORM * wso_n) + (W_DVS_WIN * dvs_m)
    # ensure finite
    return _f(Phi, default=0.0)

# --------------------------------------------------------------------
# per-application penalties (only when real fertilizer is applied)
# --------------------------------------------------------------------
def _application_costs(cur: Dict[str, Any], info: Dict[str, Any]) -> float:
    eff = _f(info.get("effective_action"), 0.0)
    # if a positive application was attempted but blocked, give a tiny nudge
    if eff <= 0.0:
        if bool(info.get("blocked", False)):
            return float(ATTEMPT_BLOCKED_COST)
        return 0.0

    r = 0.0
    # fixed per-application cost
    r += APPLY_COST

    # spacing penalty: prefer >= TARGET_GAP_DAYS between applications
    gap = _f(cur.get("last_app_gap_days"), 999.0)
    if gap < TARGET_GAP_DAYS:
        # normalized shortfall
        short = (TARGET_GAP_DAYS - gap) / max(1e-9, TARGET_GAP_DAYS)
        r -= LAM_GAP * short

    # DVS penalty: if outside agronomic window
    dvs = _f(cur.get("DVS"))
    lo, hi = DVS_BOUNDS
    if dvs < lo:
        r -= LAM_DVS_APP * ((lo - dvs) / max(1e-9, (hi - lo)))
    elif dvs > hi:
        r -= LAM_DVS_APP * ((dvs - hi) / max(1e-9, (hi - lo)))

    return float(r)

# --------------------------------------------------------------------
# terminal (harvest) reward
# --------------------------------------------------------------------
def _terminal_reward(cur: Dict[str, Any], info: Dict[str, Any]) -> float:
    """
    Paper-like terminal payoff:
      r_T = φ_NUE * φ_NSURP + 1{φ_NUE=1 & φ_NSURP=1} * WSO_norm
    Uses ground-truth NUE_EUNEP / Nsurp (in info at done).
    """
    nue   = info.get("NUE_EUNEP", None)
    nsurp = info.get("Nsurp", None)

    phiN = _phi_nue_final(_f(nue))     if nue   is not None else 0.0
    phiS = _phi_nsurp_final(_f(nsurp)) if nsurp is not None else 0.0
    prod = phiN * phiS

    ycond = 0.0
    if (phiN == 1.0) and (phiS == 1.0):
        # use raw WSO from state (scaler may also normalize upstream; that's fine)
        ycond = _wso_norm_terminal(_f(cur.get("WSO", 0.0)))

    rT = float(prod + ycond)

    nsurp_val = _f(info.get("Nsurp", 0.0), 0.0)
    if nsurp_val > 40.0:
        rT -= LAMBDA_POS_SURPLUS * (nsurp_val - 40.0)
    elif nsurp_val < 0.0:
        rT -= LAMBDA_NEG_SURPLUS * (-nsurp_val)

    if not math.isfinite(rT):
        rT = 0.0
    return rT

# --------------------------------------------------------------------
# PUBLIC REWARD FUNCTIONS
# --------------------------------------------------------------------
def reward_nue_paper(prev: Optional[Dict[str, Any]],
                     cur: Dict[str, Any],
                     info: Dict[str, Any]) -> float:
    """
    Dense + terminal reward for XCSR:
      r_t = (γ Φ(s') − Φ(s)) + costs_on_application(s', info)
      If done: add terminal payoff r_T (paper-like, NUE/Nsurp/WSO).
    Notes:
      - prev may be None at the first step → treat Φ(prev)=0.
      - All terms are finite by construction.
    """
    phi_prev = _Phi(prev) if isinstance(prev, dict) else 0.0
    phi_cur  = _Phi(cur)

    r = KAPPA_SHAPING * (phi_cur - phi_prev)
    r += _application_costs(cur, info)

    # terminal add-on
    done = bool(info.get("done", False))
    # Some trainers don't pass "done" in info; fall back on DVS>=2 if available
    if not done:
        dvs = _f(cur.get("DVS"), 0.0)
        done = done or (dvs >= 2.0)

    if done:
        r += _terminal_reward(cur, info)

    # final sanity
    if not math.isfinite(r):
        r = 0.0
    return float(r)

def reward_n_policy(prev: Optional[Dict[str, Any]],
                    cur: Dict[str, Any],
                    info: Dict[str, Any]) -> float:
    """
    Simpler variant: potential on (WSO_norm + DVS_window) only + application costs.
    Useful as ablation/baseline.
    """
    def Phi_simple(s: Dict[str, Any]) -> float:
        return 0.7 * _wso_norm_running(_f(s.get("WSO", 0.0))) + 0.3 * _dvs_window(_f(s.get("DVS", 0.0)))
    phi_prev = Phi_simple(prev) if isinstance(prev, dict) else 0.0
    phi_cur  = Phi_simple(cur)
    r = KAPPA_SHAPING * (phi_cur - phi_prev)
    r += _application_costs(cur, info)
    # terminal: small bump if harvest inside window (optional)
    dvs = _f(cur.get("DVS", 0.0))
    if dvs >= 2.0:
        r += 0.2
    return float(r)

def reward_constrained(prev: Optional[Dict[str, Any]],
                       cur: Dict[str, Any],
                       info: Dict[str, Any]) -> float:
    """
    Constrained-style: potential + hinge penalties on terminal metrics.
    Here kept lightweight; consider using 'nue_paper' as main reward.
    """
    r = reward_nue_paper(prev, cur, info)
    # (Optional) You could add extra hinge penalties at terminal here.
    return float(r)

# Reward function parameters for yield-driven and financial agents
# Reward function parameters for yield-driven and financial agents
YIELD_BETA = 1.0             # Multiplier (β) for fertilizer penalty in yield-based reward
YIELD_REWARD_SCALE = 0.01    # Scaling factor to keep yield-driven rewards comparable to others
YIELD_SHAPING_ENABLED = True  # Enable optional shaping terms for yield-driven reward
YIELD_SHAPING_WEIGHT_DVS = 0.3        # Weight for DVS-window shaping (if shaping enabled)
YIELD_SHAPING_WEIGHT_PROGRESS = 0.5   # Weight for progress (WSO norm) shaping (if enabled)
N_APPLY_CONST_BONUS_YIELD = 0.05

GRAIN_PRICE_PER_KG = 0.18167  # Grain price in €/kg (≈ €181.67 per 1000 kg)
N_PRICE_PER_KG = 0.2049       # Fertilizer N price in €/kg (≈ €20.49 per 100 kg)
PROFIT_REWARD_SCALE = 0.1     # Scaling factor to keep profit-based rewards comparable to others

# =======================
# Add/Update these constants (near your other reward params)
# =======================

BETA_IN_WINDOW_MULT = 0.3        # β is discounted inside the productive DVS window
APPLY_BONUS_YIELD_POST = 0.05    # +0.03..0.07 added AFTER scaling when dosing in-window
BONUS_WINDOW = (0.40, 1.20)      # productive sub-window (env allows 0.2–1.4)

# very small post-scale shaping (kept tiny so ΔWSO term dominates)
SHAPE_W_WSO = 0.02               # on Δ(WSO_norm) ≥ 0
SHAPE_W_DVS = 0.01               # on Δ(DVS_window) ≥ 0

def reward_relative_yield(prev: Optional[Dict[str, Any]],
                          cur: Dict[str, Any],
                          info: Dict[str, Any]) -> float:
    """
    Yield-driven reward (flipped to fertilize more when it pays):
      r_t = (ΔWSO - β_eff * N_applied) * scale
            + 1{in_window & N>0} * APPLY_BONUS_YIELD_POST
            + SHAPE_W_WSO * max(0, ΔWSO_norm) + SHAPE_W_DVS * max(0, ΔDVS_window)

    Where β_eff is discounted inside a productive DVS window to avoid
    punishing sensible, timely applications.
    """
    # --- read state
    prev_wso = _f(prev.get("WSO"), 0.0) if isinstance(prev, dict) else 0.0
    cur_wso  = _f(cur.get("WSO"), 0.0)
    delta_wso = cur_wso - prev_wso

    n_applied = _f(info.get("effective_action", info.get("dose_effective", 0.0)), 0.0)
    dvs_cur   = _f(cur.get("DVS"), 0.0)
    dvs_prev  = _f(prev.get("DVS"), 0.0) if isinstance(prev, dict) else 0.0

    lo, hi = BONUS_WINDOW

    # --- DVS-aware β: make N cheaper where it actually drives yield
    base_beta = globals().get("YIELD_BETA", 3.0)
    beta_mult = BETA_IN_WINDOW_MULT if (lo <= dvs_cur <= hi) else 1.0
    beta_eff  = base_beta * beta_mult

    # --- base reward (raw units), then scale
    reward = (delta_wso - beta_eff * n_applied)
    reward *= YIELD_REWARD_SCALE

    # --- tiny post-scale application bonus in the productive window
    if (n_applied > 0.0) and (lo <= dvs_cur <= hi):
        reward += APPLY_BONUS_YIELD_POST

    # --- very small, positive-only shaping to keep early dosing viable
    #     (uses your existing helpers; kept tiny so ΔWSO term dominates)
    wso_norm_cur  = _wso_norm_running(cur_wso)
    wso_norm_prev = _wso_norm_running(prev_wso)
    d_wso_norm = max(0.0, wso_norm_cur - wso_norm_prev)

    dvs_win_cur  = _dvs_window(dvs_cur)
    dvs_win_prev = _dvs_window(dvs_prev)
    d_dvs_win = max(0.0, dvs_win_cur - dvs_win_prev)

    reward += SHAPE_W_WSO * d_wso_norm
    reward += SHAPE_W_DVS * d_dvs_win

    # --- safety
    if not math.isfinite(reward):
        reward = 0.0
    return float(reward)



def reward_financial(prev: Optional[Dict[str, Any]],
                    cur: Dict[str, Any],
                    info: Dict[str, Any]) -> float:
    """
    Financial (profit-maximizing) reward:
    R_t = (ΔWSO * GRAIN_PRICE_PER_KG) - (N_applied * N_PRICE_PER_KG).
    Rewards immediate profit: grain yield increase value minus fertilizer cost.
    Scaled by PROFIT_REWARD_SCALE for compatibility.
    """
    # Calculate grain yield increase in kg/ha since last step
    prev_wso = _f(prev.get("WSO"), 0.0) if isinstance(prev, dict) else 0.0
    cur_wso  = _f(cur.get("WSO"), 0.0)
    delta_yield = cur_wso - prev_wso

    # Revenue from yield increase (€/ha for this step)
    revenue = delta_yield * GRAIN_PRICE_PER_KG

    # Cost of fertilizer applied this step (€/ha)
    n_applied = _f(info.get("effective_action", info.get("dose_effective", 0.0)), 0.0)
    cost = n_applied * N_PRICE_PER_KG

    # Profit = revenue - cost
    reward = revenue - cost

    # Scale the reward for numeric stability
    reward *= PROFIT_REWARD_SCALE

    # Ensure the reward is a finite number
    if not math.isfinite(reward):
        reward = 0.0
    return float(reward)




# --------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------
def make_reward(name: str = "nue_paper", **kwargs):
    """
    Factory function to retrieve a reward function by name.
    """
    key = (name or "").lower()
    if key in ("nue_paper", "paper_nue", "eunep_scalar"):
        return lambda prev, cur, info: reward_nue_paper(prev, cur, info)
    if key in ("n_policy", "n_management"):
        return lambda prev, cur, info: reward_n_policy(prev, cur, info)
    if key in ("constrained", "n_policy_constrained"):
        return lambda prev, cur, info: reward_constrained(prev, cur, info)
    if key in ("yield", "yield_relative", "relative_yield", "yield_driven"):
        return lambda prev, cur, info: reward_relative_yield(prev, cur, info)
    if key in ("financial", "profit", "financial_profit", "profit_max"):
        return lambda prev, cur, info: reward_financial(prev, cur, info)
    raise ValueError(f"Unknown reward name: {name}")

