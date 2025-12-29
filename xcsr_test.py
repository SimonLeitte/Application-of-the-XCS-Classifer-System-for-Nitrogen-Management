# xcsr_test.py — Frozen-policy evaluation, classifier usage & fertilizer-step logging
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from xcsr_environment import default_paths, WofostXcsEnv
from xcsr_wrapper import XcsObsWrapper
from xcsr_rewards import make_reward
from xcsr_agent import XCSRAgent


# ---------------------------------------------------------------------------
# Feature names (must match training feature_order / env observation keys)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "DVS",
    "LAI",
    "RFTRA",
    "WSO",
    "Nsurp_sofar",
    "last_dose_kg_ha",
    "last_app_gap_days",
    "cum_N_applied_kg_ha",
    "IRRAD_7",
    "TMEAN_7",
    "RAIN_7",
    "Ndep_sum",
    "SM_mean",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def default_validation_years(n: int = 32, lo: int = 1983, hi: int = 2019) -> List[int]:
    """Evenly spaced years in [lo, hi], length n (rounded to ints, unique)."""
    if n <= 0:
        return []
    if lo > hi:
        lo, hi = hi, lo
    years = np.linspace(lo, hi, num=n, endpoint=True)
    years = np.unique(np.round(years).astype(int)).tolist()
    return years


def freeze_agent(agent: XCSRAgent) -> None:
    """Make policy deterministic and non-learning without modifying the rule base."""
    agent.eps = 0.0
    try:
        agent.cfg = dc_replace(
            agent.cfg,
            eps_start=0.0,
            eps_end=0.0,
            novelty_prob=0.0,
            ga_enabled=False,
            covering_enabled=False,
            deletion_enabled=False,
            updates_enabled=False,
        )
    except Exception:
        # If cfg is not present / unexpected, at least enforce eps=0
        pass


def build_env(paths, years: List[int], stride_days: int, layer_count: int) -> WofostXcsEnv:
    """
    Build WofostXcsEnv that will iterate deterministically over `years`.
    Supports both env signatures (with or without stride_days).
    """
    try:
        env = WofostXcsEnv(
            paths,
            mode="test",
            layer_count=layer_count,
            train_years=years,
            stride_days=stride_days,  # may raise TypeError on older envs
        )
    except TypeError:
        env = WofostXcsEnv(paths, mode="test", layer_count=layer_count, train_years=years)
        print("[eval] Note: your environment does not accept stride_days; using its built-in stride.")
    return env


def make_id_to_index(agent: XCSRAgent) -> Dict[int, int]:
    """
    Map Python object id(pop[i]) -> classifier_index (i).
    This lets us map trace.action_ids/trace.match_ids back to indices.
    """
    return {id(cl): i for i, cl in enumerate(agent.pop)}


def get_feature_name_index_pairs(agent: XCSRAgent) -> List[Tuple[str, int | None]]:
    """
    Map requested FEATURE_NAMES to indices in agent.feature_order.

    Returns list of (feature_name, dim_index or None).
    """
    feature_order = list(getattr(agent, "feature_order", []))
    if not feature_order:
        feature_order = [f"f{i}" for i in range(agent.D)]
    name_to_idx = {name: i for i, name in enumerate(feature_order)}

    pairs: List[Tuple[str, int | None]] = []
    missing: List[str] = []
    for name in FEATURE_NAMES:
        idx = name_to_idx.get(name)
        if idx is None:
            missing.append(name)
            pairs.append((name, None))
        else:
            pairs.append((name, idx))

    if missing:
        print(f"[export] Warning: requested features not in agent.feature_order: {missing}")
    return pairs


def export_population_dataframe(
    agent: XCSRAgent,
    feature_pairs: List[Tuple[str, int | None]],
) -> pd.DataFrame:
    """
    Build a DataFrame with one row per classifier, including:

      classifier_index, action, p, eps, F, exp, num, as_size, ts_ga,
      and for each feature in FEATURE_NAMES:

        <feat>_lo, <feat>_hi, <feat>_mask, <feat>_lo_raw, <feat>_hi_raw

    Notes:
      - mask=True  -> DON'T-CARE (wildcard)
      - mask=False -> feature is cared about
    """
    pop = agent.pop
    if not pop:
        raise ValueError("Agent population is empty; nothing to export.")

    scaler = getattr(agent, "scaler", None)
    rows: List[Dict[str, Any]] = []

    for idx, cl in enumerate(pop):
        lower = np.asarray(cl.lower, dtype=float).ravel()
        upper = np.asarray(cl.upper, dtype=float).ravel()
        mask = np.asarray(cl.mask, dtype=bool).ravel()

        D = lower.shape[0]
        if not (upper.shape[0] == D and mask.shape[0] == D):
            raise ValueError(
                f"Classifier {idx} condition shape mismatch: "
                f"lower={lower.shape}, upper={upper.shape}, mask={mask.shape}"
            )

        row: Dict[str, Any] = {
            "classifier_index": idx,
            "action": int(getattr(cl, "action", -1)),
            "p": float(getattr(cl, "p", 0.0)),
            "eps": float(getattr(cl, "eps", 0.0)),
            "F": float(getattr(cl, "F", 0.01)),
            "exp": int(getattr(cl, "exp", 0)),
            "num": int(getattr(cl, "num", 1)),
            "as_size": float(getattr(cl, "as_size", 1.0)),
            "ts_ga": int(getattr(cl, "ts_ga", 0)),
        }

        for feat_name, dim in feature_pairs:
            if dim is None or dim < 0 or dim >= D:
                row[f"{feat_name}_lo"] = np.nan
                row[f"{feat_name}_hi"] = np.nan
                row[f"{feat_name}_mask"] = np.nan
                row[f"{feat_name}_lo_raw"] = np.nan
                row[f"{feat_name}_hi_raw"] = np.nan
                continue

            lo_val = float(lower[dim])
            hi_val = float(upper[dim])
            m_val = bool(mask[dim])

            row[f"{feat_name}_lo"] = lo_val
            row[f"{feat_name}_hi"] = hi_val
            # export as 0/1: 1 = don't-care, 0 = cared
            row[f"{feat_name}_mask"] = int(m_val)

            if scaler is not None:
                try:
                    lo_raw = float(scaler.denorm_i(dim, lo_val))
                    hi_raw = float(scaler.denorm_i(dim, hi_val))
                except Exception:
                    lo_raw, hi_raw = lo_val, hi_val
            else:
                lo_raw, hi_raw = lo_val, hi_val

            row[f"{feat_name}_lo_raw"] = lo_raw
            row[f"{feat_name}_hi_raw"] = hi_raw

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Episode runner with classifier usage + fertilizer-step logging
# ---------------------------------------------------------------------------
def run_frozen_episode(
    agent: XCSRAgent,
    env: WofostXcsEnv,
    wrapper: XcsObsWrapper,
    reward_fn,
    id_to_index: Dict[int, int],
    fert_rows: List[Dict[str, Any]],
    max_steps: int = 500,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    One episode with frozen policy, using select_action_with_trace for usage stats.

    Returns:
      ep_metrics   : dict with episode metrics (reward, NUE, Nsurp, ...)
      match_counts : (n_classifiers,) how many steps each classifier matched
      fire_counts  : (n_classifiers,) how many steps each classifier fired

    Additionally, for each step where fertilizer was actually applied
    (effective_action > 0), appends rows to fert_rows with:

      {
        "year": episode_year,
        "step": step_index,
        "date": info["date"],
        "dose_kg_ha": effective_action,
        "action_index": action_index,
        "blocked": False,
        "classifier_index": <index in agent.pop>,
      }
    """
    freeze_agent(agent)

    obs = wrapper.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    episode_year = int(getattr(env, "episode_year", -1))

    done = False
    steps = 0
    total_reward = 0.0

    prev_obs_dict = None
    n_classifiers = len(id_to_index)
    match_counts = np.zeros(n_classifiers, dtype=np.int64)
    fire_counts = np.zeros(n_classifiers, dtype=np.int64)

    while not done and steps < max_steps:
        steps += 1

        # Use trace-enabled action selection
        a_idx, trace = agent.select_action_with_trace(obs)

        next_obs, _, done, info = wrapper.step(a_idx)
        cur_obs_dict = env._observe()

        # reward as before
        info = dict(info)
        info.setdefault("done", bool(done))
        r = float(reward_fn(prev_obs_dict, cur_obs_dict, info))
        total_reward += r

        # --- classifier usage (all steps) ---
        for rid in getattr(trace, "match_ids", []) or []:
            idx = id_to_index.get(rid)
            if idx is not None:
                match_counts[idx] += 1
        for rid in getattr(trace, "action_ids", []) or []:
            idx = id_to_index.get(rid)
            if idx is not None:
                fire_counts[idx] += 1

        # --- fertilizer-step logging (only when effective_action > 0) ---
        dose = float(info.get("effective_action", info.get("last_dose_kg_ha", 0.0)))
        if dose > 0.0:
            date_val = info.get("date", None)
            blocked = bool(info.get("blocked", False))
            reasons = info.get("blocked_reasons", [])
            if isinstance(reasons, (list, tuple)):
                reasons_str = ";".join(map(str, reasons))
            else:
                reasons_str = str(reasons)

            for rid in getattr(trace, "action_ids", []) or []:
                cls_idx = id_to_index.get(rid)
                if cls_idx is None:
                    continue
                fert_rows.append(
                    {
                        "year": episode_year,
                        "step": steps,
                        "date": date_val,
                        "dose_kg_ha": dose,
                        "action_index": int(a_idx),
                        "blocked": blocked,
                        "blocked_reasons": reasons_str,
                        "classifier_index": int(cls_idx),
                    }
                )

        obs = next_obs
        prev_obs_dict = cur_obs_dict

    metrics = env.episode_metrics()
    ep = dict(
        steps=steps,
        total_reward=float(total_reward),
        NUE_EUNEP=metrics.get("NUE_EUNEP", np.nan),
        Nsurp=metrics.get("Nsurp", np.nan),
        Napplied=metrics.get("N_applied", np.nan),
        Napps=metrics.get("N_apps", np.nan),
        Yield=metrics.get("Yield", np.nan),
        NUE_FERT=metrics.get("NUE_FERT", np.nan),
    )
    return ep, match_counts, fire_counts


# ---------------------------------------------------------------------------
# Main evaluation & exports
# ---------------------------------------------------------------------------
def evaluate_policy(
    agent_path: Path,
    reward_name: str = "nue_paper",
    n_years: int = 34,
    years: List[int] | None = None,
    band_lo: float = 0.5,
    band_hi: float = 1.0,
    nsurp_lo: float = 0.0,
    nsurp_hi: float = 40.0,
    cv_max: float = 0.50,
    band_hit_min: float = 0.50,
    stride_days: int = 7,
    layer_count: int = 7,
) -> Dict[str, Any]:
    """
    Loads agent, runs frozen eval on fixed years, returns summary and saves:

      - eval_results.csv
      - convergence_report.json
      - xcsr_classifiers_global.csv
      - xcsr_classifier_usage_by_year.csv
      - xcsr_classifiers_used_by_year.csv
      - xcsr_fert_steps_rules.csv
      - episodes/xcsr_ep00001_pcse.xlsx, ...   <-- NEW (per-episode PCSE export)
    """
    paths = default_paths()
    out_dir = paths.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # NEW: where to dump per-episode PCSE outputs (same as training)
    episodes_dir = out_dir / "episodes"  # change name if you want per-scenario folders
    episodes_dir.mkdir(parents=True, exist_ok=True)

    # load agent
    if not agent_path.exists():
        raise FileNotFoundError(f"Agent file not found: {agent_path}")
    with open(agent_path, "rb") as f:
        agent: XCSRAgent = pickle.load(f)

    # Stable mapping from classifier object id -> index
    if not agent.pop:
        raise ValueError("Loaded agent has an empty population (agent.pop).")
    id_to_index = make_id_to_index(agent)

    # Resolve which dimensions correspond to our feature names
    feature_pairs = get_feature_name_index_pairs(agent)

    # years to evaluate
    if not years:
        years = default_validation_years(n_years)

    # build env/wrapper
    env = build_env(paths, years=years, stride_days=stride_days, layer_count=layer_count)
    wrapper = XcsObsWrapper(env)
    reward_fn = make_reward(reward_name)

    rows = []
    usage_rows: List[Dict[str, Any]] = []
    fert_rows: List[Dict[str, Any]] = []

    for i, y in enumerate(years):
        ep, match_counts, fire_counts = run_frozen_episode(
            agent, env, wrapper, reward_fn, id_to_index, fert_rows
        )
        ep["episode_idx"] = i + 1
        ep["year"] = y

        # --- NEW: export PCSE output for this episode (same pattern as training) ---
        try:
            df_pcse = pd.DataFrame(env.model.get_output())
            # xcsr_ep00001_pcse.xlsx, xcsr_ep00002_pcse.xlsx, ...
            df_pcse.to_excel(episodes_dir / f"xcsr_ep{(i+1):05d}_pcse.xlsx", index=False)
            # If you prefer to include the year:
            # df_pcse.to_excel(episodes_dir / f"xcsr_ep{(i+1):05d}_y{y}_pcse.xlsx", index=False)
        except Exception as e:
            print(f"[warn] could not export PCSE output for episode {i+1} (year={y}): {e}")

        nue = ep.get("NUE_EUNEP", float("nan"))
        nsurp = ep.get("Nsurp", float("nan"))
        ret = ep.get("total_reward", float("nan"))
        nappl = ep.get("Napplied", float("nan"))
        print(
            f"[eval] year={y} | total_reward={ret:.3f} | NUE_EUNEP={nue:.3f} | "
            f"Nsurp={nsurp:.1f} | N_applied={nappl:.1f} kg/ha"
        )

        rows.append(ep)

        # per-year classifier usage (counts + booleans)
        n_cls = len(agent.pop)
        for cls_idx in range(n_cls):
            m = int(match_counts[cls_idx])
            f_cnt = int(fire_counts[cls_idx])
            if m == 0 and f_cnt == 0:
                continue
            usage_rows.append(
                {
                    "year": y,
                    "classifier_index": cls_idx,
                    "match_count": m,
                    "fire_count": f_cnt,
                    "matched": bool(m > 0),
                    "fired": bool(f_cnt > 0),
                }
            )

    df = pd.DataFrame(rows)

    # core metrics
    mean_ret = float(df["total_reward"].mean())
    std_ret = float(df["total_reward"].std(ddof=1)) if len(df) > 1 else 0.0
    cv_ret = float(std_ret / (abs(mean_ret) + 1e-9))

    band_hit = (
        (df["NUE_EUNEP"].between(band_lo, band_hi, inclusive="both"))
        & (df["Nsurp"].between(nsurp_lo, nsurp_hi, inclusive="both"))
    )
    band_hit_rate = float(band_hit.mean())

    converged = (band_hit_rate >= band_hit_min) and (cv_ret <= cv_max)

    summary = {
        "episodes": int(len(df)),
        "years": years,
        "reward_name": reward_name,
        "mean_total_reward": mean_ret,
        "std_total_reward": std_ret,
        "coef_variation_total_reward": cv_ret,
        "band_lo": band_lo,
        "band_hi": band_hi,
        "nsurp_lo": nsurp_lo,
        "nsurp_hi": nsurp_hi,
        "band_hit_rate": band_hit_rate,
        "Napplied_mean": float(df["Napplied"].mean()),
        "Napplied_std": float(df["Napplied"].std(ddof=1)) if len(df) > 1 else 0.0,
        "Napps_mean": float(df["Napps"].mean()),
        "Napps_std": float(df["Napps"].std(ddof=1)) if len(df) > 1 else 0.0,
        "converged": bool(converged),
        "criteria": {
            "band_hit_min": band_hit_min,
            "cv_max": cv_max,
        },
    }

    # --- Save core eval results ---
    eval_csv = out_dir / "eval_results.csv"
    report_json = out_dir / "convergence_report.json"
    df.to_csv(eval_csv, index=False)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # --- Export classifier population (global) ---
    pop_df = export_population_dataframe(agent, feature_pairs)
    pop_csv = out_dir / "xcsr_classifiers_global.csv"
    pop_df.to_csv(pop_csv, index=False)

    # --- Per-year classifier usage ---
    usage_df = pd.DataFrame(usage_rows)
    usage_csv = out_dir / "xcsr_classifier_usage_by_year.csv"
    usage_df.to_csv(usage_csv, index=False)

    # --- Combined per-year + classifier attributes (for Jupyter) ---
    used_detail = usage_df.merge(pop_df, on="classifier_index", how="left")

    # Column order: EXACTLY what you requested, plus counts at the end
    desired_cols = [
        "year",
        "classifier_index",
        "matched",
        "fired",
        "action",
        "p",
        "eps",
        "F",
        "exp",
        "num",
        "as_size",
        "ts_ga",
        # Feature-wise columns (bounds, mask, raw bounds)
        "DVS_lo", "DVS_hi", "DVS_mask", "DVS_lo_raw", "DVS_hi_raw",
        "LAI_lo", "LAI_hi", "LAI_mask", "LAI_lo_raw", "LAI_hi_raw",
        "RFTRA_lo", "RFTRA_hi", "RFTRA_mask", "RFTRA_lo_raw", "RFTRA_hi_raw",
        "WSO_lo", "WSO_hi", "WSO_mask", "WSO_lo_raw", "WSO_hi_raw",
        "Nsurp_sofar_lo", "Nsurp_sofar_hi", "Nsurp_sofar_mask",
        "Nsurp_sofar_lo_raw", "Nsurp_sofar_hi_raw",
        "last_dose_kg_ha_lo", "last_dose_kg_ha_hi", "last_dose_kg_ha_mask",
        "last_dose_kg_ha_lo_raw", "last_dose_kg_ha_hi_raw",
        "last_app_gap_days_lo", "last_app_gap_days_hi", "last_app_gap_days_mask",
        "last_app_gap_days_lo_raw", "last_app_gap_days_hi_raw",
        "cum_N_applied_kg_ha_lo", "cum_N_applied_kg_ha_hi", "cum_N_applied_kg_ha_mask",
        "cum_N_applied_kg_ha_lo_raw", "cum_N_applied_kg_ha_hi_raw",
        "IRRAD_7_lo", "IRRAD_7_hi", "IRRAD_7_mask", "IRRAD_7_lo_raw", "IRRAD_7_hi_raw",
        "TMEAN_7_lo", "TMEAN_7_hi", "TMEAN_7_mask", "TMEAN_7_lo_raw", "TMEAN_7_hi_raw",
        "RAIN_7_lo", "RAIN_7_hi", "RAIN_7_mask", "RAIN_7_lo_raw", "RAIN_7_hi_raw",
        "Ndep_sum_lo", "Ndep_sum_hi", "Ndep_sum_mask",
        "Ndep_sum_lo_raw", "Ndep_sum_hi_raw",
        "SM_mean_lo", "SM_mean_hi", "SM_mean_mask",
        "SM_mean_lo_raw", "SM_mean_hi_raw",
    ]

    existing_cols = [c for c in desired_cols if c in used_detail.columns]
    extra_cols = [c for c in used_detail.columns if c not in existing_cols]
    used_detail = used_detail[existing_cols + extra_cols]

    used_detail_csv = out_dir / "xcsr_classifiers_used_by_year.csv"
    used_detail.to_csv(used_detail_csv, index=False)

    # --- Fertilizer-step classifier log ---
    if fert_rows:
        fert_df = pd.DataFrame(fert_rows)
        fert_csv = out_dir / "xcsr_fert_steps_rules.csv"
        fert_df.to_csv(fert_csv, index=False)
        print(f"Fertilizer-step classifier log saved to: {fert_csv}")
    else:
        print("No fertilizer applications recorded; nothing to log for classifier usage on fert steps.")

    # Logs
    print(f"Classifier population (global) saved to: {pop_csv}")
    print(f"Classifier usage by year saved to: {usage_csv}")
    print(f"Per-year classifier details saved to: {used_detail_csv}")

    # pretty print summary
    print("\n=== Frozen-policy evaluation ===")
    print(f"Episodes (years): {len(df)}  -> saved: {eval_csv}")
    print(f"Reward: {reward_name}")
    print(f"Mean total_reward: {mean_ret:.4f}  | Std: {std_ret:.4f}  | CV: {cv_ret:.3f}")
    print(
        f"Band-hit rate (NUE∈[{band_lo},{band_hi}] & Nsurp∈[{nsurp_lo},{nsurp_hi}]): "
        f"{band_hit_rate * 100:.1f}%"
    )
    print(
        f"Napplied mean±std: {summary['Napplied_mean']:.1f} ± {summary['Napplied_std']:.1f}  | "
        f"Napps mean±std: {summary['Napps_mean']:.2f} ± {summary['Napps_std']:.2f}"
    )
    print(f"Converged? {converged}  (criteria: band_hit_rate≥{band_hit_min}, CV≤{cv_max})")
    print(f"Report saved to: {report_json}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate frozen XCSR agent for convergence proxies.")
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Path to xcsr_agent.pkl (defaults to outputs/xcsr_agent.pkl).",
    )
    parser.add_argument(
        "--reward",
        type=str,
        default="nue_paper",
        help="Reward name for evaluation (must match training reward).",
    )
    parser.add_argument(
        "--n_years",
        type=int,
        default=32,
        help="Number of validation years (if --years not given).",
    )
    parser.add_argument(
        "--years",
        type=str,
        default="",
        help="Comma-separated list of years to evaluate (overrides --n_years).",
    )
    parser.add_argument("--band_lo", type=float, default=0.5)
    parser.add_argument("--band_hi", type=float, default=1.0)
    parser.add_argument("--nsurp_lo", type=float, default=0.0)
    parser.add_argument("--nsurp_hi", type=float, default=40.0)
    parser.add_argument(
        "--cv_max",
        type=float,
        default=0.50,
        help="Max coefficient of variation for total_reward to call 'converged'.",
    )
    parser.add_argument(
        "--band_hit_min",
        type=float,
        default=0.50,
        help="Min band-hit rate to call 'converged'.",
    )
    parser.add_argument("--stride_days", type=int, default=7)
    parser.add_argument("--layer_count", type=int, default=7)
    args = parser.parse_args()

    paths = default_paths()
    agent_path = Path(args.agent) if args.agent else (paths.output_dir / "xcsr_agent.pkl")

    years: List[int] = []
    if args.years.strip():
        years = [int(s) for s in args.years.split(",") if s.strip()]

    summary = evaluate_policy(
        agent_path=agent_path,
        reward_name=args.reward,
        n_years=args.n_years,
        years=years,
        band_lo=args.band_lo,
        band_hi=args.band_hi,
        nsurp_lo=args.nsurp_lo,
        nsurp_hi=args.nsurp_hi,
        cv_max=args.cv_max,
        band_hit_min=args.band_hit_min,
        stride_days=args.stride_days,
        layer_count=args.layer_count,
    )

    # Exit code 0 if converged else 1 (useful for CI)
    import sys

    sys.exit(0 if summary["converged"] else 1)


if __name__ == "__main__":
    main()
