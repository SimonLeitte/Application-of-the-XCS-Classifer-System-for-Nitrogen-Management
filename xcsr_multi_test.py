# xcsr_test.py — Frozen-policy evaluation & simple convergence proxy (hard-coded root)
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

from xcsr_environment import default_paths, WofostXcsEnv
from xcsr_wrapper import XcsObsWrapper
from xcsr_rewards import make_reward
from xcsr_agent import XCSRAgent


# ========= HARD-CODED LOCATION OF YOUR MODELS =========
# Change this to the parent folder that contains seed_* subfolders.
AGENT_ROOT: Path = Path(
    r"C:\Users\Simon\PycharmProjects\Masterthesis7\data\wofost81\output\multi_seed_parallel_yield"
)

# Pattern to find pkl files under the root. Keep as-is if your layout is ...\seed_XX\xcsr_agent.pkl
AGENT_PATTERN: str = "seed_*/xcsr_agent.pkl"

# (Optional) restrict to specific seed numbers. Example: {31, 32, 33}. Leave as None to include all.
ALLOWED_SEEDS: Optional[set[int]] = { 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100}
# =======================================================


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
    """Build WofostXcsEnv that will iterate deterministically over `years`."""
    try:
        env = WofostXcsEnv(paths, mode="test", layer_count=layer_count,
                           train_years=years, stride_days=stride_days)
    except TypeError:
        env = WofostXcsEnv(paths, mode="test", layer_count=layer_count, train_years=years)
        print("[eval] Note: your environment does not accept stride_days; using its built-in stride.")
    return env


def run_frozen_episode(
    agent: XCSRAgent,
    env: WofostXcsEnv,
    wrapper: XcsObsWrapper,
    reward_fn,
    max_steps: int = 500,
) -> Dict[str, Any]:
    """Run one episode with a frozen policy; compute shaped reward like in training."""
    freeze_agent(agent)

    obs = wrapper.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    done = False
    steps = 0
    total_reward = 0.0
    prev_obs_dict = None

    while not done and steps < max_steps:
        steps += 1
        a_idx = agent.select_action(obs)
        next_obs, _, done, info = wrapper.step(a_idx)
        cur_obs_dict = env._observe()

        info = dict(info)
        info.setdefault("done", bool(done))

        r = float(reward_fn(prev_obs_dict, cur_obs_dict, info))
        total_reward += r

        obs = next_obs
        prev_obs_dict = cur_obs_dict

    metrics = env.episode_metrics()
    out = dict(
        steps=steps,
        total_reward=float(total_reward),
        NUE_EUNEP=metrics.get("NUE_EUNEP", np.nan),
        Nsurp=metrics.get("Nsurp", np.nan),
        Napplied=metrics.get("N_applied", np.nan),
        Napps=metrics.get("N_apps", np.nan),
        Yield=metrics.get("Yield", np.nan),
        NUE_FERT=metrics.get("NUE_FERT", np.nan),
    )
    return out


def infer_seed_tag(agent_path: Path) -> str:
    """
    Try to infer a label like 'seed_31' from the agent path.
    Falls back to the parent directory name if needed.
    """
    for p in agent_path.parents:
        name = p.name
        if name.lower().startswith("seed_"):
            return name
    return agent_path.parent.name or "model"


def parse_seed_num(agent_path: Path) -> Optional[int]:
    """Extract the integer part from a folder name like 'seed_31' -> 31."""
    name = agent_path.parent.name
    try:
        return int(name.split("_")[-1])
    except Exception:
        return None


def save_artifacts_per_seed(
    df: pd.DataFrame,
    summary: Dict[str, Any],
    seed_dir: Path,
    seed_tag: str,
) -> Tuple[Path, Path, Path]:
    """
    Save episode-level results (Excel + CSV) and a JSON summary into the seed directory.
    Filenames are tied to the seed tag to avoid mix-ups between models.
    """
    seed_dir.mkdir(parents=True, exist_ok=True)

    excel_path = seed_dir / f"eval_{seed_tag}.xlsx"              # e.g., eval_seed_31.xlsx
    csv_path = seed_dir / f"eval_results_{seed_tag}.csv"         # e.g., eval_results_seed_31.csv
    json_path = seed_dir / f"convergence_report_{seed_tag}.json" # e.g., convergence_report_seed_31.json

    # 1) Excel (episodes + summary as a second sheet)
    try:
        with pd.ExcelWriter(excel_path) as xw:
            df.to_excel(xw, index=False, sheet_name="episodes")
            s = pd.json_normalize(summary)
            s.to_excel(xw, index=False, sheet_name="summary")
    except Exception as e:
        print(f"[warn] Could not write Excel at {excel_path} ({e}). "
              f"CSV + JSON still saved. If needed, install 'openpyxl'.")

    # 2) CSV (episodes)
    df.to_csv(csv_path, index=False)

    # 3) JSON summary
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return excel_path, csv_path, json_path


def evaluate_policy(
    agent_path: Path,
    reward_name: str = "nue_paper",
    n_years: int = 32,
    years: List[int] | None = None,
    band_lo: float = 0.5,
    band_hi: float = 1.0,
    nsurp_lo: float = 0.0,
    nsurp_hi: float = 40.0,
    cv_max: float = 0.50,
    band_hit_min: float = 0.50,
    stride_days: int = 7,
    layer_count: int = 7,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Loads agent, runs frozen eval on fixed years, returns (summary, dataframe).
    Artifacts are saved by the caller (per-seed).
    """
    paths = default_paths()

    if not agent_path.exists():
        raise FileNotFoundError(f"Agent file not found: {agent_path}")
    with open(agent_path, "rb") as f:
        agent: XCSRAgent = pickle.load(f)

    if not years:
        years = default_validation_years(n_years)

    env = build_env(paths, years=years, stride_days=stride_days, layer_count=layer_count)
    wrapper = XcsObsWrapper(env)
    reward_fn = make_reward(reward_name)

    rows = []
    for i, y in enumerate(years):
        ep = run_frozen_episode(agent, env, wrapper, reward_fn)
        ep["episode_idx"] = i + 1
        ep["year"] = y

        nue = ep.get("NUE_EUNEP", float("nan"))
        nsurp = ep.get("Nsurp", float("nan"))
        ret = ep.get("total_reward", float("nan"))
        nappl = ep.get("Napplied", float("nan"))
        print(f"[eval] year={y} | total_reward={ret:.3f} | NUE_EUNEP={nue:.3f} | "
              f"Nsurp={nsurp:.1f} | N_applied={nappl:.1f} kg/ha")

        rows.append(ep)

    df = pd.DataFrame(rows)

    mean_ret = float(df["total_reward"].mean())
    std_ret = float(df["total_reward"].std(ddof=1)) if len(df) > 1 else 0.0
    cv_ret = float(std_ret / (abs(mean_ret) + 1e-9))

    band_hit = (
        (df["NUE_EUNEP"].between(band_lo, band_hi, inclusive="both")) &
        (df["Nsurp"].between(nsurp_lo, nsurp_hi, inclusive="both"))
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
        "criteria": {"band_hit_min": band_hit_min, "cv_max": cv_max},
    }

    print("\n=== Frozen-policy evaluation ===")
    print(f"Episodes (years): {len(df)}")
    print(f"Reward: {reward_name}")
    print(f"Mean total_reward: {mean_ret:.4f}  | Std: {std_ret:.4f}  | CV: {cv_ret:.3f}")
    print(f"Band-hit rate (NUE∈[{band_lo},{band_hi}] & Nsurp∈[{nsurp_lo},{nsurp_hi}]): {band_hit_rate*100:.1f}%")
    print(f"Napplied mean±std: {summary['Napplied_mean']:.1f} ± {summary['Napplied_std']:.1f}  | "
          f"Napps mean±std: {summary['Napps_mean']:.2f} ± {summary['Napps_std']:.2f}")
    print(f"Converged? {converged}  (criteria: band_hit_rate≥{band_hit_min}, CV≤{cv_max})\n")

    return summary, df


def discover_hardcoded_agents() -> List[Path]:
    """Find all agents under the hard-coded root, optionally filtering by seed numbers."""
    if not AGENT_ROOT.exists():
        raise FileNotFoundError(f"Hard-coded root does not exist: {AGENT_ROOT}")
    candidates = sorted((AGENT_ROOT.glob(AGENT_PATTERN)))
    if ALLOWED_SEEDS is not None:
        filtered = []
        for p in candidates:
            sn = parse_seed_num(p)
            if sn is None:
                continue
            if sn in ALLOWED_SEEDS:
                filtered.append(p)
        candidates = filtered
    if not candidates:
        raise FileNotFoundError(
            f"No xcsr_agent.pkl found under {AGENT_ROOT} (pattern: {AGENT_PATTERN}). "
            f"Check the path or relax ALLOWED_SEEDS."
        )
    return [c.resolve() for c in candidates]


def batch_evaluate(
    agent_paths: List[Path],
    reward_name: str = "nue_paper",
    n_years: int = 32,
    years: List[int] | None = None,
    band_lo: float = 0.5,
    band_hi: float = 1.0,
    nsurp_lo: float = 0.0,
    nsurp_hi: float = 40.0,
    cv_max: float = 0.50,
    band_hit_min: float = 0.50,
    stride_days: int = 7,
    layer_count: int = 7,
) -> bool:
    """Evaluate multiple agents and save per-seed artifacts in each seed folder."""
    all_ok = True
    for ap in agent_paths:
        ap = ap.resolve()
        print("=" * 80)
        print(f"[batch] Evaluating agent: {ap}")

        seed_tag = infer_seed_tag(ap)
        seed_dir = ap.parent

        summary, df = evaluate_policy(
            agent_path=ap,
            reward_name=reward_name,
            n_years=n_years,
            years=years,
            band_lo=band_lo,
            band_hi=band_hi,
            nsurp_lo=nsurp_lo,
            nsurp_hi=nsurp_hi,
            cv_max=cv_max,
            band_hit_min=band_hit_min,
            stride_days=stride_days,
            layer_count=layer_count,
        )

        excel_path, csv_path, json_path = save_artifacts_per_seed(df, summary, seed_dir, seed_tag)

        print(f"[batch] Saved results for {seed_tag}:")
        print(f"    Excel: {excel_path}")
        print(f"    CSV:   {csv_path}")
        print(f"    JSON:  {json_path}\n")

        if not summary.get("converged", False):
            all_ok = False

    return all_ok


def main():
    # Keep the knobs for evaluation (reward, years, etc.) but ignore any CLI for agent paths.
    parser = argparse.ArgumentParser(description="Evaluate frozen XCSR agent(s) from a hard-coded root path.")
    parser.add_argument("--reward", type=str, default="nue_paper")
    parser.add_argument("--n_years", type=int, default=32)
    parser.add_argument("--years", type=str, default="", help="Comma-separated years to evaluate (overrides --n_years).")
    parser.add_argument("--band_lo", type=float, default=0.5)
    parser.add_argument("--band_hi", type=float, default=1.0)
    parser.add_argument("--nsurp_lo", type=float, default=0.0)
    parser.add_argument("--nsurp_hi", type=float, default=40.0)
    parser.add_argument("--cv_max", type=float, default=0.50)
    parser.add_argument("--band_hit_min", type=float, default=0.50)
    parser.add_argument("--stride_days", type=int, default=7)
    parser.add_argument("--layer_count", type=int, default=7)
    args = parser.parse_args()

    # Discover agents strictly from the hard-coded root:
    agent_paths = discover_hardcoded_agents()
    print("[batch] Discovered agents:")
    for p in agent_paths:
        print(f"  - {p}")

    years: List[int] = []
    if args.years.strip():
        years = [int(s) for s in args.years.split(",") if s.strip()]

    all_ok = batch_evaluate(
        agent_paths=agent_paths,
        reward_name=args.reward,
        n_years=args.n_years,
        years=years if years else None,
        band_lo=args.band_lo,
        band_hi=args.band_hi,
        nsurp_lo=args.nsurp_lo,
        nsurp_hi=args.nsurp_hi,
        cv_max=args.cv_max,
        band_hit_min=args.band_hit_min,
        stride_days=args.stride_days,
        layer_count=args.layer_count,
    )

    # Exit code: success only if all models converged
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
