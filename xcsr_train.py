# xcsr_train.py — training loop for XCSR agent (reward-agnostic)
from __future__ import annotations
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

from xcsr_environment import default_paths, WofostXcsEnv
from xcsr_wrapper import XcsObsWrapper, feature_names, feature_dim, FEATURE_ORDER
from xcsr_rewards import make_reward
from xcsr_agent import XCSRAgent, DEFAULT_XCSR_CONFIG
from xcsr_scaler import make_scaler_for_features, scaler_summary
from xcsr_logs import RewardCSVLogger  # keep; it will just have empty terminal columns

def run_training(
    episodes: int = 400,
    mode: str = "train",
    seed: int = 42,
    reward_name: str = "yield",
    resume: bool = False,
    reward_logging: bool = False,   # keep the on/off switch
):
    paths = default_paths()
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    # folders
    episodes_dir = paths.output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    agent_log_dir = paths.output_dir / "agent_logs"
    agent_log_dir.mkdir(parents=True, exist_ok=True)

    # files
    model_file = paths.output_dir / "xcsr_agent.pkl"
    log_file = paths.output_dir / "episode_log.csv"

    # reward csv logger (optional)
    reward_logger = None
    reward_log_path = paths.output_dir / "reward_log.csv"
    if reward_logging:
        reward_logger = RewardCSVLogger(reward_log_path, FEATURE_ORDER)

    def log_step_safe(**kwargs):
        if reward_logger is not None:
            reward_logger.log_step(**kwargs)

    def log_terminal_safe(**kwargs):
        if reward_logger is not None:
            reward_logger.log_terminal(**kwargs)

    # reset episode log unless resuming
    if log_file.exists() and not resume:
        log_file.unlink()
    first_write = True

    # resume offset
    start_offset = 0
    if resume and log_file.exists():
        try:
            last_row = pd.read_csv(log_file).tail(1)
            if not last_row.empty and "episode" in last_row.columns:
                start_offset = int(last_row["episode"].values[0])
        except Exception as e:
            print(f"[warn] Could not read existing log for offset ({e}). Starting offset=0.")
            start_offset = 0

    # env/wrapper
    env = WofostXcsEnv(paths, mode=mode, layer_count=7,
                       seed=seed, #fixed_year=3002,
                       stride_days=7)  # daily stride to match scaler domain

    wrapper = XcsObsWrapper(env)
    feat_dim = feature_dim()
    feats = feature_names()
    print(f"[schema] Using {feat_dim} features:", ", ".join(feats))

    scaler = make_scaler_for_features(feats)
    print("[scaler]\n" + scaler_summary(scaler, feats))

    # agent
    if resume and model_file.exists():
        print(f"[resume] Loading XCSR agent from: {model_file}")
        with open(model_file, "rb") as f:
            agent: XCSRAgent = pickle.load(f)
        print(f"[resume] Continuing episodes from offset={start_offset}. "
              f"This run will log episodes {start_offset+1}..{start_offset+episodes}")
    else:
        print("[init] Starting NEW XCSR agent with DEFAULT_XCSR_CONFIG")
        agent = XCSRAgent(
            n_features=feat_dim,
            n_actions=len(env.ACTION_SET),
            seed=seed,
            config=DEFAULT_XCSR_CONFIG,
            scaler=scaler,
            feature_order=list(FEATURE_ORDER),
            log_dir=agent_log_dir,
        )

    # reward fn (name preserved)
    reward_fn = make_reward(reward_name)

    # episode-wide rolling df
    chunk_rows = []
    np.random.seed(seed)
    #prev_obs_dict = None

    for ep_local in range(1, episodes + 1):
        ep_global = start_offset + ep_local
        agent._episode_idx = ep_global  # for agent telemetry

        obs = wrapper.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        assert obs.shape == (feat_dim,), f"First observation has shape {obs.shape}, expected {(feat_dim,)}"
        prev_obs_dict = None

        done = False
        total_reward = 0.0
        steps = 0

        # cumulative return
        G = 0.0
        last_step_ctx = None

        # optional population snapshot
        if getattr(agent, "logger", None) is not None:
            try:
                pop_ser = [agent.logger.serialize_classifier(cl, cid=id(cl)) for cl in agent.pop]
                agent.logger.log_population_snapshot(episode=ep_global, when="before_episode", population=pop_ser)
            except Exception as e:
                print(f"[warn] pop snapshot(before) failed: {e}")

        while not done and steps < 500:
            steps += 1
            agent._step_idx = steps

            # --- select action on current obs
            a_idx = agent.select_action(obs)

            # --- step env
            next_obs, _, done, info = wrapper.step(a_idx)

            # --- build current raw dict AFTER the step
            cur_obs_dict = env._observe()

            # --- make sure reward sees terminal
            info = dict(info)
            info.setdefault("done", bool(done))

            # --- compute reward with (prev -> cur)
            r = float(reward_fn(prev_obs_dict, cur_obs_dict, info))

            # --- learning
            agent.learn(obs, a_idx, r, next_obs, done)

            # --- logging dicts (unchanged)
            try:
                raw_vec = np.array([cur_obs_dict[k] for k in FEATURE_ORDER], dtype=float)
            except Exception:
                raw_vec = np.full(len(FEATURE_ORDER), np.nan, dtype=float)
            try:
                scaled_vec = agent.scaler.fwd_vec(raw_vec) if agent.scaler is not None else raw_vec
                scaled_obs = {k: float(scaled_vec[i]) for i, k in enumerate(FEATURE_ORDER)}
            except Exception:
                scaled_obs = {k: "" for k in FEATURE_ORDER}
            raw_obs = {k: float(cur_obs_dict.get(k)) if k in cur_obs_dict and cur_obs_dict.get(k) is not None else ""
                       for k in FEATURE_ORDER}

            blocked = bool(info.get("blocked", False))
            # unify reason key(s)
            blocked_reason = info.get("blocked_reasons", info.get("blocked_reason", ""))
            action_effective = float(info.get("effective_action", info.get("dose_effective", 0.0)))
            exploration_mode = getattr(agent, "_last_exploration", "policy")
            eps_val = float(getattr(agent, "eps", 0.0))

            log_step_safe(
                run_id=str(paths.output_dir.name),
                reward_name=reward_name,
                episode=ep_global,
                step=steps,
                exploration_mode=exploration_mode,
                epsilon=eps_val,
                action_id=int(a_idx),
                action_effective=action_effective,
                blocked=blocked,
                blocked_reason=blocked_reason,
                raw_obs=raw_obs,
                scaled_obs=scaled_obs,
                step_breakdown=r,
                G_cumulative_after_step=float(G + r),
            )

            G += r
            total_reward += r

            last_step_ctx = dict(
                step_index=steps,
                exploration_mode=exploration_mode,
                epsilon=eps_val,
                action_id=int(a_idx),
                action_effective=action_effective,
                blocked=blocked,
                blocked_reason=blocked_reason,
                raw_obs=raw_obs,
                scaled_obs=scaled_obs,
            )

            # --- advance loop state
            obs = next_obs
            prev_obs_dict = cur_obs_dict  # <<< CRUCIAL: carry current state to next step

        agent.end_of_episode_backup(total_reward)
        agent.end_episode()

        # env metrics for episode log (unchanged)
        metrics = env.episode_metrics()
        nue_eunep = metrics.get("NUE_EUNEP", float("nan"))
        nsurp = metrics.get("Nsurp", float("nan"))
        n_apps = metrics.get("N_apps", getattr(env, "n_actions", np.nan))
        n_applied_kg_ha = metrics.get("N_applied", np.nan)
        nue_fert = metrics.get("NUE_FERT", np.nan)
        nsurp_fert = metrics.get("Nsurp_FERT", np.nan)
        yield_kg_ha = metrics.get("Yield", np.nan)

        # PCSE export (optional, currently disabled)
        try:
             df_pcse = pd.DataFrame(env.model.get_output())
             df_pcse.to_excel(episodes_dir / f"xcsr_ep{ep_global:05d}_pcse.xlsx", index=False)
        except Exception as e:
             print(f"[warn] could not export PCSE output for episode {ep_global}: {e}")

        # OPTIONAL: emit a terminal CSV line with empty breakdown (safe, generic)
        if last_step_ctx is not None:
            log_terminal_safe(
                run_id=str(paths.output_dir.name),
                reward_name=reward_name,
                episode=ep_global,
                step=int(last_step_ctx["step_index"]),
                exploration_mode=str(last_step_ctx["exploration_mode"]),
                epsilon=float(last_step_ctx["epsilon"]),
                action_id=int(last_step_ctx["action_id"]),
                action_effective=float(last_step_ctx["action_effective"]),
                blocked=bool(last_step_ctx["blocked"]),
                blocked_reason=str(last_step_ctx["blocked_reason"]),
                raw_obs=last_step_ctx["raw_obs"],
                scaled_obs=last_step_ctx["scaled_obs"],
                terminal_breakdown=0.0,  # no structured breakdown for now
                G_cumulative_after_step=float(G),
            )

        # optional population snapshot after episode
        if getattr(agent, "logger", None) is not None:
            try:
                pop_ser = [agent.logger.serialize_classifier(cl, cid=id(cl)) for cl in agent.pop]
                agent.logger.log_population_snapshot(episode=ep_global, when="after_episode", population=pop_ser)
            except Exception as e:
                print(f"[warn] pop snapshot(after) failed: {e}")

        # console summary
        nue_eunep_str = "nan" if pd.isna(nue_eunep) else f"{nue_eunep:.2f}"
        nsurp_str = "nan" if pd.isna(nsurp) else f"{nsurp:.2f}"
        yield_str = "nan" if pd.isna(yield_kg_ha) else f"{yield_kg_ha:.0f}"
        napplied_str = "nan" if pd.isna(n_applied_kg_ha) else f"{float(n_applied_kg_ha):.0f}"
        print(
                        f"Episode {ep_global} (this run {ep_local}/{episodes}) | "
             f"steps={steps} | total_reward={total_reward:.4f} | "
             f"eps={agent.eps:.3f} | NUE_EUNEP={nue_eunep_str} "
             f"| N_applied={napplied_str} kg/ha | Nsurp={nsurp_str} kg/ha "
             f"| Napps={int(n_apps) if not pd.isna(n_apps) else 'nan'} "
             f"| year={getattr(env, 'episode_year', 'na')} "
             f"| Yield={yield_str} kg/ha"
            )

        # episode CSV
        chunk_rows.append(dict(
            episode=ep_global,
            steps=steps,
            total_reward=total_reward,
            eps=agent.eps,
            yield_kg_ha=metrics.get("Yield", np.nan),
            n_applied_kg_ha=metrics.get("N_applied", np.nan),
            n_uptake_kg_ha=metrics.get("N_uptake", np.nan),
            nue_re=metrics.get("NUE_RE", np.nan),
            nue_ie=metrics.get("NUE_IE", np.nan),
            nue_eunep=nue_eunep,
            nue_fert=nue_fert,
            nsurp=metrics.get("Nsurp", np.nan),
            nsurp_fert=nsurp_fert,
            n_apps=int(n_apps) if not pd.isna(n_apps) else np.nan,
        ))

        # PCSE export (optional, currently disabled)
        try:
            df_pcse = pd.DataFrame(env.model.get_output())
            df_pcse.to_excel(episodes_dir / f"xcsr_ep{ep_global:05d}_pcse.xlsx", index=False)
        except Exception as e:
            print(f"[warn] could not export PCSE output for episode {ep_global}: {e}")

        # periodic write + ckpt
        log_mode = "a"
        if (ep_local % 20 == 0) or (ep_local == episodes):
            df_chunk = pd.DataFrame(chunk_rows)
            write_header = (log_mode == "w") or (not log_file.exists())
            df_chunk.to_csv(log_file, mode=log_mode, header=first_write, index=False)
            print(f"[log] wrote {len(df_chunk)} rows to: {log_file} (mode={log_mode}, header={write_header})")
            chunk_rows.clear()
            first_write = False
            try:
                with open(model_file, "wb") as f:
                    pickle.dump(agent, f)
                print(f"[ckpt] saved agent to {model_file}")
            except Exception as e:
                print(f"[warn] could not save checkpoint: {e}")

    # final checkpoint
    try:
        with open(model_file, "wb") as f:
            pickle.dump(agent, f)
        print(f"Saved trained XCSRAgent to: {model_file}")
    except Exception as e:
        print(f"[warn] could not save final agent: {e}")


if __name__ == "__main__":
    run_training(
        episodes=400,
        mode="train",
        seed=42,
        reward_name="yield",   # name is preserved (now a stub)
        resume=False,
        reward_logging=False,
    )
