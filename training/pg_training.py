import os, sys, csv, time
import numpy as np
 
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
 
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
 
from environment.custom_env import TeacherRetentionEnv
 
# ── Output dirs ───────────────────────────────────────────────────────────
os.makedirs(os.path.join(ROOT, "results"),       exist_ok=True)
os.makedirs(os.path.join(ROOT, "models", "pg"),  exist_ok=True)
 
MODELS_DIR = os.path.join(ROOT, "models", "pg")
 
# ==========================================================================
# SECTION 1 — REINFORCE (custom PyTorch implementation)
# ==========================================================================
# SB3 does not include a REINFORCE algorithm, so we implement it from scratch.
# REINFORCE = Monte Carlo Policy Gradient (Williams 1992):
#   θ ← θ + α · ∑_t G_t · ∇log π(a_t|s_t)
# where G_t = discounted return from time t.
 
class PolicyNetwork(nn.Module):
    """Simple MLP policy: state → action logits."""
 
    def __init__(self, obs_dim: int = 10, n_actions: int = 5,
                 hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # raw logits
 
    def get_action(self, state: np.ndarray):
        """Sample an action and return (action, log_prob)."""
        x      = torch.FloatTensor(state).unsqueeze(0)
        logits = self.forward(x)
        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
 
 
def compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """Compute discounted returns G_t for each timestep."""
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns_t = torch.FloatTensor(returns)
    # Normalise for training stability
    if returns_t.std() > 1e-8:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
    return returns_t
 
 
def train_reinforce(cfg: dict) -> tuple[float, float, list[float]]:
    """
    Train one REINFORCE run.
    Returns (mean_reward, std_reward, episode_rewards_list).
    """
    env    = TeacherRetentionEnv(difficulty="medium")
    policy = PolicyNetwork(obs_dim=10, n_actions=5,
                           hidden_size=cfg["hidden_size"])
    optimizer = optim.Adam(policy.parameters(), lr=cfg["learning_rate"])
 
    episode_rewards: list[float] = []
    torch.manual_seed(cfg["run_id"] * 17)
    np.random.seed(cfg["run_id"] * 17)
 
    for ep in range(cfg["n_episodes"]):
        obs, _ = env.reset()
        log_probs: list[torch.Tensor] = []
        rewards:   list[float]        = []
 
        done = False
        while not done:
            action, log_prob = policy.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            done = terminated or truncated
 
        episode_rewards.append(sum(rewards))
 
        # ── Policy gradient update ────────────────────────────────────────
        returns   = compute_returns(rewards, cfg["gamma"])
        log_probs_t = torch.stack(log_probs)
 
        # Entropy bonus to encourage exploration
        logits = policy.net(torch.FloatTensor(obs).unsqueeze(0))
        dist   = torch.distributions.Categorical(logits=logits)
        entropy_bonus = cfg["entropy_coef"] * dist.entropy().mean()
 
        loss = -(log_probs_t * returns).sum() - entropy_bonus
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"])
        optimizer.step()
 
    env.close()
 
    # Evaluate (deterministic greedy)
    eval_rewards = []
    eval_env = TeacherRetentionEnv(difficulty="medium")
    for _ in range(10):
        obs, _ = eval_env.reset()
        total, done = 0.0, False
        while not done:
            with torch.no_grad():
                x      = torch.FloatTensor(obs).unsqueeze(0)
                logits = policy.net(x)
                action = logits.argmax(dim=-1).item()
            obs, r, terminated, truncated, _ = eval_env.step(action)
            total += r
            done = terminated or truncated
        eval_rewards.append(total)
    eval_env.close()
 
    return (round(float(np.mean(eval_rewards)), 3),
            round(float(np.std(eval_rewards)),  3),
            episode_rewards,
            policy)
 
 
REINFORCE_GRID = [
    # Run  learning_rate  gamma   n_episodes  hidden_size  entropy_coef  max_grad_norm
    {"run_id": 1,  "learning_rate": 1e-3,  "gamma": 0.95, "n_episodes": 500,  "hidden_size": 64,  "entropy_coef": 0.01, "max_grad_norm": 0.5,  "note": "Baseline"},
    {"run_id": 2,  "learning_rate": 5e-3,  "gamma": 0.95, "n_episodes": 500,  "hidden_size": 64,  "entropy_coef": 0.01, "max_grad_norm": 0.5,  "note": "High LR"},
    {"run_id": 3,  "learning_rate": 1e-4,  "gamma": 0.99, "n_episodes": 500,  "hidden_size": 64,  "entropy_coef": 0.01, "max_grad_norm": 0.5,  "note": "Low LR"},
    {"run_id": 4,  "learning_rate": 1e-3,  "gamma": 0.999,"n_episodes": 500,  "hidden_size": 64,  "entropy_coef": 0.01, "max_grad_norm": 0.5,  "note": "High gamma"},
    {"run_id": 5,  "learning_rate": 1e-3,  "gamma": 0.90, "n_episodes": 500,  "hidden_size": 64,  "entropy_coef": 0.01, "max_grad_norm": 0.5,  "note": "Low gamma"},
    {"run_id": 6,  "learning_rate": 1e-3,  "gamma": 0.95, "n_episodes": 500,  "hidden_size": 128, "entropy_coef": 0.01, "max_grad_norm": 0.5,  "note": "Larger network"},
    {"run_id": 7,  "learning_rate": 1e-3,  "gamma": 0.95, "n_episodes": 500,  "hidden_size": 32,  "entropy_coef": 0.01, "max_grad_norm": 0.5,  "note": "Smaller network"},
    {"run_id": 8,  "learning_rate": 1e-3,  "gamma": 0.95, "n_episodes": 500,  "hidden_size": 64,  "entropy_coef": 0.05, "max_grad_norm": 0.5,  "note": "High entropy"},
    {"run_id": 9,  "learning_rate": 1e-3,  "gamma": 0.95, "n_episodes": 500,  "hidden_size": 64,  "entropy_coef": 0.001,"max_grad_norm": 0.5,  "note": "Low entropy"},
    {"run_id": 10, "learning_rate": 2e-3,  "gamma": 0.99, "n_episodes": 600,  "hidden_size": 128, "entropy_coef": 0.02, "max_grad_norm": 1.0,  "note": "Tuned best"},
]
 
# ==========================================================================
# SECTION 2 — PPO & A2C grids
# ==========================================================================
 
PPO_GRID = [
    {"run_id": 1,  "note": "Baseline",              "learning_rate": 3e-4,  "n_steps": 128,  "batch_size": 64,  "n_epochs": 10, "gamma": 0.99,  "gae_lambda": 0.95, "clip_range": 0.2,  "ent_coef": 0.01, "vf_coef": 0.5,  "total_timesteps": 40_000},
    {"run_id": 2,  "note": "High LR",               "learning_rate": 1e-3,  "n_steps": 128,  "batch_size": 64,  "n_epochs": 10, "gamma": 0.99,  "gae_lambda": 0.95, "clip_range": 0.2,  "ent_coef": 0.01, "vf_coef": 0.5,  "total_timesteps": 40_000},
    {"run_id": 3,  "note": "Low LR",                "learning_rate": 1e-4,  "n_steps": 128,  "batch_size": 64,  "n_epochs": 10, "gamma": 0.99,  "gae_lambda": 0.95, "clip_range": 0.2,  "ent_coef": 0.01, "vf_coef": 0.5,  "total_timesteps": 40_000},
    {"run_id": 4,  "note": "Wide clip",             "learning_rate": 3e-4,  "n_steps": 256,  "batch_size": 64,  "n_epochs": 10, "gamma": 0.99,  "gae_lambda": 0.95, "clip_range": 0.3,  "ent_coef": 0.01, "vf_coef": 0.5,  "total_timesteps": 40_000},
    {"run_id": 5,  "note": "Narrow clip",           "learning_rate": 3e-4,  "n_steps": 128,  "batch_size": 64,  "n_epochs": 10, "gamma": 0.99,  "gae_lambda": 0.95, "clip_range": 0.1,  "ent_coef": 0.01, "vf_coef": 0.5,  "total_timesteps": 40_000},
    {"run_id": 6,  "note": "More epochs",           "learning_rate": 3e-4,  "n_steps": 128,  "batch_size": 64,  "n_epochs": 20, "gamma": 0.99,  "gae_lambda": 0.95, "clip_range": 0.2,  "ent_coef": 0.01, "vf_coef": 0.5,  "total_timesteps": 40_000},
    {"run_id": 7,  "note": "High entropy coef",     "learning_rate": 3e-4,  "n_steps": 128,  "batch_size": 64,  "n_epochs": 10, "gamma": 0.99,  "gae_lambda": 0.95, "clip_range": 0.2,  "ent_coef": 0.05, "vf_coef": 0.5,  "total_timesteps": 40_000},
    {"run_id": 8,  "note": "High vf_coef",          "learning_rate": 3e-4,  "n_steps": 128,  "batch_size": 64,  "n_epochs": 10, "gamma": 0.99,  "gae_lambda": 0.95, "clip_range": 0.2,  "ent_coef": 0.01, "vf_coef": 1.0,  "total_timesteps": 40_000},
    {"run_id": 9,  "note": "Long horizon (gae=0.99)","learning_rate": 3e-4, "n_steps": 256,  "batch_size": 128, "n_epochs": 10, "gamma": 0.999, "gae_lambda": 0.99, "clip_range": 0.2,  "ent_coef": 0.01, "vf_coef": 0.5,  "total_timesteps": 40_000},
    {"run_id": 10, "note": "Tuned best",            "learning_rate": 2e-4,  "n_steps": 256,  "batch_size": 64,  "n_epochs": 15, "gamma": 0.995, "gae_lambda": 0.97, "clip_range": 0.2,  "ent_coef": 0.02, "vf_coef": 0.5,  "total_timesteps": 40_000},
]
 
A2C_GRID = [
    {"run_id": 1,  "note": "Baseline",          "learning_rate": 7e-4,  "n_steps": 5,   "gamma": 0.99,  "gae_lambda": 1.0,  "ent_coef": 0.00, "vf_coef": 0.5,  "max_grad_norm": 0.5, "total_timesteps": 40_000},
    {"run_id": 2,  "note": "High LR",           "learning_rate": 2e-3,  "n_steps": 5,   "gamma": 0.99,  "gae_lambda": 1.0,  "ent_coef": 0.00, "vf_coef": 0.5,  "max_grad_norm": 0.5, "total_timesteps": 40_000},
    {"run_id": 3,  "note": "Low LR",            "learning_rate": 1e-4,  "n_steps": 5,   "gamma": 0.99,  "gae_lambda": 1.0,  "ent_coef": 0.00, "vf_coef": 0.5,  "max_grad_norm": 0.5, "total_timesteps": 40_000},
    {"run_id": 4,  "note": "More steps",        "learning_rate": 7e-4,  "n_steps": 20,  "gamma": 0.99,  "gae_lambda": 1.0,  "ent_coef": 0.00, "vf_coef": 0.5,  "max_grad_norm": 0.5, "total_timesteps": 40_000},
    {"run_id": 5,  "note": "Many steps",        "learning_rate": 7e-4,  "n_steps": 50,  "gamma": 0.99,  "gae_lambda": 1.0,  "ent_coef": 0.00, "vf_coef": 0.5,  "max_grad_norm": 0.5, "total_timesteps": 40_000},
    {"run_id": 6,  "note": "High gamma",        "learning_rate": 7e-4,  "n_steps": 5,   "gamma": 0.999, "gae_lambda": 1.0,  "ent_coef": 0.00, "vf_coef": 0.5,  "max_grad_norm": 0.5, "total_timesteps": 40_000},
    {"run_id": 7,  "note": "With entropy",      "learning_rate": 7e-4,  "n_steps": 5,   "gamma": 0.99,  "gae_lambda": 1.0,  "ent_coef": 0.05, "vf_coef": 0.5,  "max_grad_norm": 0.5, "total_timesteps": 40_000},
    {"run_id": 8,  "note": "High vf_coef",      "learning_rate": 7e-4,  "n_steps": 5,   "gamma": 0.99,  "gae_lambda": 1.0,  "ent_coef": 0.00, "vf_coef": 1.0,  "max_grad_norm": 0.5, "total_timesteps": 40_000},
    {"run_id": 9,  "note": "GAE lambda 0.9",    "learning_rate": 7e-4,  "n_steps": 10,  "gamma": 0.99,  "gae_lambda": 0.90, "ent_coef": 0.01, "vf_coef": 0.5,  "max_grad_norm": 1.0, "total_timesteps": 40_000},
    {"run_id": 10, "note": "Tuned best",        "learning_rate": 5e-4,  "n_steps": 20,  "gamma": 0.995, "gae_lambda": 0.95, "ent_coef": 0.02, "vf_coef": 0.5,  "max_grad_norm": 0.5, "total_timesteps": 40_000},
]
 
 
# ==========================================================================
# SECTION 3 — Shared training utilities
# ==========================================================================
 
class EpisodeRewardCallback(BaseCallback):
    """Extracts episode rewards from Monitor wrapper during SB3 training."""
 
    def __init__(self):
        super().__init__(verbose=0)
        self.episode_rewards: list[float] = []
 
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True
 
 
def find_convergence(rewards: list[float], threshold=15.0, window=5) -> int:
    for i in range(window, len(rewards)):
        if np.mean(rewards[i - window: i]) >= threshold:
            return i
    return -1
 
 
def run_sb3_sweep(algo_class, grid: list[dict], algo_name: str,
                  extra_params_fn) -> tuple[list[dict], dict]:
    """
    Generic 10-run sweep for any SB3 algorithm (PPO or A2C).
 
    Parameters
    ----------
    algo_class     : PPO or A2C
    grid           : list of 10 hyperparameter dicts
    algo_name      : "ppo" or "a2c"
    extra_params_fn: callable(cfg) → dict of additional SB3 kwargs
    """
    results_csv = os.path.join(ROOT, "results", f"{algo_name}_results.csv")
    rewards_npy = os.path.join(ROOT, "results", f"{algo_name}_rewards.npy")
 
    print("\n" + "=" * 70)
    print(f"  {algo_name.upper()} Hyperparameter Sweep — 10 runs × 40,000 timesteps")
    print("=" * 70)
 
    results:     list[dict] = []
    all_rewards: dict       = {}
    best_mean   = -np.inf
    best_run_id = None
 
    for cfg in grid:
        rid = cfg["run_id"]
        print(f"\n── Run {rid:>2}/10 ─── {cfg['note']}")
        print(f"   lr={cfg['learning_rate']:.0e}  "
              f"γ={cfg['gamma']}  "
              f"ent={cfg['ent_coef']}  "
              f"vf={cfg['vf_coef']}")
 
        train_env = Monitor(TeacherRetentionEnv(difficulty="medium"))
        extra     = extra_params_fn(cfg)
 
        model = algo_class(
            policy        = "MlpPolicy",
            env           = train_env,
            learning_rate = cfg["learning_rate"],
            gamma         = cfg["gamma"],
            ent_coef      = cfg["ent_coef"],
            vf_coef       = cfg["vf_coef"],
            verbose       = 0,
            seed          = rid * 11,
            **extra,
        )
 
        tracker = EpisodeRewardCallback()
        t0 = time.time()
        model.learn(total_timesteps=cfg["total_timesteps"], callback=tracker,
                    progress_bar=False)
        elapsed = round(time.time() - t0, 2)
 
        eval_env = Monitor(TeacherRetentionEnv(difficulty="medium"))
        mean_r, std_r = evaluate_policy(model, eval_env,
                                        n_eval_episodes=10, deterministic=True)
        mean_r, std_r = round(float(mean_r), 3), round(float(std_r), 3)
        eval_env.close()
        train_env.close()
 
        conv = find_convergence(tracker.episode_rewards)
        print(f"   → mean_reward={mean_r:+.3f} ± {std_r:.3f}  "
              f"conv_ep={conv}  episodes={len(tracker.episode_rewards)}  "
              f"time={elapsed}s")
 
        run_path = os.path.join(MODELS_DIR, f"{algo_name}_run{rid}")
        model.save(run_path)
        if mean_r > best_mean:
            best_mean, best_run_id = mean_r, rid
            model.save(os.path.join(MODELS_DIR, f"best_{algo_name}_model"))
 
        all_rewards[rid] = tracker.episode_rewards
 
        row = {k: cfg[k] for k in cfg if k not in ("note",)}
        row["note"]               = cfg["note"]
        row["mean_reward"]        = mean_r
        row["std_reward"]         = std_r
        row["convergence_episode"]= conv
        row["train_time_s"]       = elapsed
        results.append(row)
 
    # Write CSV
    all_keys = list(results[0].keys())
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(results)
 
    np.save(rewards_npy, all_rewards, allow_pickle=True)
 
    print(f"\n  {algo_name.upper()} complete.  "
          f"Best: Run {best_run_id}  mean_reward={best_mean:+.3f}")
    print(f"  CSV  → {results_csv}")
 
    return results, all_rewards
 
 
# ==========================================================================
# SECTION 4 — Full sweep orchestration
# ==========================================================================
 
def run_pg_sweep():
    print("=" * 70)
    print("  Policy Gradient Sweep — REINFORCE · PPO · A2C")
    print("  Teacher Retention Environment")
    print("=" * 70)
 
    all_results = {}
 
    # ── REINFORCE ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  REINFORCE — 10 runs × 500 episodes (custom PyTorch)")
    print("=" * 70)
 
    rf_results:  list[dict] = []
    rf_rewards:  dict       = {}
    best_rf_mean = -np.inf
    best_rf_id   = None
    best_rf_policy = None
 
    RF_CSV_COLS = [
        "run_id", "note", "learning_rate", "gamma", "n_episodes",
        "hidden_size", "entropy_coef", "max_grad_norm",
        "mean_reward", "std_reward", "convergence_episode", "train_time_s",
    ]
 
    for cfg in REINFORCE_GRID:
        rid = cfg["run_id"]
        print(f"\n── Run {rid:>2}/10 ─── {cfg['note']}")
        print(f"   lr={cfg['learning_rate']:.0e}  γ={cfg['gamma']}  "
              f"hidden={cfg['hidden_size']}  ent={cfg['entropy_coef']}  "
              f"episodes={cfg['n_episodes']}")
 
        t0 = time.time()
        mean_r, std_r, ep_rewards, policy = train_reinforce(cfg)
        elapsed = round(time.time() - t0, 2)
 
        conv = find_convergence(ep_rewards)
        print(f"   → mean_reward={mean_r:+.3f} ± {std_r:.3f}  "
              f"conv_ep={conv}  episodes={len(ep_rewards)}  time={elapsed}s")
 
        # Save policy
        torch.save(policy.state_dict(),
                   os.path.join(MODELS_DIR, f"reinforce_run{rid}.pt"))
        if mean_r > best_rf_mean:
            best_rf_mean, best_rf_id = mean_r, rid
            best_rf_policy = policy
            torch.save(policy.state_dict(),
                       os.path.join(MODELS_DIR, "best_reinforce_model.pt"))
 
        rf_rewards[rid] = ep_rewards
        rf_results.append({
            "run_id": rid, "note": cfg["note"],
            "learning_rate": cfg["learning_rate"], "gamma": cfg["gamma"],
            "n_episodes": cfg["n_episodes"], "hidden_size": cfg["hidden_size"],
            "entropy_coef": cfg["entropy_coef"], "max_grad_norm": cfg["max_grad_norm"],
            "mean_reward": mean_r, "std_reward": std_r,
            "convergence_episode": conv, "train_time_s": elapsed,
        })
 
    rf_csv = os.path.join(ROOT, "results", "reinforce_results.csv")
    with open(rf_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RF_CSV_COLS)
        writer.writeheader()
        writer.writerows(rf_results)
    np.save(os.path.join(ROOT, "results", "reinforce_rewards.npy"),
            rf_rewards, allow_pickle=True)
 
    print(f"\n  REINFORCE complete.  "
          f"Best: Run {best_rf_id}  mean_reward={best_rf_mean:+.3f}")
    all_results["reinforce"] = (rf_results, rf_rewards)
 
    # ── PPO ────────────────────────────────────────────────────────────────
    def ppo_extra(cfg):
        return {
            "n_steps":    cfg["n_steps"],
            "batch_size": cfg["batch_size"],
            "n_epochs":   cfg["n_epochs"],
            "gae_lambda": cfg["gae_lambda"],
            "clip_range": cfg["clip_range"],
        }
 
    ppo_results, ppo_rewards = run_sb3_sweep(
        PPO, PPO_GRID, "ppo", ppo_extra
    )
    all_results["ppo"] = (ppo_results, ppo_rewards)
 
    # ── A2C ────────────────────────────────────────────────────────────────
    def a2c_extra(cfg):
        return {
            "n_steps":       cfg["n_steps"],
            "gae_lambda":    cfg["gae_lambda"],
            "max_grad_norm": cfg["max_grad_norm"],
        }
 
    a2c_results, a2c_rewards = run_sb3_sweep(
        A2C, A2C_GRID, "a2c", a2c_extra
    )
    all_results["a2c"] = (a2c_results, a2c_rewards)
 
    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ALL POLICY GRADIENT SWEEPS COMPLETE")
    print("=" * 70)
    for algo, (res, _) in all_results.items():
        best = max(res, key=lambda r: r["mean_reward"])
        print(f"  {algo.upper():<12} best run {best['run_id']}  "
              f"mean_reward={best['mean_reward']:+.3f}  "
              f"({best['note']})")
 
    return all_results
 
 
if __name__ == "__main__":
    run_pg_sweep()