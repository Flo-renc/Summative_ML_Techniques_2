import os, sys, csv, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from environment.custom_env import TeacherRetentionEnv


# Output Directories

os.makedirs(os.path.join(ROOT, "results"),        exist_ok=True)
os.makedirs(os.path.join(ROOT, "models", "dqn"),  exist_ok=True)
 
RESULTS_CSV = os.path.join(ROOT, "results", "dqn_results.csv")
REWARDS_NPY = os.path.join(ROOT, "results", "dqn_rewards.npy")
MODELS_DIR  = os.path.join(ROOT, "models",  "dqn")

# 10-run hyperparameter grid
# Columns become the table headers in the report (rubric requirement).
# Rationale for each run is documented inline so Discussion section
# can explain the effect of each hyperparameter.
 
HYPERPARAMETER_GRID = [
    # Run 1 — Baseline: conservative defaults
    {
        "run_id": 1,  "note": "Baseline defaults",
        "learning_rate": 1e-4,  "batch_size": 32,   "gamma": 0.95,
        "exploration_fraction": 0.30, "exploration_final_eps": 0.05,
        "buffer_size": 5_000,   "train_freq": 4,   "target_update_interval": 500,
        "total_timesteps": 40_000,
    },
    # Run 2 — Higher LR: faster Q-value updates, risk of instability
    {
        "run_id": 2,  "note": "High learning rate",
        "learning_rate": 5e-4,  "batch_size": 32,   "gamma": 0.95,
        "exploration_fraction": 0.30, "exploration_final_eps": 0.05,
        "buffer_size": 5_000,   "train_freq": 4,   "target_update_interval": 500,
        "total_timesteps": 40_000,
    },
    # Run 3 — Low LR + longer exploration: stable but slow convergence
    {
        "run_id": 3,  "note": "Low LR + high exploration",
        "learning_rate": 5e-5,  "batch_size": 64,   "gamma": 0.97,
        "exploration_fraction": 0.50, "exploration_final_eps": 0.10,
        "buffer_size": 10_000,  "train_freq": 4,   "target_update_interval": 1000,
        "total_timesteps": 40_000,
    },
    # Run 4 — Large batch + fast target updates: less variance per update
    {
        "run_id": 4,  "note": "Large batch, fast target updates",
        "learning_rate": 2e-4,  "batch_size": 128,  "gamma": 0.99,
        "exploration_fraction": 0.20, "exploration_final_eps": 0.02,
        "buffer_size": 10_000,  "train_freq": 1,   "target_update_interval": 250,
        "total_timesteps": 40_000,
    },
    # Run 5 — Very high gamma: agent values long-horizon retention heavily
    {
        "run_id": 5,  "note": "High gamma (long-horizon)",
        "learning_rate": 1e-4,  "batch_size": 64,   "gamma": 0.999,
        "exploration_fraction": 0.40, "exploration_final_eps": 0.05,
        "buffer_size": 10_000,  "train_freq": 4,   "target_update_interval": 500,
        "total_timesteps": 40_000,
    },
    # Run 6 — Small replay buffer: forces learning from recent transitions only
    {
        "run_id": 6,  "note": "Small buffer (recency bias)",
        "learning_rate": 3e-4,  "batch_size": 32,   "gamma": 0.95,
        "exploration_fraction": 0.30, "exploration_final_eps": 0.08,
        "buffer_size": 2_000,   "train_freq": 2,   "target_update_interval": 200,
        "total_timesteps": 40_000,
    },
    # Run 7 — Very high exploration: maximises state coverage, delayed exploitation
    {
        "run_id": 7,  "note": "Very high exploration",
        "learning_rate": 1e-4,  "batch_size": 64,   "gamma": 0.97,
        "exploration_fraction": 0.70, "exploration_final_eps": 0.15,
        "buffer_size": 10_000,  "train_freq": 4,   "target_update_interval": 500,
        "total_timesteps": 40_000,
    },
    # Run 8 — Aggressive learning: high LR + tiny batch + fast updates
    {
        "run_id": 8,  "note": "Aggressive (high LR, tiny batch)",
        "learning_rate": 1e-3,  "batch_size": 16,   "gamma": 0.95,
        "exploration_fraction": 0.20, "exploration_final_eps": 0.05,
        "buffer_size": 5_000,   "train_freq": 1,   "target_update_interval": 100,
        "total_timesteps": 40_000,
    },
    # Run 9 — Very slow / stable: low LR + large buffer + infrequent training
    {
        "run_id": 9,  "note": "Very slow, large buffer",
        "learning_rate": 1e-5,  "batch_size": 128,  "gamma": 0.99,
        "exploration_fraction": 0.60, "exploration_final_eps": 0.05,
        "buffer_size": 20_000,  "train_freq": 8,   "target_update_interval": 1000,
        "total_timesteps": 40_000,
    },
    # Run 10 — Tuned best: balanced LR, moderate batch, high gamma
    {
        "run_id": 10, "note": "Balanced / tuned best",
        "learning_rate": 2e-4,  "batch_size": 64,   "gamma": 0.99,
        "exploration_fraction": 0.40, "exploration_final_eps": 0.05,
        "buffer_size": 10_000,  "train_freq": 4,   "target_update_interval": 500,
        "total_timesteps": 40_000,
    },
]
 
# Callback: episode reward tracker
class RewardTracker(BaseCallback):
    """Records per-episode total rewards during training."""
 
    def __init__(self):
        super().__init__(verbose=0)
        self.episode_rewards: list[float] = []
        self._ep_reward: float = 0.0
 
    def _on_step(self) -> bool:
        # SB3 Monitor wrapper already logs episode rewards into infos
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True
 
 
def convergence_episode(rewards: list[float], threshold=15.0, window=5) -> int:
    """First episode where rolling mean over `window` exceeds threshold."""
    for i in range(window, len(rewards)):
        if np.mean(rewards[i - window: i]) >= threshold:
            return i
    return -1   # did not converge
 
 
#  Main sweep 
 
def run_dqn_sweep() -> tuple[list[dict], dict]:
    print("=" * 70)
    print("  DQN Hyperparameter Sweep — Teacher Retention Environment")
    print(f"  {len(HYPERPARAMETER_GRID)} runs  ×  40,000 timesteps each")
    print("=" * 70)
 
    results: list[dict] = []
    all_rewards: dict   = {}
    best_mean   = -np.inf
    best_run_id = None
 
    CSV_COLS = [
        "run_id", "note",
        "learning_rate", "batch_size", "gamma",
        "exploration_fraction", "exploration_final_eps",
        "buffer_size", "train_freq", "target_update_interval",
        "total_timesteps",
        "mean_reward", "std_reward", "convergence_episode", "train_time_s",
    ]
 
    for cfg in HYPERPARAMETER_GRID:
        rid = cfg["run_id"]
        print(f"\n── Run {rid:>2}/10 ─── {cfg['note']}")
        print(f"   lr={cfg['learning_rate']:.0e}  batch={cfg['batch_size']}  "
              f"γ={cfg['gamma']}  ε_frac={cfg['exploration_fraction']}  "
              f"ε_final={cfg['exploration_final_eps']}  "
              f"buf={cfg['buffer_size']}  tf={cfg['train_freq']}  "
              f"tui={cfg['target_update_interval']}")
 
        train_env = Monitor(TeacherRetentionEnv(difficulty="medium"))
 
        model = DQN(
            policy                   = "MlpPolicy",
            env                      = train_env,
            learning_rate            = cfg["learning_rate"],
            batch_size               = cfg["batch_size"],
            gamma                    = cfg["gamma"],
            exploration_fraction     = cfg["exploration_fraction"],
            exploration_final_eps    = cfg["exploration_final_eps"],
            buffer_size              = cfg["buffer_size"],
            train_freq               = cfg["train_freq"],
            target_update_interval   = cfg["target_update_interval"],
            verbose                  = 0,
            seed                     = rid * 13,
        )
 
        tracker = RewardTracker()
        t0 = time.time()
        model.learn(total_timesteps=cfg["total_timesteps"], callback=tracker,
                    progress_bar=False)
        elapsed = round(time.time() - t0, 2)
 
        # Evaluate on fresh env
        eval_env = Monitor(TeacherRetentionEnv(difficulty="medium"))
        mean_r, std_r = evaluate_policy(model, eval_env,
                                        n_eval_episodes=10, deterministic=True)
        mean_r, std_r = round(float(mean_r), 3), round(float(std_r), 3)
        eval_env.close()
 
        conv = convergence_episode(tracker.episode_rewards)
 
        print(f"   → mean_reward={mean_r:+.3f} ± {std_r:.3f}  "
              f"conv_ep={conv}  episodes={len(tracker.episode_rewards)}  "
              f"time={elapsed}s")
 
        # Save model
        run_path = os.path.join(MODELS_DIR, f"dqn_run{rid}")
        model.save(run_path)
        if mean_r > best_mean:
            best_mean, best_run_id = mean_r, rid
            model.save(os.path.join(MODELS_DIR, "best_dqn_model"))
 
        all_rewards[rid] = tracker.episode_rewards
 
        results.append({
            "run_id":                  rid,
            "note":                    cfg["note"],
            "learning_rate":           cfg["learning_rate"],
            "batch_size":              cfg["batch_size"],
            "gamma":                   cfg["gamma"],
            "exploration_fraction":    cfg["exploration_fraction"],
            "exploration_final_eps":   cfg["exploration_final_eps"],
            "buffer_size":             cfg["buffer_size"],
            "train_freq":              cfg["train_freq"],
            "target_update_interval":  cfg["target_update_interval"],
            "total_timesteps":         cfg["total_timesteps"],
            "mean_reward":             mean_r,
            "std_reward":              std_r,
            "convergence_episode":     conv,
            "train_time_s":            elapsed,
        })
 
        train_env.close()
    
    #  Write CSV 
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS)
        writer.writeheader()
        writer.writerows(results)
 
    np.save(REWARDS_NPY, all_rewards, allow_pickle=True)
 
    print("\n" + "=" * 70)
    print(f"  Sweep complete.  Best: Run {best_run_id}  "
          f"mean_reward={best_mean:+.3f}")
    print(f"  CSV    → {RESULTS_CSV}")
    print(f"  Rewards→ {REWARDS_NPY}")
    print(f"  Model  → {os.path.join(MODELS_DIR, 'best_dqn_model.zip')}")
    print("=" * 70)
 
    return results, all_rewards


if __name__ == "__main__":
    run_dqn_sweep()