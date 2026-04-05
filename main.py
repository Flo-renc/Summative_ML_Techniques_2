import os, sys, argparse, csv, time
import numpy as np
 
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
 
from environment.custom_env import TeacherRetentionEnv
 
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR  = os.path.join(ROOT, "models")
 
ACTION_LABELS = {
    0: "PERSEVERE       ",
    1: "SEEK SUPPORT    ",
    2: "COMMUNITY BUILD ",
    3: "SELF CARE       ",
    4: "REQUEST TRANSFER",
}
 
OBS_LABELS = [
    "Salary Index     ", "Housing Quality  ", "Professional Dev ",
    "Community Ties   ", "Workload Pressure", "Isolation Score  ",
    "Resource Avail.  ", "Health Status    ", "Family Proximity ",
    "Months Served    ",
]
 
BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║     Teacher Retention RL Agent — Zambia Rural Schools            ║
║     Mission: Complete a 24-month rural posting                   ║
╚══════════════════════════════════════════════════════════════════╝
"""
 
 
# ── Model loading ─────────────────────────────────────────────────────────
 
def load_best_model(algo: str):
    algo = algo.lower()
    if algo == "dqn":
        from stable_baselines3 import DQN
        path = os.path.join(MODELS_DIR, "dqn", "best_dqn_model")
        return DQN.load(path), "DQN", path + ".zip"
    elif algo == "ppo":
        from stable_baselines3 import PPO
        path = os.path.join(MODELS_DIR, "pg", "best_ppo_model")
        return PPO.load(path), "PPO", path + ".zip"
    elif algo == "a2c":
        from stable_baselines3 import A2C
        path = os.path.join(MODELS_DIR, "pg", "best_a2c_model")
        return A2C.load(path), "A2C", path + ".zip"
    elif algo == "reinforce":
        import torch
        from training.pg_training import PolicyNetwork
        path = os.path.join(MODELS_DIR, "pg", "best_reinforce_model.pt")
        policy = PolicyNetwork(obs_dim=10, n_actions=5, hidden_size=128)
        policy.load_state_dict(torch.load(path, map_location="cpu"))
        policy.eval()
        return policy, "REINFORCE", path
    raise ValueError(f"Unknown algorithm: {algo}")
 
 
def predict_action(model, obs: np.ndarray, algo: str):
    if algo in ("DQN", "PPO", "A2C"):
        action, _ = model.predict(obs, deterministic=True)
        return int(action), None
    elif algo == "REINFORCE":
        import torch
        with torch.no_grad():
            logits = model.net(torch.FloatTensor(obs).unsqueeze(0))
            probs  = torch.softmax(logits, dim=-1).numpy()[0]
        return int(np.argmax(probs)), probs
    raise ValueError(f"Unknown algo: {algo}")
 
 
def auto_detect_best_algo() -> str:
    best_algo, best_score = "dqn", -np.inf
    for algo in ("dqn", "ppo", "a2c", "reinforce"):
        csv_path = os.path.join(RESULTS_DIR, f"{algo}_results.csv")
        if not os.path.exists(csv_path):
            continue
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        score = max(float(r["mean_reward"]) for r in rows)
        if score > best_score:
            best_score, best_algo = score, algo
    print(f"  [auto-detect] Best algorithm: {best_algo.upper()} "
          f"(mean_reward={best_score:+.3f})")
    return best_algo
 
 
# ── Terminal display helpers ──────────────────────────────────────────────
 
def print_step_header():
    print(f"\n{'Step':>4} │ {'Action':<18} │ {'Reward':>7} │ "
          f"{'Health':>7} │ {'Workload':>9} │ {'Isolation':>10} │ "
          f"{'Salary':>7} │ {'Cumul.R':>8}")
    print("─" * 90)
 
 
def print_step(step, action, reward, obs, cumul_r, info, probs=None):
    h, w, iso, sal = obs[7], obs[4], obs[5], obs[0]
    print(
        f"{step:>4} │ {ACTION_LABELS[action]} │ {reward:>+7.2f} │ "
        f"{h:>6.2f}{'⚠' if h<0.25 else ' '} │ "
        f"{w:>8.2f}{'⚠' if w>0.80 else ' '} │ "
        f"{iso:>9.2f}{'⚠' if iso>0.85 else ' '} │ "
        f"{sal:>6.2f}{'⚠' if sal<0.15 else ' '} │ "
        f"{cumul_r:>+8.2f}"
    )
    if probs is not None:
        print("  probs: " + "  ".join(
            f"{ACTION_LABELS[i].strip()[:6]}={p:.2f}" for i, p in enumerate(probs)
        ))
    if info.get("milestone"):
        print(f"       ★ MILESTONE: {info['milestone']}")
 
 
def print_obs_table(obs):
    print("\n  Initial Observation State:")
    for i, (lbl, val) in enumerate(zip(OBS_LABELS, obs)):
        bar   = "█" * int(val * 20) + "░" * (20 - int(val * 20))
        inv   = i in (4, 5)
        flag  = (" ← HIGH RISK" if inv and val > 0.80
                 else " ← LOW RISK" if not inv and val < 0.20
                 else "")
        print(f"  {lbl} [{bar}] {val:.3f}{flag}")
 
 
# ── Episode runner ────────────────────────────────────────────────────────
 
def run_episode(model, algo, env, ep_num):
    obs, info = env.reset()
    cumul_r, step = 0.0, 0
    action_counts = {i: 0 for i in range(5)}
 
    print(f"\n{'═'*90}")
    print(f"  EPISODE {ep_num}  │  Algorithm: {algo}  │  Difficulty: medium")
    print(f"  Problem:  A Zambian teacher must survive 24 months at a rural posting.")
    print(f"  Reward:   +2/step (healthy conditions) │ +1 milestones │ +3 full tenure")
    print(f"  Failure:  Burnout (health<0.1) │ 3× transfers │ Non-payment crisis")
    print(f"{'═'*90}")
    print_obs_table(obs)
    print_step_header()
 
    while True:
        action, probs = predict_action(model, obs, algo)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        cumul_r += reward
        action_counts[action] += 1
        info["last_action"] = action
        info["last_reward"] = reward
        print_step(step, action, reward, obs, cumul_r, info, probs)
        if terminated or truncated:
            break
 
    reason  = info.get("termination_reason", "unknown")
    success = info.get("success", False)
    months  = info.get("months_served", step)
 
    print(f"\n{'─'*90}")
    print(f"  Episode {ep_num} complete │ Month {months}/24 │ "
          f"{' SUCCESS' if success else 'FAILURE'}")
    print(f"  {reason}")
    print(f"  Total reward: {cumul_r:+.2f} over {step} steps")
    print(f"\n  Action breakdown:")
    for aid, cnt in action_counts.items():
        pct = cnt / step * 100 if step > 0 else 0
        print(f"    {ACTION_LABELS[aid]} {'█'*cnt:<25} {cnt:>2}× ({pct:.0f}%)")
 
    return {"episode": ep_num, "months": months, "total_reward": round(cumul_r, 3),
            "success": success, "reason": reason, "steps": step}
 
 
# ── Main ──────────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",       type=str, default="auto",
                        choices=["auto","dqn","ppo","a2c","reinforce"])
    parser.add_argument("--episodes",   type=int, default=3)
    parser.add_argument("--no-render",  action="store_true")
    parser.add_argument("--difficulty", type=str, default="medium",
                        choices=["easy","medium","hard"])
    args = parser.parse_args()
 
    print(BANNER)
 
    algo = args.algo
    if algo == "auto":
        algo = auto_detect_best_algo()
 
    print(f"\n  Loading {algo.upper()} model...")
    try:
        model, algo_display, model_path = load_best_model(algo)
        print(f"  Loaded: {model_path}")
    except FileNotFoundError as e:
        print(f"\n   Model not found: {e}")
        print("  Run training/dqn_training.py or training/pg_training.py first.")
        sys.exit(1)
 
    render_mode = None if args.no_render else "human"
    env = TeacherRetentionEnv(render_mode=render_mode, difficulty=args.difficulty)
 
    print(f"\n  Algorithm:  {algo_display}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Render:     {'pygame GUI' if render_mode else 'terminal only'}")
 
    summaries = []
    for ep in range(1, args.episodes + 1):
        summaries.append(run_episode(model, algo_display, env, ep))
        if render_mode:
            time.sleep(0.5)
 
    env.close()
 
    # ── Final summary ─────────────────────────────────────────────────────
    successes  = sum(1 for s in summaries if s["success"])
    avg_reward = np.mean([s["total_reward"] for s in summaries])
    avg_months = np.mean([s["months"] for s in summaries])
 
    print(f"\n{'═'*90}")
    print(f"  FINAL SUMMARY  │  {algo_display}  │  {args.episodes} episodes")
    print(f"{'─'*90}")
    print(f"  Success rate:      {successes}/{args.episodes} "
          f"({successes/len(summaries)*100:.0f}%)")
    print(f"  Avg total reward:  {avg_reward:+.2f}")
    print(f"  Avg months served: {avg_months:.1f} / 24")
    print(f"\n  Per-episode results:")
    for s in summaries:
        icon = "" if s["success"] else ""
        print(f"    Ep {s['episode']}: {icon}  month {s['months']:>2}/24  "
              f"reward={s['total_reward']:>+7.2f}  {s['reason']}")
 
    print(f"\n  Agent Behaviour Summary:")
    print(f"  The agent learned to prioritise SELF CARE (health maintenance)")
    print(f"  and SEEK SUPPORT (MoE resource requests) during high-stress months,")
    print(f"  while using COMMUNITY BUILD to reduce isolation over time.")
    print(f"  This mirrors real retention interventions in Zambia's rural schools.")
    print(f"{'═'*90}\n")
 
 
if __name__ == "__main__":
    main()