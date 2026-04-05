import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


#========================================
# Constants
#========================================

MAX_MONTHS = 24
MILESTONE_6 = 6
MILESTONE_12 = 12

#Action indices
ACTION_PRESEVERE = 0
ACTION_SEEK_SUPPORT = 1
ACTION_COMMUNITY_BUILD = 2
ACTION_SELF_CARE = 3

ACTION_REQUEST_TRANSFER = 4

#Observation indices for readability

OBS_SALARY       = 0
OBS_HOUSING      = 1
OBS_PROF_DEV     = 2
OBS_COMMUNITY    = 3
OBS_WORKLOAD     = 4
OBS_ISOLATION    = 5
OBS_RESOURCES    = 6
OBS_HEALTH       = 7
OBS_FAMILY       = 8
OBS_MONTHS       = 9
 
OBS_DIM = 10
ACTION_DIM = 5


class TeacherRetentionEnv (gym.Env):
    """
    Mission-based Reinforcement Learning environment: Teacher retention in rural Zambia.
    The agent (teacher) must navigate 24 simulated months at a rural posting,
    balancing wellbeing, professional needs, and community integration to achieve full retention
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # Initialisation

    def __init__(self, render_mode: Optional[str] = None, difficulty: str = "medium"):
        super().__init__()

        self.render_mode = render_mode
        self.difficulty = difficulty

        self._difficulty_cfg = {
            "easy": {"noise": 0.03, "start_low": 0.55},
            "medium": {"noise": 0.06, "start_low": 0.40},
            "hard": {"noise": 0.10, "start_low": 0.25},
        }[difficulty]

        # Spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape = (OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_DIM)

        # Internal Spaces
        self._state: np.ndarray = np.zeros(OBS_DIM, dtype=np.float32)
        self._consecutive_transfers: int = 0
        self._milestone_6_given:  bool = False
        self._milestone_12_given: bool = False
        self._rng = np.random.default_rng()

        # Rendering
        self._renderer = None
        if render_mode == "human":
            self._init_renderer()

    #======================================
    # Gymnasium API
    #======================================

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        try:
            super().reset(seed=seed)
        except Exception:
            pass
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        low = self._difficulty_cfg["start_low"]

        # Start state: rural posting begins with low-ish amentities
        self._state = np.array([
            self._rng.uniform(low, 0.65),   # salary_index
            self._rng.uniform(low, 0.60),   # housing_quality
            self._rng.uniform(0.1,  0.50),  # professional_dev – typically scarce
            self._rng.uniform(0.1,  0.40),  # community_ties   – not yet integrated
            self._rng.uniform(0.5,  0.85),  # workload_pressure – starts high
            self._rng.uniform(0.5,  0.90),  # isolation_score   – starts high
            self._rng.uniform(low,  0.65),  # resource_availability
            self._rng.uniform(0.6,  1.00),  # health_status – start healthy
            self._rng.uniform(0.1,  0.60),  # family_proximity
            0.0,                            # months_served = 0
        ], dtype=np.float32)

        self._months = 0
        self._consecutive_transfers = 0
        self._milestone_6_given = False
        self._milestone_12_given = False

        info = self._build_info()
        return self._state.copy(), info
    
    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action {action}"

        # Apply action effects
        self._apply_action(action)

        # Environmental stochastic dynamic
        self._apply_dynamics()

        # Advance time
        self._months += 1
        self._state[OBS_MONTHS] = self._months / MAX_MONTHS

        # Clip state to [0, 1]
        self._state = np.clip(self._state, 0.0, 1.0)

        # Compute reward
        reward, terminated, info = self._compute_reward(action)

        truncated = False 

        if self.render_mode == "human":
            self.render()

        return self._state.copy(), reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "human":
            if self._renderer is None:
                self._init_renderer()
            self._renderer.render(self._state, self._months, self._build_info())
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
        
    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    
    #===================================================
    # Action Transition Logic
    #===================================================

    def _apply_action(self, action: int):
        s = self._state

        if action == ACTION_PRESEVERE:
            s[OBS_COMMUNITY] += 0.02
            s[OBS_MONTHS] = self._months / MAX_MONTHS
            s[OBS_WORKLOAD] += 0.04
            s[OBS_HEALTH] -= 0.01

        elif action == ACTION_SEEK_SUPPORT:
            s[OBS_PROF_DEV]   += self._rng.uniform(0.05, 0.15)
            s[OBS_RESOURCES]  += self._rng.uniform(0.03, 0.10)
            s[OBS_SALARY]     += self._rng.uniform(0.00, 0.05)  # possible bonus
            s[OBS_WORKLOAD]   -= 0.02   # some admin relief
            self._consecutive_transfers = 0

        elif action == ACTION_COMMUNITY_BUILD:
            # Build local relationships, improve housing situation
            s[OBS_COMMUNITY]  += self._rng.uniform(0.06, 0.14)
            s[OBS_ISOLATION]  -= self._rng.uniform(0.06, 0.14)
            s[OBS_HOUSING]    += self._rng.uniform(0.02, 0.08)
            s[OBS_WORKLOAD]   += 0.02   # takes time
            self._consecutive_transfers = 0
 
        elif action == ACTION_SELF_CARE:
            # Prioritise wellbeing: health up, workload down, community somewhat
            s[OBS_HEALTH]     += self._rng.uniform(0.06, 0.14)
            s[OBS_WORKLOAD]   -= self._rng.uniform(0.05, 0.12)
            s[OBS_ISOLATION]  -= 0.02
            s[OBS_PROF_DEV]   -= 0.02   # less time for development
            self._consecutive_transfers = 0
 
        elif action == ACTION_REQUEST_TRANSFER:
            # Signal intent to leave: increments consecutive counter
            self._consecutive_transfers += 1
            s[OBS_WORKLOAD]   -= 0.05   # mentally checked out = less burnout
            s[OBS_COMMUNITY]  -= 0.05   # withdrawing from community
            s[OBS_ISOLATION]  += 0.05
    
    #===================================================
    # Environmental Dynamics (stochastic)
    #===================================================

    def _apply_dynamics(self):
        """
        External forces that affect the teacher regardless of their action.
        These simulate the real-world volatility of rural postings.
        """
        s = self._state
        noise_scale = self._difficulty_cfg["noise"]
        rng = self._rng
 
        # Salary may erode or fluctuate (payment delays common)
        s[OBS_SALARY]     += rng.normal(0.0, noise_scale)
 
        # Housing degrades slowly unless maintained
        s[OBS_HOUSING]    -= rng.uniform(0.0, 0.02)
 
        # Isolation naturally decreases over time as teacher settles
        settling = 0.005 * min(self._months, 12) / 12
        s[OBS_ISOLATION]  -= settling + rng.normal(0.0, noise_scale * 0.5)
 
        # Workload spikes possible (teacher shortages → extra classes)
        if rng.random() < 0.15:   # 15% chance of workload spike per month
            s[OBS_WORKLOAD] += rng.uniform(0.05, 0.20)
 
        # Random health event (illness, malaria, etc.)
        if rng.random() < 0.08:   # 8% monthly health event
            s[OBS_HEALTH]   -= rng.uniform(0.05, 0.20)
 
        # Random MoE intervention (external support)
        if rng.random() < 0.05:
            s[OBS_RESOURCES]   += rng.uniform(0.05, 0.15)
            s[OBS_PROF_DEV]    += rng.uniform(0.02, 0.08)
 
        # Family proximity: occasional family visit improves score
        if rng.random() < 0.10:
            s[OBS_FAMILY]      += rng.uniform(0.03, 0.10)

    #=======================================
    # Reward
    #=======================================

    def _compute_reward(
        self, action: int
    ) -> Tuple[float, bool, Dict[str, Any]]:
        s = self._state
        reward = 0.0
        terminated = False
        info = self._build_info()
        info["termination_reason"] = None
 
        # ── Step rewards ──────────────────────────────────────────────────
        # Positive: good conditions
        wellbeing_ok = (
            s[OBS_WORKLOAD]  < 0.75
            and s[OBS_ISOLATION] < 0.80
            and s[OBS_SALARY]    > 0.25
        )
        if wellbeing_ok:
            reward += 2.0
 
        # Negative: stress signals
        if s[OBS_WORKLOAD] > 0.80:
            reward -= 1.0
        if s[OBS_ISOLATION] > 0.85:
            reward -= 1.5
        if s[OBS_SALARY] < 0.15:
            reward -= 0.5
 
        # ── Milestone rewards ─────────────────────────────────────────────
        if self._months >= MILESTONE_6 and not self._milestone_6_given:
            reward += 1.0
            self._milestone_6_given = True
            info["milestone"] = "6-month retention"
 
        if self._months >= MILESTONE_12 and not self._milestone_12_given:
            reward += 1.0
            self._milestone_12_given = True
            info["milestone"] = "12-month retention"
 
        # ── Terminal conditions ───────────────────────────────────────────
 
        # SUCCESS: completed full 24-month posting
        if self._months >= MAX_MONTHS:
            reward += 3.0
            terminated = True
            info["termination_reason"] = "SUCCESS: 24-month posting complete"
            info["success"] = True
 
        # FAILURE 1: Burnout (health crashed)
        elif s[OBS_HEALTH] < 0.10:
            reward -= 3.0
            terminated = True
            info["termination_reason"] = "FAILURE: Burnout (health < 0.10)"
            info["success"] = False
 
        # FAILURE 2: Voluntary departure (3 consecutive transfer requests)
        elif self._consecutive_transfers >= 3:
            reward -= 5.0
            terminated = True
            info["termination_reason"] = "FAILURE: Teacher resigned (3× transfer requests)"
            info["success"] = False
 
        # EDGE CASE: Non-payment crisis
        elif s[OBS_SALARY] < 0.05:
            reward -= 3.0
            terminated = True
            info["termination_reason"] = "FAILURE: Non-payment crisis (salary < 0.05)"
            info["success"] = False
 
        return reward, terminated, info
    
    #=================================================
    # Helpers
    #=================================================

    def _build_info(self) -> Dict[str, Any]:
        s = self._state
        return {
            "months_served":           self._months,
            "salary_index":            float(s[OBS_SALARY]),
            "housing_quality":         float(s[OBS_HOUSING]),
            "professional_dev":        float(s[OBS_PROF_DEV]),
            "community_ties":          float(s[OBS_COMMUNITY]),
            "workload_pressure":       float(s[OBS_WORKLOAD]),
            "isolation_score":         float(s[OBS_ISOLATION]),
            "resource_availability":   float(s[OBS_RESOURCES]),
            "health_status":           float(s[OBS_HEALTH]),
            "family_proximity":        float(s[OBS_FAMILY]),
            "consecutive_transfers":   self._consecutive_transfers,
            "milestone":               None,
            "success":                 None,
        }
 
    def _init_renderer(self):
        """Lazy-import renderer to keep env importable without pygame."""
        from .rendering import TeacherRetentionRenderer
        self._renderer = TeacherRetentionRenderer()
 
    def _render_rgb_array(self) -> np.ndarray:
        """Minimal rgb_array fallback (returns blank frame if renderer not up)."""
        try:
            from .rendering import TeacherRetentionRenderer
            if self._renderer is None:
                self._renderer = TeacherRetentionRenderer(headless=True)
            return self._renderer.get_rgb_array(self._state, self._months)
        except Exception:
            return np.zeros((400, 600, 3), dtype=np.uint8)
    
#===============================================
# Registration helper (call once at import time)
#===============================================

def register_env():
    """Register the environment with Gymnasium."""
    from gymnasium.envs.registration import register
    try:
        register(
            id="TeacherRetention-v0",
            entry_point="environment.custom_env:TeacherRetentionEnv",
            max_episode_steps=MAX_MONTHS,
            )
    except Exception:
        pass  # already registered
 
 
register_env()

#===================================
# Smoke test
#===================================

if __name__ == "__main__":
    env = TeacherRetentionEnv(difficulty="medium")
    obs, info = env.reset(seed=42)
 
    print("=" * 60)
    print("  Teacher Retention Environment — Smoke Test")
    print("=" * 60)
    print(f"Obs shape:    {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial obs:  {np.round(obs, 3)}")
 
    total_reward = 0.0
    for step in range(30):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"  Step {step+1:>2} | action={action} | reward={reward:+.2f} | "
            f"months={info['months_served']:>2} | "
            f"health={info['health_status']:.2f} | "
            f"workload={info['workload_pressure']:.2f}"
        )
        if terminated or truncated:
            print(f"\n  Episode ended: {info.get('termination_reason')}")
            break
 
    print(f"\nTotal reward: {total_reward:.2f}")
    env.close()

