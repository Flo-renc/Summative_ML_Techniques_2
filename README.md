# Teacher Retention RL — Rural Zambia

A mission-based reinforcement learning project that models rural teacher retention in Zambia as a sequential decision problem. The agent is a teacher navigating a 24-month rural school posting, learning to manage workload, health, isolation, and professional development under realistic stochastic conditions drawn from Zambia's 2024–2029 Ministry of Education Partnership Compact.

Four algorithms were trained and compared: DQN, REINFORCE, PPO, and A2C, each with 10 hyperparameter experiments. The best trained agent (PPO) achieved a mean evaluation reward of +51.55, compared to +6.50 for a random policy.

---

## Project Structure

```
project_root/
├── environment/
│   ├── custom_env.py       # Custom Gymnasium environment
│   ├── rendering.py        # Pygame 2D visualisation
│   └── __init__.py
├── training/
│   ├── dqn_training.py     # DQN hyperparameter sweep (10 runs)
│   ├── pg_training.py      # REINFORCE, PPO, A2C sweeps (10 runs each)
│   └── __init__.py
├── models/
│   ├── dqn/                # Saved DQN models
│   └── pg/                 # Saved PPO, A2C, REINFORCE models
├── results/
│   ├── figures/            # Generated plots and charts
│   ├── dqn_results.csv
│   ├── ppo_results.csv
│   ├── a2c_results.csv
│   └── reinforce_results.csv
├── main.py                 # Run best-performing agent
├── generate_report.py      # Generate all report figures
├── requirements.txt
└── README.md
```

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/your-username/florence_rl_summative.git
cd florence_rl_summative
```

**2. Create and activate a virtual environment**

```bash
python -m venv rl-env

# Windows
rl-env\Scripts\activate

# Mac / Linux
source rl-env/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Requirements

```
gymnasium>=0.29.0
stable-baselines3>=2.3.0
torch>=2.0.0
numpy>=1.24.0
pygame>=2.5.0
matplotlib>=3.7.0
pandas>=2.0.0
```

---

## Usage

### Run the best-performing agent (with GUI)

```bash
python main.py
```

This auto-detects the best algorithm from the results CSVs and launches the pygame visualisation alongside verbose terminal output.

### Force a specific algorithm

```bash
python main.py --algo ppo
python main.py --algo dqn
python main.py --algo a2c
python main.py --algo reinforce
```

### Run without the GUI (terminal only)

```bash
python main.py --algo ppo --no-render
```

### Change number of episodes or difficulty

```bash
python main.py --algo ppo --episodes 5 --difficulty hard
```

Difficulty options: `easy`, `medium`, `hard`

### Run the environment smoke test

```bash
python environment/custom_env.py
```

---

## Training

### Train DQN (10-run hyperparameter sweep)

```bash
python training/dqn_training.py
```

Runs 10 experiments across learning rate, batch size, gamma, exploration fraction, buffer size, train frequency, and target update interval. Saves results to `results/dqn_results.csv` and the best model to `models/dqn/best_dqn_model.zip`.

### Train REINFORCE, PPO, and A2C

```bash
python training/pg_training.py
```

Runs all three policy gradient sweeps sequentially. REINFORCE is a custom PyTorch implementation. PPO and A2C use Stable Baselines3. Best models saved to `models/pg/`.

Total training time is approximately 30 to 60 minutes depending on hardware.

### Generate report figures

```bash
python generate_report.py
```

Produces all plots to `results/figures/`: cumulative reward curves, DQN objective curves, policy entropy curves, convergence plots, generalisation tests, and hyperparameter sensitivity charts. Uses real training results if available, otherwise generates illustrative synthetic curves.

---

## Environment

### Observation Space

10-dimensional continuous vector, all values in [0, 1]:

| Index | Variable | Direction |
|-------|----------|-----------|
| 0 | Salary index | High = good |
| 1 | Housing quality | High = good |
| 2 | Professional development | High = good |
| 3 | Community ties | High = good |
| 4 | Workload pressure | High = bad |
| 5 | Isolation score | High = bad |
| 6 | Resource availability | High = good |
| 7 | Health status | High = good |
| 8 | Family proximity | High = good |
| 9 | Months served | 0.0 = start, 1.0 = month 24 |

### Action Space

Discrete(5):

| Action | Name | Effect |
|--------|------|--------|
| 0 | PERSEVERE | Standard effort. Workload creeps up slowly. |
| 1 | SEEK SUPPORT | Request MoE resources. Improves professional development and salary. |
| 2 | COMMUNITY BUILD | Build local ties. Reduces isolation, increases short-term workload. |
| 3 | SELF CARE | Prioritise health. Reduces workload, small cost to professional development. |
| 4 | REQUEST TRANSFER | Signal intent to leave. Three consecutive uses trigger resignation. |

### Reward Structure

| Condition | Reward |
|-----------|--------|
| Workload < 0.75, isolation < 0.80, salary > 0.25 | +2.0 per step |
| Workload > 0.80 | -1.0 per step |
| Isolation > 0.85 | -1.5 per step |
| Salary < 0.15 | -0.5 per step |
| Month 6 milestone | +1.0 |
| Month 12 milestone | +1.0 |
| Complete month 24 | +3.0 |
| Burnout (health < 0.10) | -3.0 terminal |
| Resignation (3x transfer) | -5.0 terminal |
| Non-payment crisis (salary < 0.05) | -3.0 terminal |

### Terminal Conditions

- **Success:** months served reaches 24
- **Burnout:** health status drops below 0.10
- **Resignation:** REQUEST TRANSFER used 3 times consecutively
- **Non-payment crisis:** salary index drops below 0.05

---

## Results Summary

| Algorithm | Best Run | Mean Reward | Std Dev | Key Configuration |
|-----------|----------|-------------|---------|-------------------|
| PPO | Run 5 | +51.55 | 3.12 | Narrow clip = 0.10 |
| DQN | Run 4 | +51.50 | 2.34 | Batch = 128, gamma = 0.99 |
| A2C | Run 9 | +49.15 | 6.24 | GAE lambda = 0.90 |
| REINFORCE | Run 3 | +36.65 | 26.37 | LR = 1e-4, gamma = 0.99 |
| Random policy | — | +6.50 | — | Baseline |

PPO and DQN achieved near-identical peak rewards. PPO Run 6 (20 training epochs) produced the most stable result across all 40 experiments with a standard deviation of just 1.50. REINFORCE significantly underperformed due to the absence of a value function baseline, which makes gradient estimates noisy in this stochastic environment.

---

## Key Findings

- Actor-critic methods (PPO, A2C) outperform policy-only (REINFORCE) on stochastic environments by using a value function to reduce gradient variance.
- Very high discount factors (gamma = 0.999) hurt all algorithms. The 24-step episode is short enough that standard discounting already captures the full return.
- The trained PPO agent learned to prioritise SELF CARE and SEEK SUPPORT during high-workload months, and to switch to community building once immediate risks are resolved. This mirrors evidence-based retention interventions in Zambia's rural education literature.
- PERSEVERE, COMMUNITY BUILD, and REQUEST TRANSFER were rarely or never selected by the best-performing agents, reflecting the reward structure's emphasis on workload and health management over community investment.

---

## Policy Relevance

This environment is grounded in Zambia's real teacher retention challenge. The observation space variables (salary, isolation, workload, housing, professional development) reflect the documented drivers of attrition at rural postings. The reward function encodes the MoE's retention priorities. The trained policy offers a simplified but principled model of which monthly decisions best support a teacher completing their posting, with relevance to incentive design, early-warning attrition systems, and provincial resource allocation.

---

## Assignment Context

Summative Assignment — Mission Based Reinforcement Learning
Course: ML Techniques 2
Institution: ALU (African Leadership University)
Due: 1 April 2025