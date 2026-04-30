# Train Speed Profile Optimization with Reinforcement Learning

Three RL algorithms — tabular **Q-SARSA**, vanilla **DQN**, and **PPO** — drive
a simulated electric multiple unit (Uzbekistan ER9E) along a 75 km route, picking
one of four actions per 100 m segment (Brake, Coast, Cruise, Power) to balance
**route completion** against **energy consumption**.

The physics model is shared across all algorithms and uses the **Davis equation**
for rolling/aerodynamic resistance, plus grade and curvature terms.

---

## Project structure

```
.
├── data/
│   ├── data.csv                # route: grade %, speed limit (m/s), curvature %
│   ├── coordinates.dat
│   └── utils.py                # load_data, discretize_state helpers
│
├── env_settings/               # PHYSICS + ENV — shared by every algorithm
│   ├── config.py               # masses, power, Davis coefficients, dt, dx, paths
│   ├── physics.py              # Davis r_t(V) = C0 + C1·V + C2·V² + grade + curvature
│   ├── environment.py          # TrainEnv: step physics + base reward (no energy term)
│   └── gym_env.py              # Gymnasium adapter + load_data + require_gpu (used by DQN / PPO v1)
│
├── qsarsa_dqn/
│   └── qsarsa.py               # Tabular Q-SARSA with CM (convergence-monitor) switching
│
├── ppo/                        # ── new, improved PPO experiment ──
│   ├── env_wrapper.py          # PPOTrainEnv: 8-D obs, energy-aware reward, tougher limits
│   ├── train.py                # Tianshou PPO + obs normalization + best-checkpoint + logger
│   └── plot.py                 # Speed profile w/ grade band & limit overlay; training curves
│
├── train_qsarsa.py             # entrypoint — Q-SARSA
├── train_dqn.py                # entrypoint — Tianshou DQN (vanilla, target net, GPU only)
├── train_ppo.py                # entrypoint — PPO v1 (raw obs, no energy reward; baseline)
├── main.py                     # Q-SARSA + CM analysis driver
├── cm_analyzer.py              # convergence-monitoring analysis on Q-SARSA Q-table
│
├── results_cm/                 # all outputs land here
│   ├── qsarsa/                 # Q-SARSA artifacts
│   ├── dqn/                    # Tianshou DQN artifacts
│   ├── ppo/                    # PPO v1 artifacts
│   └── ppo_v2/                 # PPO v2 artifacts (from ppo/)
│
├── requirements.txt
└── README.md
```

### What each algorithm directory does

| Where | Role |
|---|---|
| `env_settings/physics.py` | Davis equation + traction-force model (untouched across all experiments) |
| `env_settings/environment.py` | Step dynamics + base reward (progress, completion, soft limit penalty) |
| `env_settings/gym_env.py` | Gymnasium wrapper used by DQN and PPO v1; identical 2-D obs `[seg_idx, v]` |
| `qsarsa_dqn/qsarsa.py` | Tabular CM-switching Q-SARSA from the paper |
| `train_dqn.py` | Tianshou vanilla DQN with target net, ε-greedy, single env, **CUDA required** |
| `train_ppo.py` | PPO v1 — Tianshou PPO on the bare 2-D obs; baseline that ignores energy |
| `ppo/` | PPO v2 — rich obs, energy-aware reward, obs normalization, best-checkpoint, training curves |

---

## Setup

Python 3.12, an NVIDIA GPU, and a CUDA-enabled PyTorch build are required for
the neural-network trainers (DQN and PPO). Q-SARSA is pure NumPy and runs on
CPU.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install tianshou gymnasium                    # for DQN / PPO trainers
```

Quick GPU sanity check:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
The DQN / PPO trainers abort with a clear error if CUDA is not available.

---

## How to run

All commands assume the venv is active (or you call `./venv/bin/python` directly).
Run from the repository root.

### 1. Q-SARSA (tabular, CPU)

```bash
python train_qsarsa.py
python main.py                  # Q-SARSA + convergence-monitor analysis
```
Outputs land in `results_cm/qsarsa/`.

### 2. DQN (Tianshou, GPU required)

Vanilla DQN with a target network, ε-greedy decay 1.0 → 0.05, single env (canonical).

```bash
python train_dqn.py --steps 1500000                       # default
python train_dqn.py --steps 50000                         # quick sanity
python train_dqn.py --steps 1500000 --target-update-freq 1000   # less aggressive sync
python train_dqn.py --steps 1500000 --n-envs 8            # multi-env if you really want
```
Outputs land in `results_cm/dqn/`:
- `dqn.pt` — final policy
- `speed_profile.npz`, `speed_profile.png`

### 3. PPO v1 — baseline (Tianshou, GPU required)

PPO on the same 2-D obs as the original DQN. No energy reward. Useful as a
reference point only.

```bash
python train_ppo.py --steps 2000000
```
Outputs land in `results_cm/ppo/`.

### 4. PPO v2 — improved (`ppo/`, GPU required)

This is the main PPO experiment. Rich 8-D observation (current/lookahead grade,
current/lookahead speed limit, distance to next station), energy-aware reward
shaping, observation normalization, best-checkpoint saving, training-curve PNG.

```bash
# Full run (recommended)
python -m ppo.train --steps 1500000

# Longer
python -m ppo.train --steps 3000000

# Quick sanity (~5 s)
python -m ppo.train --steps 8000

# Custom output directory
python -m ppo.train --steps 1500000 --out-dir results_cm/ppo_v2_run2
```

Useful tuning knobs:
- `--energy-coef 2.0` — reward penalty per kWh of step energy. Raise to push harder on energy.
- `--limit-pen 2.0` — proportional penalty per m/s of speed-limit overshoot.
- `--limit-term 5.0` — m/s overshoot above the limit that ends the episode.
- `--hidden 256 256` — actor / critic MLP layer sizes.
- `--n-envs 8` — parallel envs for rollout collection.
- `--clip 0.2 --gamma 0.99 --gae-lambda 0.95 --ent-coef 0.01 --vf-coef 0.5` — standard PPO knobs.

Outputs (`results_cm/ppo_v2/`):
- `ppo_best.pt` — checkpoint of the best test-return policy (loaded for the final rollout)
- `ppo_last.pt` — checkpoint at end of training
- `speed_profile.png` — 4-panel chart: speed + limit overlay, grade band, cumulative energy, action timeline (all sharing the X axis)
- `speed_profile.npz` — raw arrays for downstream analysis
- `training_curves.png` — return / episode length / loss vs. env steps
- `train.csv`, `test.csv`, `update.csv`, `info.csv` — all training stats in flat CSVs

---

## Reading a `speed_profile.png` (PPO v2)

The four stacked panels share the X axis (position in km):

1. **Speed (km/h)** — blue line is the train, red step line is the speed limit.
   Vertical dotted lines are stations. The train should sit comfortably under
   the red line.
2. **Grade band** — red where the route climbs (uphill), green where it
   descends. A good policy powers in the red zones and coasts in the green.
3. **Cumulative energy (kWh)** — should stay flat in green (downhill) zones
   and tick up in red ones. Long flat stretches indicate good coasting.
4. **Action strip** — Brake / Coast / Cruise / Power. Lots of Coast (orange)
   on flats and downhills is the canonical fuel-efficient pattern.

---

## Physics in one paragraph

`env_settings/physics.py` computes the total resistance acting on the train as
the sum of three components, all in Newtons:

- **Davis equation** (rolling + aerodynamic): `r_t(V) = C0 + C1·V + C2·V²`,
  with `V` in km/h and `r_t` in kg·s/ton, multiplied by `mass_tons · g`.
  Coefficients (`C0=1.1, C1=0.01, C2=0.000227`) are tuned for the Uzbekistan
  ER9E.
- **Grade resistance**: `m · g · sin(θ) ≈ m · g · (grade%/100)`.
- **Curve resistance**: `mass_tons · 6 · |curvature%|` (Roeckl approximation).

The traction force is capped by three limits — instantaneous power
(`F = P·η/v`), wheel-rail adhesion (`μ·m·g`), and a passenger-comfort cap
(`m · a_max`).

Energy at each timestep, in `environment.py:step`:
```
mechanical_power = F_traction × v_avg            (W)
electrical_power = mechanical_power / η          (η = 0.85)
energy_step      = electrical_power × dt / 3.6e6 (kWh)
```
Brake and Coast are zero-energy actions; there is no regenerative braking
modeled. This is why a good policy avoids braking — kinetic energy paid for
during Power cannot be recovered.

---

## GPU requirements

- **Q-SARSA**: pure NumPy, CPU only. GPU does nothing useful here.
- **DQN, PPO v1, PPO v2**: hard-require CUDA. The trainers abort with
  `ERROR: CUDA is required` if no NVIDIA GPU is available.

Tested on an NVIDIA GB10 (DGX Spark) with `torch 2.11.0+cu130`.
