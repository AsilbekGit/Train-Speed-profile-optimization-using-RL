"""
Self-contained PPO package.

The project's PPO implementation — rich 8-D observations, energy-aware reward,
observation normalization, best-checkpoint saving, training-curve logging, and a
richer speed-profile plot. Lives entirely under ppo/ (results in ppo/results/)
and does not modify any shared code (env_settings/, train_dqn.py, etc.).

Run from the project root:
    python -m ppo.train --steps 1500000
"""
