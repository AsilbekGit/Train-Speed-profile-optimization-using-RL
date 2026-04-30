"""
Self-contained PPO package.

Goal: a stronger PPO baseline than train_ppo.py — rich observations, energy-aware
reward, observation normalization, best-checkpoint saving, training-curve logging,
richer speed-profile plot. Lives entirely under ppo/ and does not modify any
existing code (env_settings/, train_ppo.py, train_dqn.py, etc.).

Run from the project root:
    python -m ppo.train --steps 1500000
"""
