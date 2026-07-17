"""
DREAM RL Extension
==================
Risk-aware reinforcement learning built on top of the DRIFT + IDEAM stack.

Stages:
  Stage 2: DREAMHighwayEnv (gym.Env wrapper) + ObservationBuilder
  Stage 3: RewardFn + SafetyCostFn
  Stage 4: PPO training script + eval
  Stage 5: Benchmark vs DREAM conservative baseline
  Stage 6: PINN risk encoding improvements
"""
