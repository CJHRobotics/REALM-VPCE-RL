"""Reinforcement learning over the VPCE representation.

    VpceObserver   live simulation  -> place-cell activation vector
    WebotsEnv      Gymnasium navigation task over a connected robot
    RewardConfig   what the agent is paid for
    learning_curve plotting the Monitor CSV a run leaves behind

Nothing here imports Webots at module scope, so the whole package is
importable -- and the task is testable -- off the cluster. The Webots
dependency enters only when a robot is handed to the environment.
"""

from realm_tools.rl_lib.webots_env import (
    WebotsEnv,
    RewardConfig,
    N_ACTIONS,
    action_masks,
)
from realm_tools.rl_lib.vpce_observer import VpceObserver, THETAS

__all__ = [
    'WebotsEnv', 'RewardConfig', 'N_ACTIONS', 'action_masks',
    'VpceObserver', 'THETAS',
]
