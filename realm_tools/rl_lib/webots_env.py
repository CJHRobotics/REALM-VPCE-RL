"""Gymnasium navigation environment over Webots, observed through VPCE.

The agent starts somewhere in the arena and has to reach the goal. What it
sees is a place-cell population vector and nothing else: no coordinates, no
heading, no goal bearing, no odometry. That is the experiment -- the
representation is the independent variable, so anything else in the
observation would confound it.

    obs      (n_cells,) float32   place-cell activations, summing to 1
    action   Discrete(8)          the robot's own action set, 45 deg apart
    reward   sparse by default    see RewardConfig
    done     at the goal, or when the step budget runs out

Separation of concerns. This class owns the task -- episodes, actions,
reward, termination. `VpceObserver` owns the mapping from the simulation to
an observation. `MyRobot` owns the robot. The env never touches a camera or
a feature vector itself, which is what lets the whole task be exercised off
-cluster against a stand-in robot (see `tests/test_webots_env.py`); Webots
cannot be imported outside a running simulation, so an env that reached into
it directly could not be tested anywhere but on the cluster.

What the env is allowed to know. The OBSERVATION is VPCE only. The REWARD
may use ground truth -- it is the task definition, not something the agent
perceives, and every RL benchmark defines reward privileged in this way.
Distance shaping is nevertheless off by default: a dense distance signal
turns navigation into gradient-following and would compress the differences
between representations that this experiment exists to measure. Turn it on
deliberately, as a condition, not as a convenience.

Running it. HamBot subclasses Webots' Supervisor, so it can only be built
inside a controller process -- this env cannot be constructed from an
ordinary Python script. Two ways round that:

  * the controller IS the training script. `simulation/controllers/rl_agent/`
    does this, and Webots launches it. Simplest, one env per Webots.
  * an extern controller. The training script is the parent, launches Webots
    with the robot's controller field set to `<extern>`, and the controller
    library connects back over WEBOTS_CONTROLLER_URL. Needed for vectorised
    envs, where N Webots instances each need their own display and port.

The env is identical either way -- it takes a robot that is already
connected, and does not care how that happened.
"""

from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# The robot's own action set: 8 headings, 45 degrees apart. Taken from
# MyRobot.action_set rather than redefined, so the two cannot drift.
N_ACTIONS = 8


@dataclass
class RewardConfig:
    """What the agent is paid for.

    The defaults are deliberately sparse. `goal` dominates, `step` creates
    pressure to be quick, `blocked` discourages driving into walls, and
    nothing else is given away. Learning time under this reward is a
    reasonably clean read on how usable the representation is.
    """

    goal: float = 1.0
    # Per step. Small relative to `goal` so the optimum is still to reach the
    # goal rather than to end the episode quickly by any means -- with a
    # large step cost and an escapable episode, not moving can beat trying.
    step: float = -0.01
    # Attempting a direction the lidar says is blocked. The robot does not
    # move; without a penalty the agent can stall against a wall for free.
    blocked: float = -0.05
    # Potential-based shaping on distance to the goal, scaled by this. 0 is
    # off. Potential-based (gamma * phi(s') - phi(s)) rather than raw
    # distance, because that form provably leaves the optimal policy
    # unchanged -- a raw bonus does not, and can pay an agent to loiter near
    # the goal instead of entering it.
    shaping: float = 0.0
    shaping_gamma: float = 0.99


class WebotsEnv(gym.Env):
    """Navigation in a Webots arena, observed as place-cell activations.

    Parameters
    ----------
    robot : MyRobot
        An already-connected robot. The env does not create it: construction
        requires a live Webots connection, and keeping that outside makes
        the env testable.
    observer : VpceObserver, or any object with `observe(robot)` and `n_cells`
    max_steps : int
        Episode truncation. With teleport actions a step is cheap; with
        physical actions it is not.
    teleport : bool
        True moves the robot by teleporting it one action length, skipping
        the drive. Much faster and the usual choice for training. False
        drives with PID, which is what a policy would do for real, and is
        the honest setting for evaluation.
    start_mode, start_index : str, int
        Passed to `robot.move_to_start`. index -1 picks a random start,
        which is what training should normally use -- a fixed start lets an
        agent memorise one trajectory and learn nothing about place.
    mask_actions : bool
        Report the lidar action mask in `info['action_mask']` and refuse
        blocked actions. The mask is information the robot's own sensors
        provide, so using it is not a representation leak.
    """

    metadata = {'render_modes': []}

    def __init__(self, robot, observer, max_steps=200, teleport=True,
                 start_mode='training', start_index=-1, mask_actions=True,
                 reward=None):
        super().__init__()
        self.robot = robot
        self.observer = observer
        self.max_steps = int(max_steps)
        self.teleport = bool(teleport)
        self.start_mode = start_mode
        self.start_index = int(start_index)
        self.mask_actions = bool(mask_actions)
        self.reward_cfg = reward or RewardConfig()

        self.action_space = spaces.Discrete(N_ACTIONS)
        # Row-normalised activations are non-negative and sum to 1, so [0, 1]
        # per cell is exact rather than a guess. An ensemble observed with
        # normalize=None would need a different upper bound.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(observer.n_cells,), dtype=np.float32)

        self.current_step = 0
        self._prev_dist = None

        # Trajectory recording. The pose is already read every step for
        # `info`, so keeping the path costs nothing beyond the list. Paths
        # are what the final figure is made of: where the agent actually
        # went, from each start, which a learning curve cannot show.
        self.start_idx = -1
        self._path = []
        self._ep_reward = 0.0
        # start index -> the most recent completed episode from that start.
        self.last_paths = {}

    # ------------------------------------------------------------- helpers

    def n_starts(self):
        """How many start positions the arena defines, or 0 if unknown.

        The robot carries them on its parsed arena. A stand-in robot without
        one returns 0, and the env then leaves the choice to move_to_start.
        """
        try:
            locs = (self.robot.maze.training_starting_location
                    if self.start_mode == 'training'
                    else self.robot.maze.testing_start_location)
            return len(locs)
        except Exception:                                   # noqa: BLE001
            return 0

    def _action_mask(self):
        """Which of the 8 actions the lidar says are clear, as 0/1 float32."""
        if not self.mask_actions:
            return np.ones(N_ACTIONS, dtype=np.float32)
        return np.asarray(self.robot.get_possible_training_action_mask(),
                          dtype=np.float32)

    def _info(self, mask, blocked=False, terminated=False):
        x, y, heading = self.robot.get_robot_pose()
        return {
            'action_mask': mask,
            'dist_to_goal': float(self.robot.get_dist_to_goal()),
            'pose': (float(x), float(y), float(heading)),
            'blocked': bool(blocked),
            'steps': self.current_step,
            # Read by SB3's Monitor and by the learning-time analysis. Only
            # meaningful on the final step of an episode.
            'is_success': bool(terminated),
        }

    # ----------------------------------------------------------- gym API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        index = self.start_index
        if options and 'start_index' in options:
            index = int(options['start_index'])
        # Resolve -1 here rather than letting move_to_start pick, so the env
        # knows WHICH start this episode came from. Without that the paths
        # cannot be grouped by start, and the final figure has nothing to
        # put in its panels. Drawn from the env's seeded RNG, so a seed
        # still reproduces the whole run.
        n = self.n_starts()
        if index == -1 and n > 0:
            index = int(self.np_random.integers(n))
        self.robot.move_to_start(mode=self.start_mode, index=index)
        self.start_idx = index

        x, y, _ = self.robot.get_robot_pose()
        self._path = [(float(x), float(y))]
        self._ep_reward = 0.0

        self._prev_dist = float(self.robot.get_dist_to_goal())
        obs = self.observer.observe(self.robot)
        return obs, self._info(self._action_mask())

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f'action {action} outside {self.action_space}')
        self.current_step += 1
        cfg = self.reward_cfg

        mask = self._action_mask()
        blocked = self.mask_actions and mask[action] == 0

        if not blocked:
            if self.teleport:
                self.robot.perform_training_action_teleport(action)
            else:
                self.robot.perform_training_action(action)

        terminated = bool(self.robot.check_at_goal())
        dist = float(self.robot.get_dist_to_goal())

        reward = cfg.step
        if blocked:
            reward += cfg.blocked
        if terminated:
            reward += cfg.goal
        if cfg.shaping:
            # phi(s) = -distance, so approaching the goal is positive.
            reward += cfg.shaping * (
                cfg.shaping_gamma * (-dist) - (-self._prev_dist))
        self._prev_dist = dist

        # Truncation is the step budget running out, which Gymnasium keeps
        # separate from termination so a bootstrapping agent can tell "the
        # episode ended" from "the task was solved".
        truncated = (not terminated) and self.current_step >= self.max_steps

        # The observation is skipped on the last step of an episode only if
        # it were expensive to produce and unused -- it is not unused: SB3
        # bootstraps from the final observation on truncation.
        obs = self.observer.observe(self.robot)
        info = self._info(self._action_mask(), blocked=blocked,
                          terminated=terminated)

        self._path.append((info['pose'][0], info['pose'][1]))
        self._ep_reward += float(reward)
        if terminated or truncated:
            # Overwrites the previous episode from this start, so what is
            # kept is always the latest -- which is what the figure wants:
            # how the agent behaves NOW, not how it once did.
            self.last_paths[self.start_idx] = dict(
                start_idx=self.start_idx,
                xy=list(self._path),
                terminated=bool(terminated),
                truncated=bool(truncated),
                steps=self.current_step,
                reward=self._ep_reward,
                final_dist=info['dist_to_goal'],
            )
        return obs, float(reward), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        # The robot's lifetime belongs to whoever connected it -- the
        # controller process or the extern launcher -- so the env does not
        # tear down the simulation it was handed.
        pass


def action_masks(env):
    """Mask accessor in the shape sb3-contrib's MaskablePPO expects.

    Not a dependency: stable-baselines3 proper has no action masking, and
    MaskablePPO lives in sb3-contrib, which is not in requirements.txt. This
    is here so that wiring it up later is a one-line change rather than a
    rework of the env.
    """
    return env._action_mask().astype(bool)
