import os
import sys

# Resolve the repo root from __file__ (same as collect_data) so the working
# directory and sys.path cannot disagree about where realm_tools lives.
REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
print(f"repo root: {REPO}", flush=True)

import pandas as pd
from PIL import Image

from realm_tools.robot_lib.my_robot import MyRobot
from realm_tools.simulation_lib.arena_paths import (arena_positions,
                                                     arena_xml)


MAZE = 'corr_lm8_l10w2'
MAZE_FILE = arena_xml(MAZE)
POSITIONS_FILE = arena_positions(MAZE)
OUTPUT_DIR = 'data_cache/camera_capture'

# Pick from the central box spanning this fraction of the arena on each axis.
CENTER_FRACTION = 0.2

# The same eight headings collect_data captures at every position.
THETAS = [0.0, 0.7854, 1.5708, 2.3562, 3.1416, 3.9270, 4.7124, 5.4978]


os.makedirs(OUTPUT_DIR, exist_ok=True)

robot = MyRobot()
robot.load_environment(MAZE_FILE, floor_texture='carpet')

positions = pd.read_csv(POSITIONS_FILE)
cx = (positions.x.min() + positions.x.max()) / 2
cy = (positions.y.min() + positions.y.max()) / 2
half_w = CENTER_FRACTION * (positions.x.max() - positions.x.min()) / 2
half_h = CENTER_FRACTION * (positions.y.max() - positions.y.min()) / 2
middle = positions[((positions.x - cx).abs() <= half_w)
                   & ((positions.y - cy).abs() <= half_h)]
pos = middle.sample(n=1).iloc[0]
print(f"position: x={pos.x:.3f}  y={pos.y:.3f}", flush=True)

robot.teleport_robot(x=pos.x, y=pos.y, theta=THETAS[0])
images, _, _, _ = robot.capture_pov_images(THETAS)

for theta, image in zip(THETAS, images):
    fname = f'{MAZE}_x{pos.x:.2f}_y{pos.y:.2f}_th{theta:.4f}.png'
    Image.fromarray(image).save(os.path.join(OUTPUT_DIR, fname))
    print(f"saved {fname}", flush=True)

robot.experiment_supervisor.simulationReset()
