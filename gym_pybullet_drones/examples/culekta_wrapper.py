"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` or `VisionAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python fly.py

Notes
-----
The drones move, at different altitudes, along circular trajectories
in the X-Y plane, around point (0, 0).

"""
import os
import copy
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import itertools

from tqdm import tqdm
from functools import partial
import joblib

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync, str2bool

from culekta_hike import run
from culekta_utils import PERMUTATIONS_COLORS

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_VISION = False
DEFAULT_GUI = False
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 240
DEFAULT_RECORD_FREQ_HZ = 8
DEFAULT_DURATION_SEC = 8
DEFAULT_COLAB = False
aligned_follower = True
samples = 6
fbal = False
DEFAULT_OUTPUT_FOLDER = f'train_holodeck2_h0f_hr_{"fbal_" if fbal else ""}{samples}'
multi = False

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone', default=DEFAULT_DRONES, type=DroneModel, help='Drone model (default: CF2X)',
                        metavar='', choices=DroneModel)
    parser.add_argument('--num_drones', default=DEFAULT_NUM_DRONES, type=int, help='Number of drones (default: 3)',
                        metavar='')
    parser.add_argument('--physics', default=DEFAULT_PHYSICS, type=Physics, help='Physics updates (default: PYB)',
                        metavar='', choices=Physics)
    parser.add_argument('--vision', default=DEFAULT_VISION, type=str2bool,
                        help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Whether to use PyBullet GUI (default: True)',
                        metavar='')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VISION, type=str2bool,
                        help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot', default=DEFAULT_PLOT, type=str2bool,
                        help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui', default=DEFAULT_USER_DEBUG_GUI, type=str2bool,
                        help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate', default=DEFAULT_AGGREGATE, type=str2bool,
                        help='Whether to aggregate physics steps (default: True)', metavar='')
    parser.add_argument('--obstacles', default=DEFAULT_OBSTACLES, type=str2bool,
                        help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int,
                        help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz', default=DEFAULT_CONTROL_FREQ_HZ, type=int,
                        help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec', default=DEFAULT_DURATION_SEC, type=int,
                        help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
                        help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool,
                        help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()
    
    NUM_INITIALIZATIONS = (samples // 5) if fbal else (samples // 3)
    OBJECTS = ["R", "R", "G", "B", "B"] if fbal else ["R", "G", "B"]
    # OBJECTS = ["R", "G", "B"]
    # TOTAL_OBJECTS = OBJECTS
    TOTAL_OBJECTS = [list(perm) for perm in itertools.combinations_with_replacement(OBJECTS, 2)] + [list(perm) for perm in itertools.combinations_with_replacement(OBJECTS, 1)]

    if multi:
        TOTAL_OBJECTS = TOTAL_OBJECTS * (NUM_INITIALIZATIONS)
        LOCATIONS_REL = []
        for targets in TOTAL_OBJECTS:
            locations = []
            cur_point = (0, 0) #random.uniform(0.75, 1.5)
            cur_direction = 0 
            for target in targets:
                cur_dist = random.uniform(1.0, 2.0) - 0.2
                target_loc = (cur_point[0] + (cur_dist + 0.2) * math.cos(cur_direction), cur_point[1] + (cur_dist + 0.2) * math.sin(cur_direction))
                cur_point = (cur_point[0] + cur_dist * math.cos(cur_direction), cur_point[1] + cur_dist * math.sin(cur_direction))
                locations.append(target_loc)

                if target[0] == 'R':
                    cur_direction += math.pi / 2
                elif target[0] == 'G':
                    cur_direction += 0
                elif target[0] == 'B':
                    cur_direction += -math.pi / 2
            LOCATIONS_REL.append(locations)
    else:
        TOTAL_OBJECTS = OBJECTS * NUM_INITIALIZATIONS
        LOCATIONS_REL = [[(random.uniform(1, 2), 0)] for _ in range(len(TOTAL_OBJECTS))]


    total_list = []
    for i, (obj, loc) in enumerate(zip(TOTAL_OBJECTS, LOCATIONS_REL)):
        total_list.append((obj, loc))
    assert len(total_list) == NUM_INITIALIZATIONS * (5 if fbal else 3), f"len(total_list): {len(total_list)}"
    random.shuffle(total_list)

    run_func = partial(run, **vars(ARGS))

    futures = []
    returns = []
    joblib.Parallel(n_jobs=16)(joblib.delayed(run_func)(d) for d in tqdm(total_list))
