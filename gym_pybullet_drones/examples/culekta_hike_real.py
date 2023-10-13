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
import time
from datetime import datetime
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from tqdm import trange

from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

from culekta_utils import *

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
DEFAULT_OUTPUT_FOLDER = f'train_h0'
DEFAULT_COLAB = False
aligned_follower = True


#* Env Params
NUM_OBJECTS = 3
HS = [0.1]

#* Augmentation Params
EARLY_STOP = False
EARLY_STOP_FRAME = random.randint(73, 138)
SWITCH = False
NUM_SWITCHES = 30

CRITICAL_DIST = 0.5
CRITICAL_DIST_BUFFER = 0.1
FINISH_COUNTER_THRESHOLD = 32
# TURN_ANGLE = np.pi / 2

def run(
        loc_color_tuple,
        output_folder,
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        vision=DEFAULT_VISION,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        aggregate=DEFAULT_AGGREGATE,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        record_freq_hz=DEFAULT_RECORD_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        colab=DEFAULT_COLAB,
):

    #! Trajectory-specific parameters
    #* Env Params
    sim_name = "save-flight-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S.%f") # include milliseconds in save name for parallel runs
    sim_dir = os.path.join(output_folder, sim_name)
    setup_folders(sim_dir, num_drones)

    Theta = 0
    Theta0 = Theta
    Theta_offset = random.choice([0.175 * np.pi, -0.175 * np.pi])
    start_H = 0 #random.uniform(0.05, 0.25)
    print(f"start_H: {start_H}")

    #* Object setup
    obj_loc_global = [convert_to_global(obj_loc_rel, Theta) for obj_loc_rel in ordered_locs]
    TARGET_LOCATIONS = obj_loc_global
    print(f"TARGET_LOCATIONS: {TARGET_LOCATIONS}")

    target_index = 0

    #* Save starting env params
    with open(os.path.join(sim_dir, 'colors.txt'), 'w') as f:
        f.write(str("".join(ordered_objs)))
    
    # ! Initialize drone locations
    rel_drone_locs = [(0, 0)]

    FINAL_THETA = [angle_between_two_points(rel_drone, rel_obj) for rel_drone, rel_obj in zip(rel_drone_locs, ordered_locs)]
    INIT_XYZS = np.array([[*convert_to_global(rel_pos, Theta), start_H] for rel_pos in rel_drone_locs])
    INIT_RPYS = np.array([[0, 0, Theta0 + Theta_offset] for d in range(num_drones)])
    AGGR_PHY_STEPS = int(simulation_freq_hz / control_freq_hz) if aggregate else 1

    TARGET_POS = [[arry] for arry in INIT_XYZS]
    TARGET_ATT = [[arry] for arry in INIT_RPYS]
    INIT_THETA = [init_rpys[2] for init_rpys in INIT_RPYS]
    # angular distance between init and final theta
    DELTA_THETA = [signed_angular_distance(init_theta, final_theta + Theta) for final_theta, init_theta in zip(FINAL_THETA, INIT_RPYS[:, 2])]


    # ! Precompute Trajectories
    # usually, computed at 240 hz

    # hold for a second for stabilization
    frame_counter = 0
    finish_counter = 0
    previously_reached_critical = False
    start_dropping = False
    for i in range(240):
        reached_critical, speeds, TARGET_POS, TARGET_ATT, FINAL_THETA = step_hike(CRITICAL_DIST, CRITICAL_DIST_BUFFER, ordered_objs[target_index], INIT_THETA, FINAL_THETA, DELTA_THETA, TARGET_LOCATIONS, target_index, TARGET_POS, TARGET_ATT, Theta, control_freq_hz, HS, hold=True)

    # * Main trajectory is at 240 Hz
    while target_index < len(ordered_objs):
        frame_counter = 0
        finish_counter = 0
        previously_reached_critical = False
        start_dropping = False
        while finish_counter < FINISH_COUNTER_THRESHOLD * 30:
            reached_critical, speeds, TARGET_POS, TARGET_ATT, FINAL_THETA = step_hike(CRITICAL_DIST, CRITICAL_DIST_BUFFER, ordered_objs[target_index], INIT_THETA, FINAL_THETA, DELTA_THETA, TARGET_LOCATIONS, target_index, TARGET_POS, TARGET_ATT, Theta, control_freq_hz, HS, start_dropping)
            
            if reached_critical or previously_reached_critical:
                finish_counter += 1 if not ordered_objs[target_index] == "G" else 0.34
                previously_reached_critical = True
                start_dropping = True

            if EARLY_STOP and len(TARGET_POS[0]) > EARLY_STOP_FRAME * 30: # 73
                break

            frame_counter += 1
        target_index += 1
        if target_index < len(ordered_objs):
            cur_drone_pos = [convert_to_relative((TARGET_POS[d][-1][0], TARGET_POS[d][-1][1]), Theta) for d in range(num_drones)]

            FINAL_THETA = [angle_between_two_points(cur_drone_pos[0], ordered_locs[target_index])]
            print(f"ordered_locs[target_index]: {ordered_locs[target_index]}")
            TARGET_LOCATIONS = convert_array_to_global([ordered_locs[target_index]], Theta)
            INIT_THETA = [target_att[-1][2] for target_att in TARGET_ATT]
            DELTA_THETA = [signed_angular_distance(init_theta, final_theta + Theta) for final_theta, init_theta in zip(FINAL_THETA, INIT_THETA)]


    generate_debug_plots(sim_dir, TARGET_POS, TARGET_LOCATIONS, num_drones)
