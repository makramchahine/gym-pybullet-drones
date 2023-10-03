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
    ordered_objs, ordered_locs = loc_color_tuple

    #! Trajectory-specific parameters
    #* Env Params
    sim_name = "save-flight-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S.%f") # include milliseconds in save name for parallel runs
    sim_dir = os.path.join(output_folder, sim_name)
    setup_folders(sim_dir, num_drones)

    Theta = random.random() * 2 * np.pi
    Theta0 = Theta
    Theta_offset = random.choice([0.175 * np.pi, -0.175 * np.pi])
    start_H = random.uniform(0.05, 0.25)
    print(start_H)

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

    wp_counters = np.array([0 for i in range(num_drones)])

    print(f"self.INIT_XYZS: {INIT_XYZS.shape}")

    #! Simulation Setup
    if vision:
        env = VisionAviary(drone_model=drone,
                           num_drones=num_drones,
                           initial_xyzs=INIT_XYZS,
                           initial_rpys=INIT_RPYS,
                           physics=physics,
                           neighbourhood_radius=10,
                           freq=simulation_freq_hz,
                           aggregate_phy_steps=AGGR_PHY_STEPS,
                           gui=gui,
                           record=record_video,
                           obstacles=obstacles
                           )
    else:
        env = CtrlAviary(drone_model=drone,
                         num_drones=num_drones,
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPYS,
                         physics=physics,
                         neighbourhood_radius=10,
                         freq=simulation_freq_hz,
                         aggregate_phy_steps=AGGR_PHY_STEPS,
                         gui=gui,
                         record=record_video,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         custom_obj_location={
                                "colors": ordered_objs,
                                "locations": obj_loc_global
                            }
                         )
    env.IMG_RES = np.array([256, 144])

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    logger = Logger(logging_freq_hz=4,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]
    elif drone in [DroneModel.HB]:
        ctrl = [SimplePIDControl(drone_model=drone) for i in range(num_drones)]

    #! Simulation Params
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ / control_freq_hz)) # 1
    REC_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ / record_freq_hz )) #30 #240
    action = {str(i): np.array([0, 0, 0, 0]) for i in range(num_drones)}
    START = time.time()
    ACTIVE_WP = len(TARGET_POS[0])
    TARGET_POS = np.array(TARGET_POS)
    TARGET_ATT = np.array(TARGET_ATT)
    STEPS = int(CTRL_EVERY_N_STEPS * ACTIVE_WP)
    env.reset()

    #! Stepping Simulation
    # main loop at 240hz
    actions = []
    for i in trange(0, int(STEPS), AGGR_PHY_STEPS):
        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)

        #* Compute step-by-step velocities for 240hz-trajectory; Control Frequency is 240hz
        if i % CTRL_EVERY_N_STEPS == 0:
            for j in range(num_drones):
                state = obs[str(j)]["state"]
                action[str(j)], _, _ = ctrl[j].computeControlFromState(
                    control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                    state=state,
                    target_pos = TARGET_POS[j, wp_counters[j]],
                    target_rpy = TARGET_ATT[j, wp_counters[j]]
                    )

            #### Go to the next way point and loop #####################
            for j in range(num_drones):
                wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (ACTIVE_WP - 1) else wp_counters[j]

        #* Network Frequency is 30hz
        if i % REC_EVERY_N_STEPS == 0 and i>env.SIM_FREQ:
            for d in range(num_drones):
                rgb, dep, seg = env._getDroneImages(d)
                env._exportImage(img_type=ImageType.RGB,
                                img_input=rgb,
                                path=sim_dir + f"/pics{d}",
                                frame_num=int(i / REC_EVERY_N_STEPS),
                                )
                actions.append(action[str(d)])
                ### Log the simulation ####################################
                logger.log(drone=d,
                           timestamp=int(i / REC_EVERY_N_STEPS),
                           state=obs[str(d)]["state"],
                           control=np.hstack(
                               [TARGET_POS[d, wp_counters[d], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
                           )

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.TIMESTEP)
    
    env.close()
    logger.save_as_csv(sim_name)  # Optional CSV save

    generate_debug_plots(sim_dir, TARGET_POS, TARGET_LOCATIONS, num_drones)
