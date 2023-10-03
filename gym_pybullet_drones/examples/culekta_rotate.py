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
DEFAULT_NUM_DRONES = 2
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
DEFAULT_OUTPUT_FOLDER = f'train_o4_rot_es10_540'
DEFAULT_COLAB = False
aligned_follower = True


#* Env Params
NUM_OBJECTS = 3
HS = [0.2, 0.1]

#* Augmentation Params
EARLY_STOP = False
EARLY_STOP_FRAME = random.randint(73, 138)
SWITCH = True
NUM_SWITCHES = 30



def run(
        loc_color_tuple,
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
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB,
):
    LCR_obj_colors, colors_array, LCR_obj_xy = loc_color_tuple

    #! Trajectory-specific parameters
    #* Env Params
    sim_name = "save-flight-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S.%f") # include milliseconds in save name for parallel runs
    sim_dir = os.path.join(output_folder, sim_name)
    setup_folders(sim_dir, num_drones)

    Theta = random.random() * 2 * np.pi
    Theta0 = Theta

    #* Augmentation Params
    RAND_SWITCH_TIMES = []
    if colors_array is None:
        colors_array = [random.sample(PERMUTATIONS_COLORS, 1)[0]]

    #? base_obj arrays are absolutely ordered [Left, Center, Right]
    #? base_obj_colors is therefore the corresponding colors of [Left, Center, Right]
    LCR_obj_loclabels = ['L', 'C', 'R']
    if LCR_obj_xy is None:
        OBJ_START_DIST = 1.5 #random.uniform(2.5, 3)
        LCR_obj_xy = [(OBJ_START_DIST, lateral_obj_dist) for lateral_obj_dist in [random.uniform(0.35, 0.5), random.uniform(-0.1, 0.1), random.uniform(-0.35, -0.5)]]

    #? array indices indicate which drone [Leader, Follower1, Follower2, ...]
    start_obj_locs = convert_color_array_to_location_array(colors_array[0], LCR_obj_xy, LCR_obj_colors)
    TARGET_LOCATIONS = convert_array_to_global(start_obj_locs, Theta)
    
    # ? Targets are encoded as 0-index for original, then other indices are switched locations
    target_index = 0
    target_locations = [start_obj_locs]
    target_colors = [colors_array[0]]

    #* Save starting env params
    with open(os.path.join(sim_dir, 'colors.txt'), 'w') as f:
        f.write(str("".join(colors_array[0])))
    
    # ! Initialize drone locations
    rel_drone_locs = [(0, 0) for i in range(num_drones)]

    FINAL_THETA = [angle_between_two_points(rel_drone, rel_obj) for rel_drone, rel_obj in zip(rel_drone_locs, start_obj_locs)]
    INIT_XYZS = np.array([[*convert_to_global(rel_pos, Theta), H] for rel_pos, H in zip(rel_drone_locs, HS)])
    INIT_RPYS = np.array([[0, 0, Theta0] for d in range(num_drones)])
    AGGR_PHY_STEPS = int(simulation_freq_hz / control_freq_hz) if aggregate else 1

    LABELS = []
    UNFILTERED_LABELS = [colors_array[0]]

    TARGET_POS = [[arry] for arry in INIT_XYZS]
    TARGET_ATT = [[arry] for arry in INIT_RPYS]
    INIT_THETA = [init_rpys[2] for init_rpys in INIT_RPYS]
    # angular distance between init and final theta
    DELTA_THETA = [signed_angular_distance(init_theta, final_theta + Theta) for final_theta, init_theta in zip(FINAL_THETA, INIT_RPYS[:, 2])]


    # ! Precompute Trajectories
    # usually, computed at 240 hz
    angle_counter = 0 # only updated after Starting Hold

    # * Hold for stabilization
    # for i in range(int(1 * control_freq_hz)):
    #     step_with_distance_and_angle(angle_counter, INIT_THETA, FINAL_THETA, DELTA_THETA, TARGET_LOCATIONS, alternate_index, hold=True)

    # * Main trajectory is at 240 Hz
    speeds = [DEFAULT_SPEED / control_freq_hz for i in range(num_drones)]
    while len(RAND_SWITCH_TIMES) < NUM_SWITCHES: #np.abs(np.array(speeds)).max() > 1e-6:
        speeds, aligned, UNFILTERED_LABELS, TARGET_POS, TARGET_ATT = step_with_distance_and_angle(INIT_THETA, FINAL_THETA, DELTA_THETA, TARGET_LOCATIONS, target_index, UNFILTERED_LABELS, TARGET_POS, TARGET_ATT, Theta, control_freq_hz, num_drones, target_colors, HS)
        angle_counter += 1

        if SWITCH and (np.array(angle_counter) % 30 == 0) and np.all(aligned) and random.random() < 0.8:
            target_colors, target_locations = add_random_targets(target_colors, target_locations, LCR_obj_xy, LCR_obj_colors)
            RAND_SWITCH_TIMES.append(angle_counter // 30)

            target_index += 1
            cur_drone_pos = [convert_to_relative((TARGET_POS[d][-1][0], TARGET_POS[d][-1][1]), Theta) for d in range(num_drones)]
            FINAL_THETA = [angle_between_two_points(cur_drone_pos[d], target_locations[target_index][d]) for d in range(num_drones)]
            TARGET_LOCATIONS = convert_array_to_global(target_locations[target_index], Theta)
            INIT_THETA = [target_att[-1][2] for target_att in TARGET_ATT]
            DELTA_THETA = [signed_angular_distance(init_theta, final_theta + Theta) for final_theta, init_theta in zip(FINAL_THETA, INIT_THETA)]


        if EARLY_STOP and len(TARGET_POS[0]) > EARLY_STOP_FRAME * 30: # 73
            break

    #* Hold for learning to stop
    if not EARLY_STOP:
        for i in range(int(END_HOLD_TIME * control_freq_hz)):
            speeds, aligned, UNFILTERED_LABELS, TARGET_POS, TARGET_ATT = step_with_distance_and_angle(INIT_THETA, FINAL_THETA, DELTA_THETA, TARGET_LOCATIONS, target_index, UNFILTERED_LABELS, TARGET_POS, TARGET_ATT, Theta, control_freq_hz, num_drones, target_colors, HS)
            angle_counter += 1

    assert target_index == len(target_locations) - 1, f"target_index: {target_index}, len(target_locations): {len(target_locations)}"

    wp_counters = np.array([0 for i in range(num_drones)])

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
                                "colors": LCR_obj_colors,
                                "locations": convert_array_to_global(LCR_obj_xy, Theta)
                            }
                         )
    env.IMG_RES = np.array([256, 144])

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
                
            LABELS.append(UNFILTERED_LABELS[wp_counters[0]])

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.TIMESTEP)

    with open(sim_dir + "/values.csv", 'wb') as out_file:
        np.savetxt(out_file, LABELS, delimiter=",", fmt='%s')
    processed_switch_times = [t - 8 for t in RAND_SWITCH_TIMES]
    with open(sim_dir + "/switchpoints.csv", 'wb') as out_file:
        np.savetxt(out_file, processed_switch_times, delimiter=",", fmt='%s')
    # with open(sim_dir + "/targets.csv", 'wb') as out_file:
    #     np.savetxt(out_file, logging_targets[8:], delimiter=",", fmt='%f,%f,%f')
    # with open(sim_dir + "/final_theta.csv", 'wb') as out_file:
    #     np.savetxt(out_file, logging_final_theta[8:], delimiter=",", fmt='%f')
    # with open(sim_dir + "/init_theta.csv", 'wb') as out_file:
    #     np.savetxt(out_file, logging_init_theta[8:], delimiter=",", fmt='%f')
    # with open(sim_dir + "/delta_theta.csv", 'wb') as out_file:
    #     np.savetxt(out_file, logging_delta_theta[8:], delimiter=",", fmt='%f')
    
    env.close()
    logger.save_as_csv(sim_name)  # Optional CSV save

    generate_debug_plots(sim_dir, TARGET_POS, TARGET_LOCATIONS, num_drones)
