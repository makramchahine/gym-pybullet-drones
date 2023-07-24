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

from tqdm import trange

from scipy.signal import butter, filtfilt
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 2
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_VISION = False
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 240
DEFAULT_RECORD_FREQ_HZ = 8
DEFAULT_DURATION_SEC = 40
DEFAULT_OUTPUT_FOLDER = f'train_ff1_align2_swap_delay_{DEFAULT_RECORD_FREQ_HZ}' # 'train_v11_fast_init_pp_60hz'
DEFAULT_COLAB = False
aligned_follower = True

def run(
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
        colab=DEFAULT_COLAB
):
    #### Initialize the simulation #############################
    H = .1
    sim_name = "save-flight-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S.%f") # include milliseconds in save name for parallel runs

    sim_dir = os.path.join(output_folder, sim_name)
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir + '/')
    d = 1
    if not os.path.exists(sim_dir + f"/pics{d}"):
        os.makedirs(sim_dir + f"/pics{d}/")


    def signed_angular_distance(theta1, theta2):
        # source to target
        # If the result is positive, it means you would need to rotate counter-clockwise from theta1 to theta2. If the result is negative, it means you would need to rotate clockwise.
        return ((theta2 - theta1 + np.pi) % (2 * np.pi)) - np.pi

    Theta = random.random() * 2 * np.pi
    Theta0 = random.random() * 2 * np.pi # drone faces the direction of theta
    Delta_Theta = signed_angular_distance(Theta, Theta0)

    if random.random() < 0.2:
        OBJ_START_DIST = random.uniform(0.15, 0.5)
    else:
        OBJ_START_DIST = random.uniform(0.5, 2)
    INIT_XYZS = np.array([[np.cos(Theta) * -0.5 * (1-d), np.sin(Theta) * -0.5 * (1-d), H] for d in range(num_drones)])
    INIT_RPYS = np.array([[0, 0, Theta0], [0, 0, Theta]]) if aligned_follower else np.array([[0, 0, Theta0] for d in range(num_drones)])
    AGGR_PHY_STEPS = int(simulation_freq_hz / control_freq_hz) if aggregate else 1

    # Object distance from D0: 0.5-1.5
    CUSTOM_OBJECT_LOCATION = np.cos(Theta) * OBJ_START_DIST, np.sin(Theta) * OBJ_START_DIST
    # Stopping distance = 0.7 to 0.3
    STOPPING_START = 1.0
    STOPPING_END = 0.5
    # D0 distance from D1: 0.4-0.6
    DRONE_DIST = 0.5
    # Hold stop for 0.5 seconds
    HOLD_TIME = 0.5

    DEFAULT_SPEED = 0.15 # TILES / S
    DEFAULT_BACKSPEED = 0.03 # TILES / S
    MIN_SPEED = 0.01
    ANGLE_RECOVERY_TIMESTEPS = random.uniform(2, 10) * control_freq_hz
    SEARCHING_YAW = 0.1 * np.pi

    cur_object_dist = OBJ_START_DIST
    TARGET_POS = [[[np.cos(Theta) * -0.5 * (1-d), np.sin(Theta) * -0.5 * (1-d), H]] for d in range(num_drones)]
    TARGET_ATT = [[[0, 0, Theta0]], [[0, 0, Theta]]] if aligned_follower else [[0, 0, Theta0] for d in range(num_drones)]

    def step_with_angle(TARGET_POS, TARGET_ATT, cur_delta_theta):
        TARGET_POS[0].append(TARGET_POS[0][-1])
        TARGET_POS[1].append(TARGET_POS[1][-1])

        cur_delta_theta = cur_delta_theta + SEARCHING_YAW / control_freq_hz if cur_delta_theta < 0 else cur_delta_theta - SEARCHING_YAW / control_freq_hz
            
        TARGET_ATT[0].append([0, 0, Theta + cur_delta_theta])
        TARGET_ATT[1].append([0, 0, Theta0 if not aligned_follower else Theta])

        return cur_delta_theta

    def step_with_distance_and_angle(TARGET_POS, TARGET_ATT, rad_offset_init_0, rad_offset_init_1, counter, dist_from_origin, follower_dist_from_origin):
        TARGET_POS[0].append([np.cos(Theta) * dist_from_origin, np.sin(Theta) * dist_from_origin, H])
        TARGET_POS[1].append([np.cos(Theta) * follower_dist_from_origin, np.sin(Theta) * follower_dist_from_origin, H])

        adj_theta_0 = rad_offset_init_0 * (1 - (counter / ANGLE_RECOVERY_TIMESTEPS)) if counter < ANGLE_RECOVERY_TIMESTEPS else 0
        adj_theta_1 = rad_offset_init_1 * (1 - (counter / ANGLE_RECOVERY_TIMESTEPS)) if counter < ANGLE_RECOVERY_TIMESTEPS else 0
        TARGET_ATT[0].append([0, 0, Theta + adj_theta_0])
        TARGET_ATT[1].append([0, 0, Theta if aligned_follower else Theta + adj_theta_1])

    
    # Locating Target (turning without target in view)
    cur_delta_theta = Delta_Theta
    while abs(cur_delta_theta) > np.pi * 0.05:
        cur_delta_theta = step_with_angle(TARGET_POS, TARGET_ATT, cur_delta_theta)
    
    # Approaching Target at full speed, while turning to center target
    angle_counter = 0
    while cur_object_dist > STOPPING_START:
        cur_object_dist -= DEFAULT_SPEED / control_freq_hz

        follower_dist_from_origin = OBJ_START_DIST - cur_object_dist 
        dist_from_origin = follower_dist_from_origin - DRONE_DIST
        
        step_with_distance_and_angle(TARGET_POS, TARGET_ATT, cur_delta_theta, Delta_Theta, angle_counter, dist_from_origin, follower_dist_from_origin)
        angle_counter += 1
    # Slowing down at target
    while cur_object_dist > STOPPING_END:
        speed = DEFAULT_SPEED * (cur_object_dist - STOPPING_END) / (STOPPING_START - STOPPING_END)
        speed = np.max([speed, MIN_SPEED])

        cur_object_dist -= speed / control_freq_hz
        follower_dist_from_origin = OBJ_START_DIST - cur_object_dist 
        dist_from_origin = follower_dist_from_origin - DRONE_DIST

        step_with_distance_and_angle(TARGET_POS, TARGET_ATT, cur_delta_theta, Delta_Theta, angle_counter, dist_from_origin, follower_dist_from_origin)
        angle_counter += 1
    # Backing up if too close to target
    while cur_object_dist < STOPPING_END:
        speed = DEFAULT_BACKSPEED * (STOPPING_END - cur_object_dist) / STOPPING_END
        speed = np.max([speed, MIN_SPEED])

        cur_object_dist += speed / control_freq_hz
        follower_dist_from_origin = OBJ_START_DIST - cur_object_dist 
        dist_from_origin = follower_dist_from_origin - DRONE_DIST

        step_with_distance_and_angle(TARGET_POS, TARGET_ATT, cur_delta_theta, Delta_Theta, angle_counter, dist_from_origin, follower_dist_from_origin)
        angle_counter += 1
    # Holding position
    for i in range(int(HOLD_TIME * control_freq_hz)):
        step_with_distance_and_angle(TARGET_POS, TARGET_ATT, cur_delta_theta, Delta_Theta, angle_counter, dist_from_origin, follower_dist_from_origin)
        angle_counter += 1


    wp_counters = np.array([0 for i in range(num_drones)])

    #### Create the environment with or without video capture ##
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
                         custom_obj_location=CUSTOM_OBJECT_LOCATION
                         )
    env.IMG_RES = np.array([256, 144])

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=4,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]
    elif drone in [DroneModel.HB]:
        ctrl = [SimplePIDControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ / control_freq_hz)) # 1
    REC_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ / record_freq_hz )) #30 #240
    action = {str(i): np.array([0, 0, 0, 0]) for i in range(num_drones)}
    START = time.time()
    ACTIVE_WP = len(TARGET_POS[0])
    TARGET_POS = np.array(TARGET_POS)
    TARGET_ATT = np.array(TARGET_ATT)
    STEPS = int(CTRL_EVERY_N_STEPS * ACTIVE_WP)

    env.reset()

    actions = []
    for i in trange(0, int(STEPS), AGGR_PHY_STEPS):
        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)

        #### Compute control at the desired frequency ##############
        if i % CTRL_EVERY_N_STEPS == 0:

            #### Compute control for the current way point #############
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
                wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (ACTIVE_WP - 1) else 0


        if i % REC_EVERY_N_STEPS == 0 and i>env.SIM_FREQ:
            for d in range(num_drones):
                rgb, dep, seg = env._getDroneImages(d)
                env._exportImage(img_type=ImageType.RGB,
                                img_input=rgb,
                                path=sim_dir if d==0 else sim_dir + f"/pics{d}",
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


    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save_as_csv(sim_name)  # Optional CSV save

    debug_plots = True
    if debug_plots:
        # read log_0.csv and plot radius
        data = np.genfromtxt(sim_dir + "/log_0.csv", delimiter=",", skip_header=2)
        data2 = np.genfromtxt(sim_dir + "/log_1.csv", delimiter=",", skip_header=2)
        fig, ax = plt.subplots()
        # ax.plot(data[:, 1], data[:, 2])
        # plot with transparency
        ax.plot(data[:, 1], data[:, 2], alpha=0.9)
        ax.plot(data2[:, 1], data2[:, 2], alpha=0.9)
        ax.plot(TARGET_POS[0, :, 0], TARGET_POS[0, :, 1], alpha=0.9)
        ax.plot(CUSTOM_OBJECT_LOCATION[0], CUSTOM_OBJECT_LOCATION[1], 'ro')
        ax.legend(["Leader", "Follower", "Lead Target", "Obj"])
        
        # ax.set_aspect('equal', 'box')
        fig.savefig(sim_dir + "/path.jpg")

        fig, ax = plt.subplots()
        radius = np.sqrt(data[:, 1] **2 + data[:, 2] **2)
        ax.plot(data[:, 0], radius)
        radius2 = np.sqrt(data2[:, 1] **2 + data2[:, 2] **2)
        ax.plot(data2[:, 0], radius2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Radius (arb. units)")
        fig.savefig(sim_dir + "/radius.jpg")

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()


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

    run(**vars(ARGS))
