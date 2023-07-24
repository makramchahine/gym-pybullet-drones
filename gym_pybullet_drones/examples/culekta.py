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
DEFAULT_NUM_DRONES = 1
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
DEFAULT_OUTPUT_FOLDER = f'train_v13_fast_1r_big_{DEFAULT_RECORD_FREQ_HZ}_750' # 'train_v11_fast_init_pp_60hz'
DEFAULT_COLAB = False

deviation_mode = "initial_deviation" # "random_walk" or "initial_deviation"

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
    H_STEP = .05
    R = 1.0
    sim_name = "save-flight-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S.%f") # include milliseconds in save name for parallel runs

    sim_dir = os.path.join(output_folder, sim_name)
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir + '/')

    # make initial positions and orientations of drones centered around a circle of radius R and height H around the origin (0,0,0)
    Theta = random.random() * 2 * np.pi
    Theta0 = Theta - np.pi
    Delta_Theta0 = random.choice([-np.pi * 0.175, np.pi * 0.175])
    INIT_XYZS = np.array([[R * np.cos(Theta), R * np.sin(Theta), H]])
    INIT_RPYS = np.array([[0, 0, Theta0 + Delta_Theta0]])
    AGGR_PHY_STEPS = int(simulation_freq_hz / control_freq_hz) if aggregate else 1

    #### Initialize a circular trajectory ######################
    # PERIOD = random.randint(16, 24) # * 8
    PERIOD = random.randint(16 * 2, 24 * 2) # * 8
    ACTIVE_WP = control_freq_hz * PERIOD
    INIT_WP = int(0 * ACTIVE_WP)
    TARGET_POS = np.zeros((ACTIVE_WP + INIT_WP, 3))
    TARGET_ATT = np.zeros((ACTIVE_WP + INIT_WP, 3))
    # random number taking values -1 or 1, indicating initial direction
    sign = random.choice([-1, 1])

    # stochastic parameters
    if random.random() < 0.5:
        SWITCH_WP = random.randint(int(np.floor(ACTIVE_WP / 4)), int(np.floor(3 * ACTIVE_WP / 4))) # when to switch directions (left or right)
    else:
        SWITCH_WP = ACTIVE_WP + 2 # never switch directions

    DEVIATION_A = random.uniform(0.5, 1.5) # initial deviation from the circular trajectory

    RECOVERY_WP = random.randint(int(np.floor(ACTIVE_WP / 2)), int(np.floor(ACTIVE_WP))) # when to recover from the deviation (achieve the desired radius)
    RECOVERY_ANGLE_WP = random.randint(int(np.floor(ACTIVE_WP / 2)), int(np.floor(ACTIVE_WP))) # when to recover from the angle deviation (achieve the desired radius)

    # Forward pass
    for i in range(ACTIVE_WP + INIT_WP):
        if i < INIT_WP:
            adj_R = R * DEVIATION_A
            adj_Delta_Theta = Delta_Theta0

            TARGET_POS[i, :] = (
                adj_R * np.cos((sign * i / ACTIVE_WP) * (2 * np.pi) + Theta), 
                adj_R * np.sin((sign * i / ACTIVE_WP) * (2 * np.pi) + Theta),
                0
            )
            TARGET_ATT[i, :] = 0, 0, (sign * i / ACTIVE_WP) * (2 * np.pi) + Theta0 + adj_Delta_Theta
        else:
            adj_i = i - INIT_WP
            # Handle deviation from radius
            if adj_i < RECOVERY_WP:
                adj_R = R * (1 + (((RECOVERY_WP - adj_i) / RECOVERY_WP)) * (DEVIATION_A - 1))
                # adj_R = R * (1 + (((RECOVERY_WP - adj_i) / RECOVERY_WP)) ** 2 * (DEVIATION_A - 1))
            else:
                adj_R = R

            if adj_i < RECOVERY_ANGLE_WP:
                adj_Delta_Theta = Delta_Theta0 * (((RECOVERY_ANGLE_WP - adj_i) / RECOVERY_ANGLE_WP))
                # adj_Delta_Theta = Delta_Theta0 * (((RECOVERY_ANGLE_WP - adj_i) / RECOVERY_ANGLE_WP)) ** 2
            else:
                adj_Delta_Theta = 0

            # Handle switching directions
            if adj_i < SWITCH_WP:
                TARGET_POS[i, :] = (
                    adj_R * np.cos((sign * i / ACTIVE_WP) * (2 * np.pi) + Theta), 
                    adj_R * np.sin((sign * i / ACTIVE_WP) * (2 * np.pi) + Theta),
                    0
                )
                TARGET_ATT[i, :] = 0, 0, (sign * i / ACTIVE_WP) * (2 * np.pi) + Theta0 + adj_Delta_Theta
            else:
                TARGET_POS[i, :] = (
                    adj_R * np.cos((sign * (2 * SWITCH_WP - adj_i + INIT_WP) / ACTIVE_WP) * (2 * np.pi) + Theta),
                    adj_R * np.sin((sign * (2 * SWITCH_WP - adj_i + INIT_WP) / ACTIVE_WP) * (2 * np.pi) + Theta),
                    0
                )
                TARGET_ATT[i, :] = 0, 0, (sign * (2 * SWITCH_WP - adj_i) / ACTIVE_WP) * (2 * np.pi) + Theta0 + adj_Delta_Theta

    wp_counters = np.array([int((i * (ACTIVE_WP + INIT_WP) / 6) % (ACTIVE_WP + INIT_WP)) for i in range(num_drones)])

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
                         user_debug_gui=user_debug_gui
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
    STEPS = int(CTRL_EVERY_N_STEPS * ACTIVE_WP)
    INIT_STEPS = int(CTRL_EVERY_N_STEPS * INIT_WP)
    # STEPS = CTRL_EVERY_N_STEPS * (max(RECOVERY_WP, RECOVERY_ANGLE_WP) + 1) # early
    LABEL = []

    env.reset()

    actions = []
    for i in trange(0, int(STEPS + INIT_STEPS), AGGR_PHY_STEPS):
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
                    target_pos=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2]]),
                    target_rpy=np.hstack([INIT_RPYS[j, 0:2], TARGET_ATT[wp_counters[j], 2]]),
                    )

            #### Go to the next way point and loop #####################
            for j in range(num_drones):
                wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (ACTIVE_WP + INIT_WP - 1) else 0


        if i % REC_EVERY_N_STEPS == 0 and i>env.SIM_FREQ and i>=INIT_STEPS:
            adj_i = i - INIT_STEPS
            rgb, dep, seg = env._getDroneImages(0)
            env._exportImage(img_type=ImageType.RGB,
                             img_input=rgb,
                             path=sim_dir,
                             frame_num=int(adj_i / REC_EVERY_N_STEPS),
                             )
            actions.append(action[str(j)])
            ### Log the simulation ####################################
            for j in range(num_drones):
                logger.log(drone=j,
                           timestamp=int(adj_i / REC_EVERY_N_STEPS),
                           state=obs[str(j)]["state"],
                           control=np.hstack(
                               [TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
                           )

                if wp_counters[j] < SWITCH_WP + INIT_WP:
                    LABEL.append(sign)
                else:
                    LABEL.append(-sign)

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.TIMESTEP)

    # Write the list LABEL as a csv
    with open(sim_dir + "/values.csv", 'wb') as out_file:
        np.savetxt(out_file, LABEL, delimiter=",")

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save_as_csv(sim_name)  # Optional CSV save

    debug_plots = True
    if debug_plots:
        # read log_0.csv and plot radius
        data = np.genfromtxt(sim_dir + "/log_0.csv", delimiter=",", skip_header=2)
        fig, ax = plt.subplots()
        ax.plot(data[:, 1], data[:, 2])
        # plot reference circle with radius R
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(R * np.cos(theta), R * np.sin(theta))
        ax.plot(TARGET_POS[:, 0], TARGET_POS[:, 1])
        ax.legend(["Drone", "Reference", "Target"])
        ax.set_aspect('equal', 'box')
        fig.savefig(sim_dir + "/path.jpg")

        fig, ax = plt.subplots()
        radius = np.sqrt(data[:, 1] **2 + data[:, 2] **2)
        ax.plot(data[:, 0], radius)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Radius (arb. units)")
        fig.savefig(sim_dir + "/radius.jpg")

    # csv_data = np.array(actions, dtype=np.float32)
    # csv_header = "t1,t2,t3,t4"
    # np.savetxt(os.path.join(output_folder, "torques.csv"), csv_data, delimiter=",", header=csv_header, comments="", fmt="%f")


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
