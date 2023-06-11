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
DEFAULT_DURATION_SEC = 20
DEFAULT_OUTPUT_FOLDER = 'train_v6'
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
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
):
    #### Initialize the simulation #############################
    H = .1
    H_STEP = .05
    R = .5
    sim_name = "save-flight-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S.%f") # include milliseconds in save name for parallel runs

    sim_dir = os.path.join(output_folder, sim_name)
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir + '/')

    # make initial positions and orientations of drones centered around a circle of radius R and height H around the origin (0,0,0)
    Theta = random.random() * 2 * np.pi
    Theta0 = Theta - np.pi
    INIT_XYZS = np.array([[R * np.cos(Theta), R * np.sin(Theta), H]])
    INIT_RPYS = np.array([[0, 0, Theta0]])
    AGGR_PHY_STEPS = int(simulation_freq_hz / control_freq_hz) if aggregate else 1

    #### Initialize a circular trajectory ######################
    PERIOD = random.randint(16, 24) * 8
    NUM_WP = control_freq_hz * PERIOD
    TARGET_POS = np.zeros((NUM_WP, 3))
    TARGET_ATT = np.zeros((NUM_WP, 3))
    # random number taking values -1 or 1, indicating initial direction
    sign = random.choice([-1, 1])

    # stochastic parameters
    if random.random() < 0.5:
        SWITCH_WP = random.randint(int(np.floor(NUM_WP / 4)), int(np.floor(3 * NUM_WP / 4))) # when to switch directions (left or right)
    else:
        SWITCH_WP = NUM_WP + 2 # never switch directions
    RECOVERY_WP = random.randint(int(np.floor(NUM_WP / 4)), int(np.floor(3 * NUM_WP / 4))) # when to recover from the deviation (achieve the desired radius)
    DEVIATION_A = random.uniform(0.8, 1.2) # initial deviation from the circular trajectory

    # Forward pass
    for i in range(NUM_WP):
        # Handle deviation from radius
        if i < RECOVERY_WP:
            adj_R = R * (1 + ((RECOVERY_WP - i) / RECOVERY_WP) * (DEVIATION_A - 1))
        else:
            adj_R = R
        
        # Handle switching directions
        if i < SWITCH_WP:
            TARGET_POS[i, :] = (
                adj_R * np.cos((sign * i / NUM_WP) * (2 * np.pi) + Theta), 
                adj_R * np.sin((sign * i / NUM_WP) * (2 * np.pi) + Theta),
                0
            )
            TARGET_ATT[i, :] = 0, 0, (sign * i / NUM_WP) * (2 * np.pi) + Theta0
        else:
            TARGET_POS[i, :] = (
                adj_R * np.cos((sign * (2 * SWITCH_WP - i) / NUM_WP) * (2 * np.pi) + Theta),
                adj_R * np.sin((sign * (2 * SWITCH_WP - i) / NUM_WP) * (2 * np.pi) + Theta),
                0
            )
            TARGET_ATT[i, :] = 0, 0, (sign * (2 * SWITCH_WP - i) / NUM_WP) * (2 * np.pi) + Theta0

    wp_counters = np.array([int((i * NUM_WP / 6) % NUM_WP) for i in range(num_drones)])

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
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ / control_freq_hz))
    REC_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ / 8 * 8))
    action = {str(i): np.array([0, 0, 0, 0]) for i in range(num_drones)}
    START = time.time()
    STEPS = CTRL_EVERY_N_STEPS * NUM_WP
    LABEL = []

    for i in trange(0, int(STEPS), AGGR_PHY_STEPS):
        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)

        #### Compute control at the desired frequency ##############
        if i % CTRL_EVERY_N_STEPS == 0:

            #### Compute control for the current way point #############
            for j in range(num_drones):

                action[str(j)], _, _ = ctrl[j].computeControlFromState(
                    control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                    state=obs[str(j)]["state"],
                    target_pos=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2]]),
                    target_rpy=np.hstack([INIT_RPYS[j, 0:2], TARGET_ATT[wp_counters[j], 2]]),
                    )

            #### Go to the next way point and loop #####################
            for j in range(num_drones):
                wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP - 1) else 0


        if i % REC_EVERY_N_STEPS == 0 and i>env.SIM_FREQ:
            rgb, dep, seg = env._getDroneImages(0)
            env._exportImage(img_type=ImageType.RGB,
                             img_input=rgb,
                             path=sim_dir,
                             frame_num=int(i / REC_EVERY_N_STEPS),
                             )
            ### Log the simulation ####################################
            for j in range(num_drones):
                logger.log(drone=j,
                           timestamp=int(i / REC_EVERY_N_STEPS),
                           state=obs[str(j)]["state"],
                           control=np.hstack(
                               [TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
                           )

                if wp_counters[j] < SWITCH_WP:
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
        plt.plot(data[:, 1], data[:, 2])
        # plot reference circle with radius R
        theta = np.linspace(0, 2 * np.pi, 100)
        plt.plot(R * np.cos(theta), R * np.sin(theta))
        plt.plot(TARGET_POS[:, 0], TARGET_POS[:, 1])
        plt.legend(["Drone", "Reference", "Target"])
        plt.savefig(sim_dir + "/path.jpg")
        plt.close()
        plt.close()
        plt.close()

        radius = np.sqrt(data[:, 1] **2 + data[:, 2] **2)
        plt.plot(data[:, 0], radius)
        plt.xlabel("Time (s)")
        plt.ylabel("Radius (arb. units)")
        plt.savefig(sim_dir + "/radius.jpg")

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
