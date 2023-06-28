"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` or `VisionAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python fly.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories
in the X-Y plane, around point (0, 0).

"""
import os
import sys
import time
import argparse
from datetime import datetime
import pdb
import csv
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import cv2

from tqdm import trange
from PIL import Image

from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.out_2_way import out2way
from gym_pybullet_drones.utils.utils import sync, str2bool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "..", "..", ".."))
from drone_communication.utils.model_utils import load_model_from_weights, generate_hidden_list, get_readable_name, \
    get_params_from_json
from drone_communication.keras_models import IMAGE_SHAPE

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
DEFAULT_CONTROL_FREQ_HZ = 60 #int(240/8)
DEFAULT_DURATION_SEC = 20
DEFAULT_OUTPUT_FOLDER = 'replay_debug_base'
DEFAULT_COLAB = False
# DEFAULT_CSV_PATH = f'{DEFAULT_OUTPUT_FOLDER}/data_out.csv'
# DEFAULT_CSV_PATH = f'/home/makramchahine/repos/drone_multimodal/clean_replay_debug_base/save-flight-06.27.2023_15.07.06.384563/data_out.csv'
DEFAULT_CSV_PATH = f'/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/replay_debug_base/torques.csv'
mod = "torque"

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
        colab=DEFAULT_COLAB,
        csv_path = DEFAULT_CSV_PATH,
        convert_video = False
):

    # read csv at csv_path as np array, first row is header
    csv_data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)

    #### Initialize the simulation #############################
    H = .1
    R = .5
    sim_name = "run-flight-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    sim_dir = os.path.join(output_folder, sim_name)
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir + '/')

    # make initial positions and orientations of drones centered around a circle of radius R and height H around the origin (0,0,0)
    Theta = np.pi #random.random() * 2 * np.pi
    Theta0 = Theta - np.pi
    Delta_Theta0 = 0 # random.choice([-np.pi * 0.175, np.pi * 0.175])
    INIT_XYZS = np.array([[R * np.cos(Theta), R * np.sin(Theta), H]])
    INIT_RPYS = np.array([[0, 0, Theta0 + Delta_Theta0]])
    AGGR_PHY_STEPS = int(simulation_freq_hz / control_freq_hz) if aggregate else 1

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
    logger = Logger(logging_freq_hz=control_freq_hz,
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
    action = {str(i): np.array([0, 0, 0, 0]) for i in range(num_drones)}
    START = time.time()
    STEPS = CTRL_EVERY_N_STEPS * control_freq_hz * DEFAULT_DURATION_SEC 
    # 1 * 240 * 10 = 2400
    # 8 * 30 * 10 = 2400

    counter = 0

    waypoints = []
    control_inputs = []
    vels_states = []
    yaw_states = []
    ixigrec = []

    print(f"env.TIMESTEP: {env.TIMESTEP}")

    for i in trange(0, int(STEPS), AGGR_PHY_STEPS):
        obs, reward, done, info = env.step(action) #### Step the simulation ###################################

        # change control command every n steps
        if i % CTRL_EVERY_N_STEPS == 0: #### Compute control at the desired frequency ##############
            for j in range(num_drones): #### Compute control for the current way point #############
                rgb, dep, seg = env._getDroneImages(j)
                env._exportImage(img_type=ImageType.RGB,
                                 img_input=rgb,
                                 path=sim_dir,
                                 frame_num=int(i / CTRL_EVERY_N_STEPS),
                                 )
                try:
                    vel_cmd = csv_data[counter]
                    # multiply by sine wave with period 400:
                    # amp = 10
                    # vel_cmd[0] = amp * vel_cmd[0] * np.sin(2 * np.pi * counter / 400)
                    # vel_cmd[1] = amp * vel_cmd[1] * np.sin(2 * np.pi * counter / 400)
                    counter += 1
                except IndexError:
                    print("IndexError, using extreme command")
                    # vel_cmd = [1, 0, 0, 0.1]
                action[str(j)] = vel_cmd
                # vel_cmd = np.array([0, 0.1, 0, 0.0])
            
        # recompute simulation control otherwise
        for j in range(num_drones):
            state = obs[str(j)]["state"]

            # convert from body_frame to world_frame + integrate for target waypoint position
            # waypoint = out2way(vel_cmd, state, 1/control_freq_hz)
            # waypoint = out2way(vel_cmd, state, 1/simulation_freq_hz)
            # waypoints.append(waypoint[0:2])
            ixigrec.append(state[0:2])
            control_inputs.append(vel_cmd)
            yaw_states.append(state[9])

            vel_state = state[10:13].copy()
            # convert from world_frame to body_frame
            vel_state[0] = vel_state[0] * np.cos(state[9]) - vel_state[1] * np.sin(state[9])
            vel_state[1] = vel_state[0] * np.sin(state[9]) + vel_state[1] * np.cos(state[9])
            vels_states.append(vel_state)

            # # TODO: replay with dumped torque powers - check environment
            # # TODO: play around computeControl to match torque powers
            # # TODO: look into drone environment initialization
            # # look at dyn
            # # look at documentation
            # # look at usage of compute control  ``
            # action[str(j)], _, _ = ctrl[j].computeControl(control_timestep=env.TIMESTEP, # * CTRL_EVERY_N_STEPS,
            #                                                 cur_pos=state[0:3],
            #                                                 cur_quat=state[3:7],
            #                                                 cur_vel=state[10:13],
            #                                                 cur_ang_vel=state[13:16],
            #                                                 target_pos=state[0:3],  # same as the current position
            #                                                 target_rpy=np.array([0, 0, state[9]]),  # keep current yaw
            #                                                 target_vel=vel_cmd[0:3],
            #                                                 target_rpy_rates = np.array([0, 0, vel_cmd[3]])
            # )
            # print(f"action: {action[str(j)]}")

            logger.log(drone=j,
                        timestamp=int(i / CTRL_EVERY_N_STEPS),
                        state=obs[str(j)]["state"],
                        # control=waypoint
                        )

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save_as_csv(sim_name)  # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

    if convert_video:
        os.system(f"ffmpeg -framerate 60 -pattern_type glob -i '{sim_dir}/0*.png' -c:v libx264 -pix_fmt yuv420p {sim_dir}/rand.mp4")
        os.system(f"cp {sim_dir}/rand.mp4 {output_folder}/rand_{mod}.mp4")

    # save a plot of the waypoints (column 0 vs column 1, square axis)
    # overlay a plot of ixigrec with same layout in different color
    # make both plots have increasing marker size with index
    fig, axs = plt.subplots(1, 2)
    t = np.linspace(0, 1, len(ixigrec))  # time variable

    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm
    def plot_color_line(fig, ax, x, y, t):
        # Create a set of line segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(t.min(), t.max())
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        # Set the values used for colormapping
        lc.set_array(t)
        lc.set_linewidth(2)

        line = ax.add_collection(lc)
        fig.colorbar(line, ax=ax)

    plot_color_line(fig, axs[0], np.array(ixigrec)[:, 0], np.array(ixigrec)[:, 1], t)
    # plot_color_line(fig, axs[1], np.array(waypoints)[:, 0], np.array(waypoints)[:, 1], t)
    min_x = np.array(ixigrec)[:, 0].min()
    # min_x = min(np.array(ixigrec)[:, 0].min(), np.array(waypoints)[:, 0].min())
    max_x = np.array(ixigrec)[:, 0].max()
    # max_x = max(np.array(ixigrec)[:, 0].max(), np.array(waypoints)[:, 0].max())
    min_y = np.array(ixigrec)[:, 1].min()
    # min_y = min(np.array(ixigrec)[:, 1].min(), np.array(waypoints)[:, 1].min())
    max_y = np.array(ixigrec)[:, 1].max()
    # max_y = max(np.array(ixigrec)[:, 1].max(), np.array(waypoints)[:, 1].max())
    for ax in axs:
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
    axs[0].legend(["drone"])
    # axs[1].legend(["waypoints"])

    # axs[0].scatter(np.array(ixigrec)[:, 0], np.array(ixigrec)[:, 1], c=t, cmap='viridis')
    # im = axs[1].scatter(np.array(waypoints)[:, 0], np.array(waypoints)[:, 1], c=t, cmap='viridis')
    # axs[0].set_aspect('equal', 'box')
    # axs[1].set_aspect('equal', 'box')
    # fig.colorbar(im, ax=axs[1])


    # fig.suptitle(f'Waypoints @ {control_freq_hz}Hz')
    # fig.savefig(f'{output_folder}/waypoint_{control_freq_hz}.png')
    fig.suptitle(f'Waypoints @ {simulation_freq_hz}:{control_freq_hz}Hz')
    fig.savefig(f'{output_folder}/waypoint_{mod}_{simulation_freq_hz}_{control_freq_hz}.png')

    plt.close()
    plt.close()
    
    # plot vels and yaw
    # fig, axs = plt.subplots(4, 1)
    # for i in range(3):
    #     axs[i].plot(np.array(vels)[:, i])
    # axs[3].plot(yaws)
    # #fig.savefig(f'{output_folder}/vels_{control_freq_hz}.png')
    # fig.savefig(f'{output_folder}/vels_{mod}_{simulation_freq_hz}_{control_freq_hz}.png')

    # plt subfigure with 2 rows and 1 column
    fig, axs = plt.subplots(4, 2, figsize=(7.5, 10))
    axs = axs.flatten()

    vels_states = np.array(vels_states)
    yaw_states = np.array(yaw_states)
    control_inputs = np.array(control_inputs)
    axs[0].plot(control_inputs[:, 0], label="vx_true")
    axs[1].plot(control_inputs[:, 1], label="vy_true")
    axs[2].plot(control_inputs[:, 2], label="vz_true")
    axs[3].plot(control_inputs[:, 3], label="omega_z_true")
    axs[0+4].plot(vels_states[:, 0], label="vx_state")
    axs[1+4].plot(vels_states[:, 1], label="vy_state")
    axs[2+4].plot(vels_states[:, 2], label="vz_state")
    axs[3+4].plot(yaw_states, label="yaw_state")

    fig.suptitle(f"Control Powers and States @ {simulation_freq_hz}:{control_freq_hz}Hz")
    axs[3].set_xlabel("frame")

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()
    axs[0+4].legend()
    axs[1+4].legend()
    axs[2+4].legend()
    axs[3+4].legend()
    fig.savefig(f'{output_folder}/vels_{mod}_{simulation_freq_hz}_{control_freq_hz}.png')


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
    parser.add_argument('--csv_path', default=DEFAULT_CSV_PATH, type=str,
                        help='Path to the model parameters file', metavar='')
    parser.add_argument('--convert_video', default=False, type=str2bool, help='Whether to convert the video to mp4',)

    ARGS = parser.parse_args()

    run(**vars(ARGS))
