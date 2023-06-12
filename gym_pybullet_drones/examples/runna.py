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
DEFAULT_CONTROL_FREQ_HZ = 120
DEFAULT_DURATION_SEC = 60
DEFAULT_OUTPUT_FOLDER = 'cl_v5'
DEFAULT_COLAB = False
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-199_val-loss:0.0009_train-loss:0.0010_mse:0.0010_2023:05:17:12:02:55.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v2/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v2/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-100_val-loss:0.0006_train-loss:0.0006_mse:0.0006_2023:06:08:18:05:37.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v3/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v3/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-099_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:09:12:12:39.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v4/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v4/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-099_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:09:18:03:46.hdf5'
DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v5/val/params.json'
DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v5/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-098_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:10:17:21:33.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v6/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v6/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-100_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:11:00:10:06.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v7/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v7/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-100_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:11:09:01:33.hdf5'

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
        params_path = DEFAULT_PARAMS_PATH,
        checkpoint_path = DEFAULT_CHECKPOINT_PATH
):
    #### Initialize the model #############################
    # get model params and load model
    model_params = get_params_from_json(params_path, checkpoint_path)
    model_params.no_norm_layer = False
    model_params.single_step = True
    single_step_model = load_model_from_weights(model_params, checkpoint_path)
    hiddens = generate_hidden_list(model=single_step_model, return_numpy=True)
    print('Loaded Model')

    #### Initialize the simulation #############################
    H = .1
    R = .5
    sim_name = "run-flight-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S.%f")
    sim_dir = os.path.join(output_folder, sim_name)
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir + '/')

    # make initial positions and orientations of drones centered around a circle of radius R and height H around the origin (0,0,0)
    Theta = random.random() * 2 * np.pi
    # Theta = 0.0
    Theta0 = Theta - np.pi
    INIT_XYZS = np.array([[R * np.cos(Theta), R * np.sin(Theta), H]])
    INIT_RPYS = np.array([[0, 0, Theta0]])
    AGGR_PHY_STEPS = int(simulation_freq_hz / control_freq_hz) if aggregate else 1

    #### Initialize a circular trajectory ######################
    PERIOD = random.randint(16, 24)
    NUM_WP = control_freq_hz * DEFAULT_DURATION_SEC

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
    STEPS = CTRL_EVERY_N_STEPS * NUM_WP

    time_data = []
    x_data = []
    y_data = []

    for i in trange(0, int(STEPS), AGGR_PHY_STEPS):

        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)

        #### Compute control at the desired frequency ##############
        if i % CTRL_EVERY_N_STEPS == 0:

            #### Compute control for the current way point #############
            for j in range(num_drones):
                rgb, dep, seg = env._getDroneImages(j)
                env._exportImage(img_type=ImageType.RGB,
                                 img_input=rgb,
                                 path=sim_dir,
                                 frame_num=int(i / CTRL_EVERY_N_STEPS),
                                 )

                rgb = rgb[None,:,:,0:3]

                # if i<STEPS/3:
                #     value = np.array([0, 1])
                # elif i<2*STEPS/3:
                #     value = np.array([1, 0])
                # else:
                #     value = np.array([0, 1])
                #  randomly select [0,1] or [1,0] for value every 1/10 of STEPS
                if i % (STEPS/2) == 0:
                    hazzak = np.random.randint(2, size=1)
                    if hazzak[0] == 1:
                        value = np.array([0, 1])
                    else:
                        value = np.array([1, 0])
                    value = value[None,:]

                # if first image run it through network a 100 times to get a good estimate of the hidden state
                if i == 0:
                    for k in range(1):
                        out = single_step_model.predict([rgb, value,  *hiddens])
                        hiddens = out[1:]
                    vel_cmd = out[0][0]  # shape: 1 x 4
                else:
                    out = single_step_model.predict([rgb, value,  *hiddens])
                    vel_cmd = out[0][0]  # shape: 1 x 4
                    hiddens = out[1:]  # list num_hidden long, each el is batch x hidden_dim

                # write velocity command to a csv
                with open(sim_dir + '/vel_cmd.csv', mode='a') as vel_cmd_file:
                    vel_cmd_writer = csv.writer(vel_cmd_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    vel_cmd_writer.writerow(vel_cmd)

                state = obs[str(j)]["state"]

                yaw  = state[9]

                vel_cmd[0] = vel_cmd[0] * np.cos(-yaw) + vel_cmd[1] * np.sin(-yaw)
                vel_cmd[1] = -vel_cmd[0] * np.sin(-yaw)+ vel_cmd[1] * np.cos(-yaw)
                # vel_cmd[2] = 0 # force vertical stability (z direction)

                action[str(j)], _, _ = ctrl[j].computeControl(control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                                                              cur_pos=state[0:3],
                                                              cur_quat=state[3:7],
                                                              cur_vel=state[10:13],
                                                              cur_ang_vel=state[13:16],
                                                              target_pos=state[0:3],  # same as the current position
                                                              target_rpy=np.array([0, 0, state[9]]),  # keep current yaw
                                                              target_vel=vel_cmd[0:3],
                                                              target_rpy_rates=np.array([0, 0, vel_cmd[3]])
                                                              )

                time_data.append(CTRL_EVERY_N_STEPS * env.TIMESTEP * i)
                x_data.append(state[0])
                y_data.append(state[1])
                # logger.log(drone=j,
                #            timestamp=int(i / CTRL_EVERY_N_STEPS),
                #            state=obs[str(j)]["state"],
                #            control=waypoint
                #            )

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save_as_csv(sim_name)  # Optional CSV save

    # plot radius
    time_data, x_data, y_data = np.array(time_data), np.array(x_data), np.array(y_data)

    plt.plot(x_data, y_data)
    # plot reference circle with radius R
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(R * np.cos(theta), R * np.sin(theta))
    plt.savefig(sim_dir + "/path.jpg")
    plt.close()

    radius = np.sqrt(x_data **2 + y_data **2)
    plt.plot(time_data, radius)
    plt.xlabel("Time (s)")
    plt.ylabel("Radius (arb. units)")
    plt.savefig(sim_dir + "/radius.jpg")
    # save the radius as a csv
    with open(sim_dir + "/radius.csv", 'wb') as out_file:
        np.savetxt(out_file, radius, delimiter=",")

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

    # plot each column of the vel_cmd.csv in a subplot
    with open(sim_dir + '/vel_cmd.csv', mode='r') as vel_cmd_file:
        vel_cmd_reader = csv.reader(vel_cmd_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        vel_cmd = []
        for row in vel_cmd_reader:
            vel_cmd.append(row)
        vel_cmd = np.array(vel_cmd).astype(np.float)
        fig, axs = plt.subplots(4)
        fig.suptitle('Velocity and angular velocity vs index')
        for i in range(4):
            axs[i].plot(vel_cmd[:,i])
        fig.savefig(sim_dir + '/vel_cmd.png')

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
    parser.add_argument('--params_path', default=DEFAULT_PARAMS_PATH, type=str,
                        help='Path to the model parameters file (default: "params.json")', metavar='')

    ARGS = parser.parse_args()

    run(**vars(ARGS))
