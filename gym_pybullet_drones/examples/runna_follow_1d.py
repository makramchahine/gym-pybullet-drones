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
import pandas as pd
import pybullet as p
import matplotlib.pyplot as plt
import cv2
import copy

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
from drone_multimodal.utils.model_utils import load_model_from_weights, generate_hidden_list, get_readable_name, \
    get_params_from_json
from drone_multimodal.keras_models import IMAGE_SHAPE

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
DEFAULT_SAMPLING_FREQ_HQ = 8
DEFAULT_DURATION_SEC = 40
random_deviation = False
DEFAULT_OUTPUT_FOLDER = f'cl_f3_fix_norm_1rnn_8_{DEFAULT_SAMPLING_FREQ_HQ}'
DEFAULT_COLAB = False
normalize_path = None
normalize_path = '/home/makramchahine/repos/drone_multimodal/clean_train_f3_fix_norm_8/mean_std.csv'
DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_f3_fix_norm_1rnn_8/val/params.json'
DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_f3_fix_norm_1rnn_8/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-097_val-loss:0.0565_train-loss:0.0828_mse:0.0828_2023:07:19:18:56:12.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_f3_fix_norm_2rnn_8/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_f3_fix_norm_2rnn_8/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-097_val-loss:0.0514_train-loss:0.0706_mse:0.0706_2023:07:18:14:59:13.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_f3_2rnn_8/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_f3_2rnn_8/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-098_val-loss:0.0001_train-loss:0.0002_mse:0.0002_2023:07:18:10:49:54.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_f3_8/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_f3_8/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-100_val-loss:0.0003_train-loss:0.0006_mse:0.0006_2023:07:18:02:11:50.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_f2_8/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_f2_8/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-099_val-loss:0.0001_train-loss:0.0001_mse:0.0001_2023:07:18:00:59:03.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_f1_fix_8/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_f1_fix_8/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-099_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:07:17:17:59:30.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_f1_8/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_f1_8/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-097_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:07:17:14:28:39.hdf5'

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
    if normalize_path is not None:
        df_norm = pd.read_csv(normalize_path, index_col=0)
        np_mean = df_norm.iloc[0].to_numpy()
        np_std = df_norm.iloc[1].to_numpy()
    print('Loaded Model')

    #### Initialize the simulation #############################
    H = .1
    sim_name = "run-flight-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S.%f")
    sim_dir = os.path.join(output_folder, sim_name)
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir + '/')
    for d in range(num_drones):
        if not os.path.exists(sim_dir + f"/pics{d}"):
            os.makedirs(sim_dir + f"/pics{d}/")


    # axis of 1d motion
    Theta = random.random() * 2 * np.pi
    Theta0 = Theta # drone faces the direction of theta
    Delta_Theta0 = 0 #random.choice([-np.pi * 0.175, np.pi * 0.175])
    INIT_XYZS = np.array([[np.cos(Theta) * -0.5 * d, np.sin(Theta) * -0.5 * d, H] for d in range(num_drones)])
    INIT_RPYS = np.array([[0, 0, Theta0 + Delta_Theta0] for d in range(num_drones)])
    AGGR_PHY_STEPS = int(simulation_freq_hz / control_freq_hz) if aggregate else 1
    NUM_WP = control_freq_hz * DEFAULT_DURATION_SEC

    OBJ_START_DIST = random.uniform(0.5, 2)
    CUSTOM_OBJECT_LOCATION = np.cos(Theta) * OBJ_START_DIST, np.sin(Theta) * OBJ_START_DIST
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
    REC_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ / DEFAULT_SAMPLING_FREQ_HQ))
    action = {str(i): np.array([0, 0, 0, 0]) for i in range(num_drones)}
    START = time.time()
    STEPS = CTRL_EVERY_N_STEPS * NUM_WP

    time_data = []
    x_data = [[] for _ in range(num_drones)]
    y_data = [[] for _ in range(num_drones)]
    vels_states_body = [[] for _ in range(num_drones)]
    yaw_states = [[] for _ in range(num_drones)]
    yaw_rate_states = [[] for _ in range(num_drones)]
    labels = []
    vel_state_world = [[] for _ in range(num_drones)]
    vel_cmds = [[] for _ in range(num_drones)]
    for i in trange(0, int(STEPS), AGGR_PHY_STEPS):

        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)
        states = [obs[str(d)]["state"] for d in range(num_drones)]

        #### Compute control at the desired frequency ##############
        if i % REC_EVERY_N_STEPS == 0:
            for d in range(num_drones):
                rgb, dep, seg = env._getDroneImages(d)
                env._exportImage(img_type=ImageType.RGB,
                                 img_input=rgb,
                                 path=f'{sim_dir}/pics{d}',
                                 frame_num=int(i / CTRL_EVERY_N_STEPS),
                                 )

                rgb = rgb[None,:,:,0:3]

                if d == 0:
                    out = single_step_model.predict([rgb, *hiddens])
                    if normalize_path is not None:
                        out[0][0] = out[0][0] * np_std + np_mean
                    vel_cmd = out[0][0]  # shape: 1 x 8
                    vel_cmds[0] = copy.deepcopy(vel_cmd[:4])
                    vel_cmds[1] = copy.deepcopy(vel_cmd[4:])
                    hiddens = out[1:]  # list num_hidden long, each el is batch x hidden_dim
                    vel_cmd_world = copy.deepcopy(vel_cmds)

                # print(f"start vel_cmd: {vel_cmd_world}")
                yaw = states[d][9]
                yaw_states[d].append(yaw)
                yaw_rate_states[d].append(states[d][15])

                # convert from body_frame to world_frame
                vel_cmd_world[d][0] = vel_cmds[d][0] * np.cos(-yaw) + vel_cmds[d][1] * np.sin(-yaw)
                vel_cmd_world[d][1] = -vel_cmds[d][0] * np.sin(-yaw)+ vel_cmds[d][1] * np.cos(-yaw)
                # vel_cmd[d][2] = 0 # force vertical stability (z direction)

                vel_state_world[d] = copy.deepcopy(states[d][10:13])
                vel_state_body = copy.deepcopy(states[d][10:13])
                # convert from world_frame to body_frame
                vel_state_body[0] = vel_state_world[d][0] * np.cos(states[d][9]) + vel_state_world[d][1] * np.sin(states[d][9])
                vel_state_body[1] = -vel_state_world[d][0] * np.sin(states[d][9]) + vel_state_world[d][1] * np.cos(states[d][9])
                vels_states_body[d].append(vel_state_body)

        if i % CTRL_EVERY_N_STEPS == 0:
            time_data.append(CTRL_EVERY_N_STEPS * env.TIMESTEP * i)
            for d in range(num_drones):
                action[str(d)], _, _ = ctrl[d].computeControl(control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                                                              cur_pos=states[d][0:3],
                                                              cur_quat=states[d][3:7],
                                                              cur_vel=states[d][10:13],
                                                              cur_ang_vel=states[d][13:16],
                                                              target_pos=states[d][0:3],  # same as the current position
                                                              target_rpy=np.array([0, 0, states[d][9]]),  # keep current yaw
                                                              target_vel=vel_cmd_world[d][0:3],
                                                              target_rpy_rates=np.array([0, 0, vel_cmds[d][3]])
                                                              )

                x_data[d].append(states[d][0])
                y_data[d].append(states[d][1])

                with open(sim_dir + f'/state{d}.csv', mode='a') as state_file:
                    state_writer = csv.writer(state_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    state_writer.writerow([i, *states[d]])

                with open(sim_dir + f'/vel_cmd{d}.csv', mode='a') as vel_cmd_file:
                    vel_cmd_writer = csv.writer(vel_cmd_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    vel_cmd_writer.writerow([i, *vel_cmds[d]])

                # with open(sim_dir + '/vel_cmd_world.csv', mode='a') as vel_cmd_file:
                #     vel_cmd_writer = csv.writer(vel_cmd_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                #     vel_cmd_writer.writerow([i, *vel_cmd_world])

                # with open(sim_dir + '/action.csv', mode='a') as action_file:
                #     action_writer = csv.writer(action_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                #     action_writer.writerow([i, *action[str(d)]])

                # logger.log(drone=j,
                #            timestamp=int(i / CTRL_EVERY_N_STEPS),
                #            state=obs[str(j)]["state"],
                #            control=action[str(j)],
                #            )

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save_as_csv(sim_name)  # Optional CSV save

    from matplotlib.collections import LineCollection
    def plot_color_line(fig, ax, x, y, t, color="viridis", alpha=1.0):
        # Create a set of line segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(t.min(), t.max())
        lc = LineCollection(segments, cmap=color, norm=norm, alpha=alpha)
        # Set the values used for colormapping
        lc.set_array(t)
        lc.set_linewidth(2)

        line = ax.add_collection(lc)
        fig.colorbar(line, ax=ax)

    # plot XY data
    time_data, x_data, y_data = np.array(time_data), np.array(x_data), np.array(y_data)
    fig, ax = plt.subplots()
    ax.plot(x_data[0], y_data[0])
    ax.plot(x_data[1], y_data[1])
    ax.plot(CUSTOM_OBJECT_LOCATION[0], CUSTOM_OBJECT_LOCATION[1], 'ro')
    # t = np.linspace(0, 1, len(x_data))  # time variable
    # plot_color_line(fig, ax, x_data[0], y_data[0], t)
    # plot_color_line(fig, ax, x_data[1], y_data[1], t, alpha=0.5, color="plasma")

    ax.legend(["Leader", "Follower"])
    ax.set_title(f"XY Positions @ {DEFAULT_SAMPLING_FREQ_HQ}Hz")
    fig.savefig(sim_dir + "/path.jpg")

    # plot radius over time
    radius_leader = np.sqrt(x_data[0] **2 + y_data[0] **2)
    radius_follower = np.sqrt(x_data[1] **2 + y_data[1] **2)
    fig, ax = plt.subplots()
    ax.plot(time_data, radius_leader)
    ax.plot(time_data, radius_follower)
    ax.set_xlabel("Time (step)")
    ax.set_ylabel("Radius (arb. units)")
    ax.set_title(f"Radius vs time @ {DEFAULT_SAMPLING_FREQ_HQ}Hz")
    fig.savefig(sim_dir + "/radius.jpg")
    # save the radius as a csv
    with open(sim_dir + "/radiusL.csv", 'wb') as out_file:
        np.savetxt(out_file, radius_leader, delimiter=",")
    with open(sim_dir + "/radiusF.csv", 'wb') as out_file:
        np.savetxt(out_file, radius_follower, delimiter=",")

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

    # plot vel, yaw, and yaw_rate states and predictions
    vels_states_body = np.array(vels_states_body)
    yaw_states = np.array(yaw_states)
    yaw_rate_states = np.array(yaw_rate_states)
    for d in range(num_drones):
        fig, axs = plt.subplots(2, 2, figsize=(7.5, 5))
        axs = axs.flatten()

        with open(sim_dir + f'/vel_cmd{d}.csv', mode='r') as vel_cmd_file:
            vel_cmd_reader = csv.reader(vel_cmd_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            vel_cmd = []
            for row in vel_cmd_reader:
                vel_cmd.append(row)
            vel_cmd = np.array(vel_cmd).astype(np.float)
        
            t = np.linspace(0, 1, len(vel_cmd))  # time variable
            for i, (title, ax) in enumerate(zip(["vx_pred", "vy_pred", "vz_pred", "yaw_rate_pred"], axs)):
                ax.plot(t, vel_cmd[:,i+1], label=title)

        t = np.linspace(0, 1, len(vels_states_body[d]))  # time variable
        axs[0].plot(t, vels_states_body[d][:, 0], label="vx_obs_body")
        axs[1].plot(t, vels_states_body[d][:, 1], label="vy_obs_body")
        axs[2].plot(t, vels_states_body[d][:, 2], label="vz_obs")
        axs[3].plot(t, yaw_states[d], label="yaw_obs")
        axs[3].plot(t, yaw_rate_states[d], label="yaw_rate_obs")

        fig.suptitle(f'Velocity/Yaw Pred. and Obs. D{d}')

        for ax in axs:
            ax.legend()
        fig.savefig(f'{sim_dir}/vels{d}.png')

    # # save labels as csv
    # with open(sim_dir + "/labels.csv", 'wb') as out_file:
    #     np.savetxt(out_file, np.array(labels), delimiter=",")

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
