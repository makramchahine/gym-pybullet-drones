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
import glob
import re

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

from culekta_utils import *

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_VISION = False
DEFAULT_GUI = False
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = False
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 240
DEFAULT_SAMPLING_FREQ_HQ = 8
DEFAULT_COLAB = False
DEFAULT_PARAMS_PATH = None
DEFAULT_CHECKPOINT_PATH = None

H = 0.1
vanish_mode = True

def run(
        loc_color_tuple,
        output_folder=None,
        normalize_path=None,
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
        duration_sec=None,
        colab=DEFAULT_COLAB,
        params_path = DEFAULT_PARAMS_PATH,
        checkpoint_path = DEFAULT_CHECKPOINT_PATH
):
    ordered_objs, ordered_locs = loc_color_tuple
    print(f"ordered_objs: {ordered_objs}")
    print(f"ordered_locs: {ordered_locs}")

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

    #! Trajectory-specific parameters
    #* Env Params
    sim_name = "save-flight-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S.%f") # include milliseconds in save name for parallel runs
    sim_dir = os.path.join(output_folder, sim_name)
    setup_folders(sim_dir, num_drones)

    Theta = random.random() * 2 * np.pi
    Theta0 = Theta
    Theta_offset = 0 #random.choice([0.175 * np.pi, -0.175 * np.pi])

    #* Object setup
    obj_loc_global = [convert_to_global(obj_loc_rel, Theta) for obj_loc_rel in ordered_locs]
    TARGET_LOCATIONS = obj_loc_global
    print(f"TARGET_LOCATIONS: {TARGET_LOCATIONS}")

    #* Save starting env params
    with open(os.path.join(sim_dir, 'colors.txt'), 'w') as f:
        f.write(str("".join(ordered_objs)))
    
    # ! Initialize drone locations
    rel_drone_locs = [(0, 0)]

    INIT_XYZS = np.array([[*convert_to_global(rel_pos, Theta), H] for rel_pos in rel_drone_locs])
    INIT_RPYS = np.array([[0, 0, Theta0 + Theta_offset] for d in range(num_drones)])
    AGGR_PHY_STEPS = int(simulation_freq_hz / control_freq_hz) if aggregate else 1

    NUM_WP = control_freq_hz * duration_sec

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
                         custom_obj_location=None if vanish_mode else
                            {
                                "colors": ordered_objs,
                                "locations": obj_loc_global
                            }
                         )
    env.IMG_RES = np.array([256, 144])
    target_index = 0
    alive_obj_id = env.addObject(ordered_objs[target_index], obj_loc_global[target_index])
    previous_G = None

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
    LABELS = []
    vel_state_world = [[] for _ in range(num_drones)]
    vel_cmds = [[] for _ in range(num_drones)]
    value = np.array([0, 0, 0, 0, 0, 0])
    value = value[None,:]
    alive_obj_previously_in_view = False
    finished_within_time_flag = False
    window_outcomes = []
    range_outcomes = []
    for i in trange(0, int(STEPS), AGGR_PHY_STEPS):
        if target_index > len(ordered_objs) - 1:
            finished_within_time_flag = True
            break

        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)
        states = [obs[str(d)]["state"] for d in range(num_drones)]

        #### Compute control at the desired frequency ##############
        if i % REC_EVERY_N_STEPS == 0:
            imgs = [[], []]
            for d in range(num_drones):
                rgb, dep, seg = env._getDroneImages(d)
                env._exportImage(img_type=ImageType.RGB,
                                 img_input=rgb,
                                 path=f'{sim_dir}/pics{d}',
                                 frame_num=int(i / CTRL_EVERY_N_STEPS),
                                 )

                rgb = rgb[None,:,:,0:3]
                imgs[d] = rgb

            inputs = [*imgs, *hiddens]
            out = single_step_model.predict(inputs)
            if normalize_path is not None:
                out[0][0] = out[0][0] * np_std + np_mean
            vel_cmd = out[0][0]  # shape: 1 x 8
            vel_cmds[0] = copy.deepcopy(vel_cmd[:4])
            if num_drones > 1:
                vel_cmds[1] = copy.deepcopy(vel_cmd[4:])
                hiddens = out[1:]  # list num_hidden long, each el is batch x hidden_dim
            else:
                hiddens = out[1:]
            # print([hiddens[i].shape for i in range(len(hiddens))])
            vel_cmd_world = copy.deepcopy(vel_cmds)

            for d in range(num_drones):
                # print(f"start vel_cmd: {vel_cmd_world}")
                x, y, z = states[d][0], states[d][1], states[d][2]
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
                
                if object_in_view(x, y, yaw, obj_loc_global[target_index]):
                    alive_obj_previously_in_view = True

                if ordered_objs[target_index] == 'G' and z > 0.2:
                    previous_G = {"id": alive_obj_id, "loc": obj_loc_global[target_index]}
                    window_outcomes.append("G")

                    alive_obj_previously_in_view = False
                    target_index += 1
                    if not (target_index > len(ordered_objs) - 1):
                        env.addObject(ordered_objs[target_index], obj_loc_global[target_index])
                elif previous_G is not None and not object_in_view(x, y, yaw, previous_G["loc"]):
                    if object_in_range(x, y, previous_G["loc"]):
                        range_outcomes.append("G")
                    env.removeObject(previous_G["id"])
                    previous_G = None
                elif not object_in_view(x, y, yaw, obj_loc_global[target_index]) and alive_obj_previously_in_view:
                    if (ordered_objs[target_index] == 'R' and drone_turned_left(x, y, yaw, obj_loc_global[target_index]) or (ordered_objs[target_index] == 'B' and drone_turned_right(x, y, yaw, obj_loc_global[target_index]))):
                        window_outcomes.append(ordered_objs[target_index])
                    elif ordered_objs[target_index] == 'R' or ordered_objs[target_index] == 'B':
                        window_outcomes.append("N")
                    if object_in_range(x, y, obj_loc_global[target_index]):
                        range_outcomes.append(ordered_objs[target_index])
                    
                    alive_obj_previously_in_view = False
                    target_index += 1
                    env.removeObject(alive_obj_id)
                    if not (target_index > len(ordered_objs) - 1):
                        env.addObject(ordered_objs[target_index], obj_loc_global[target_index])



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

                # logger.log(drone=j,
                #            timestamp=int(i / CTRL_EVERY_N_STEPS),
                #            state=obs[str(j)]["state"],
                #            control=action[str(j)],
                #            )

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.TIMESTEP)

    env.close()
    logger.save_as_csv(sim_name)  # Optional CSV save

    if window_outcomes == []:
        window_outcomes.append("X")
    if range_outcomes == []:
        range_outcomes.append("X")

    try:
        with open(os.path.join(sim_dir, 'finish.txt'), 'w') as f:
            max_len = max(len(window_outcomes), len(range_outcomes), len(ordered_objs))
            # pad window_outcomes, range_outcomes, ordered_objs with X's if they are too short
            window_outcomes = window_outcomes + ['X'] * (max_len - len(window_outcomes))
            range_outcomes = range_outcomes + ['X'] * (max_len - len(range_outcomes))
            ordered_objs = ordered_objs + ['X'] * (max_len - len(ordered_objs))

            for window_outcome, range_outcome, color in zip(window_outcomes, range_outcomes, ordered_objs):
                f.write(f"{window_outcome},{range_outcome},{color}\n")
    except Exception as e:
        print(e)

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
    if num_drones > 1:
        ax.plot(x_data[1], y_data[1])
    for target in TARGET_LOCATIONS:
        ax.plot(target[0], target[1], 'ro')

    ax.legend(["Leader", "Follower"])
    ax.set_title(f"XY Positions @ {DEFAULT_SAMPLING_FREQ_HQ}Hz")
    fig.savefig(sim_dir + "/path.jpg")


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

    # save labels as csv
    with open(sim_dir + "/labels.csv", 'wb') as out_file:
        np.savetxt(out_file, np.array(LABELS), delimiter=",", fmt="%s")


