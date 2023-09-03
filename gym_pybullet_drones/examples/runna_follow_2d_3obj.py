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
DEFAULT_SAMPLING_FREQ_HQ = 8
DEFAULT_DURATION_SEC = 30
random_deviation = False
DEFAULT_OUTPUT_FOLDER = f'cl_o3_single_switch_early_bal_2rnn_2dim_4e4_d98_3888_1200sf'
# DEFAULT_OUTPUT_FOLDER = f'cl_o3_single_switch_bal_norm_2rnn_2dim_1e4_432_800'
# DEFAULT_OUTPUT_FOLDER = f'cl_o2_single_aug2_hold_2step_2rnn_{DEFAULT_SAMPLING_FREQ_HQ}_300_100'
DEFAULT_COLAB = False
aligned_follower = True
normalize_path = None
normalize_path = '/home/makramchahine/repos/drone_multimodal/clean_train_o3_single_switch_early_bal_3888/mean_std.csv'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_single_switch_early_bal_2rnn_2dim_4e4_3888_800sf/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_single_switch_early_bal_2rnn_2dim_4e4_3888_800sf/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-774_val-loss:0.0043_train-loss:0.0006_mse:0.0006_2023:08:25:09:03:43.hdf5'
DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_single_switch_early_bal_2rnn_2dim_4e4_d98_3888_1200sf/val/params.json'
DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_single_switch_early_bal_2rnn_2dim_4e4_d98_3888_1200sf/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000400_epoch-865_val-loss:0.0081_train-loss:0.0004_mse:0.0004_2023:08:25:18:08:51.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_single_switch_early_bal_2rnn_2dim_1e4_3888_1200sf/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_single_switch_early_bal_2rnn_2dim_1e4_3888_1200sf/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000100_epoch-1090_val-loss:0.0070_train-loss:0.0008_mse:0.0008_2023:08:25:15:54:16.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_single_switch_early_bal_2rnn_2dim_4e4_3888_1600sf/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_single_switch_early_bal_2rnn_2dim_4e4_3888_1600sf/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-1347_val-loss:0.0041_train-loss:0.0002_mse:0.0002_2023:08:25:03:28:14.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_single_switch_early_bal_2rnn_2dim_1e4_3888_800sf/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_single_switch_early_bal_2rnn_2dim_1e4_3888_800sf/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000100_epoch-781_val-loss:0.0111_train-loss:0.0010_mse:0.0010_2023:08:24:22:21:52.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_single_switch_early_bal_2rnn_2dim_1e4_3888_400sf/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_single_switch_early_bal_2rnn_2dim_1e4_3888_400sf/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000100_epoch-394_val-loss:0.0146_train-loss:0.0039_mse:0.0039_2023:08:24:20:08:30.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_single_bal_2rnn_2dim_1e4_1296_400sf/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_single_bal_2rnn_2dim_1e4_1296_400sf/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000100_epoch-388_val-loss:0.0007_train-loss:0.0006_mse:0.0006_2023:08:24:12:00:27.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_switch_comb_norm_2rnn_2dim_1e4_1200_400sf/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_switch_comb_norm_2rnn_2dim_1e4_1200_400sf/model-ctrnn_wiredcfccell_seq-64_lr-0.000100_epoch-257_val-loss:0.0037_train-loss:0.0021_mse:0.0021_2023:08:22:17:07:42.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_comb_norm_2rnn_2dim_1e4_1200_400sf/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_comb_norm_2rnn_2dim_1e4_1200_400sf/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000100_epoch-313_val-loss:0.0019_train-loss:0.0006_mse:0.0006_2023:08:22:03:45:46.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_switch_norm_2rnn_2dim_1e4_600_1200sf/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_switch_norm_2rnn_2dim_1e4_600_1200sf/model-ctrnn_wiredcfccell_seq-64_lr-0.000100_epoch-1099_val-loss:0.0073_train-loss:0.0009_mse:0.0009_2023:08:21:22:24:47.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_2dim_1e4_600_1200sf/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_2dim_1e4_600_1200sf/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000100_epoch-999_val-loss:0.0005_train-loss:0.0004_mse:0.0004_2023:08:21:22:23:46.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_single_switch_bal_norm_2rnn_2dim_1e4_432_800/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_single_switch_bal_norm_2rnn_2dim_1e4_432_800/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000100_epoch-778_val-loss:0.0117_train-loss:0.0061_mse:0.0061_2023:08:23:14:48:34.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_single_bal_norm_2rnn_2dim_1e4_300_800/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_single_bal_norm_2rnn_2dim_1e4_300_800/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000100_epoch-596_val-loss:0.0174_train-loss:0.0116_mse:0.0116_2023:08:23:10:48:28.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_single_bal_norm_2rnn_2dim_300_400/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_single_bal_norm_2rnn_2dim_300_400/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000100_epoch-249_val-loss:0.0341_train-loss:0.0308_mse:0.0308_2023:08:23:04:12:04.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_2dim_3e4_600_400sf/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_2dim_3e4_600_400sf/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000300_epoch-395_val-loss:0.0005_train-loss:0.0007_mse:0.0007_2023:08:20:23:17:48.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_2dim_1e4_600_400sf/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_2dim_1e4_600_400sf/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000100_epoch-400_val-loss:0.0009_train-loss:0.0011_mse:0.0011_2023:08:20:23:25:37.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_switch_norm_2rnn_300_800/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_switch_norm_2rnn_300_800/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-773_val-loss:0.0170_train-loss:0.0021_mse:0.0021_2023:08:15:20:52:47.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_switch_norm_2rnn_2dim_600_400sf/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_switch_norm_2rnn_2dim_600_400sf/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-358_val-loss:0.0053_train-loss:0.0015_mse:0.0015_2023:08:21:01:53:28.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_switch_norm_2rnn_2dim_1e4_600_400sf/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_switch_norm_2rnn_2dim_1e4_600_400sf/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000100_epoch-363_val-loss:0.0105_train-loss:0.0036_mse:0.0036_2023:08:21:02:13:14.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_2dim_cap_600_400/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_2dim_cap_600_400/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-395_val-loss:0.0009_train-loss:0.0012_mse:0.0012_2023:08:20:23:25:09.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_2dim_cap_600_200/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_2dim_cap_600_200/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-160_val-loss:0.0017_train-loss:0.0019_mse:0.0019_2023:08:20:23:11:05.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_1rnn_6dim_instr_600_400/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_1rnn_6dim_instr_600_400/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-263_val-loss:0.0020_train-loss:0.0025_mse:0.0025_2023:08:20:01:41:40.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_6dim_instr_600_400/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_6dim_instr_600_400/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-300_val-loss:0.0025_train-loss:0.0027_mse:0.0027_2023:08:20:02:06:50.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_2dim_3e4_300_100sf/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_2dim_3e4_300_100sf/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000300_epoch-100_val-loss:0.0031_train-loss:0.0040_mse:0.0040_2023:08:20:17:28:59.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_1rnn_6dim_600_400/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_1rnn_6dim_600_400/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-386_val-loss:0.0009_train-loss:0.0011_mse:0.0011_2023:08:18:13:28:51.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_6dim_600_400/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_6dim_600_400/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-399_val-loss:0.0011_train-loss:0.0011_mse:0.0011_2023:08:18:13:17:19.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_600_400s/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_600_400s/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-395_val-loss:0.0005_train-loss:0.0005_mse:0.0005_2023:08:17:19:12:08.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_600_400_3/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_600_400_3/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-317_val-loss:0.0015_train-loss:0.0018_mse:0.0018_2023:08:17:13:08:48.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_300_800_2/val/params2.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_300_800_2/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-500_val-loss:0.0010_train-loss:0.0014_mse:0.0014_2023:08:15:20:41:31.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_300_800_2/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_300_800_2/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-779_val-loss:0.0005_train-loss:0.0007_mse:0.0007_2023:08:15:20:41:31.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_300_800/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_300_800/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-793_val-loss:0.1117_train-loss:0.1643_mse:0.1643_2023:08:14:20:42:33.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_switch_norm_2rnn_300_400/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_switch_norm_2rnn_300_400/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-389_val-loss:0.0193_train-loss:0.0053_mse:0.0053_2023:08:15:10:38:32.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_300_400/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_300_400/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-288_val-loss:0.0017_train-loss:0.0024_mse:0.0024_2023:08:14:20:33:27.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_300_200/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_2rnn_300_200/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-200_val-loss:0.0022_train-loss:0.0033_mse:0.0033_2023:08:14:13:40:43.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_1rnn_300_200/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o3_norm_1rnn_300_200/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-200_val-loss:0.0042_train-loss:0.0065_mse:0.0065_2023:08:14:13:25:23.hdf5'
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
    all_comms = []
    all_comm_trash = []
    # print(f'hiddens: {[h.shape for h in hiddens]}')
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

    def convert_to_global(rel_pos, theta):
        return (np.cos(theta) * rel_pos[0] - np.sin(theta) * rel_pos[1], np.sin(theta) * rel_pos[0] + np.cos(theta) * rel_pos[1])

    # axis of 1d motion
    Theta = random.random() * 2 * np.pi
    Theta0 = Theta # drone faces the direction of theta
    OBJ_START_DIST = random.uniform(1.5, 3)
    num_objects = 3


    # Note: arrays are ordered by [Leader, Follower, Follower, ...]
    
    # Default object locations (3 options) i.e. [left, center, right]
    default_obj_locs_labels = ['L', 'C', 'R']
    default_obj_locs = [(OBJ_START_DIST, lateral_obj_dist) for lateral_obj_dist in [random.uniform(0.3, 0.7), 0, random.uniform(-0.3, -0.7)]]
    # Sample Left/Center/Right locations of objects e.g. [right, center, left]
    sampled_obj_locs_labels = random.sample(default_obj_locs_labels, num_objects)
    sampled_obj_locs = [default_obj_locs[default_obj_locs_labels.index(label)] for label in sampled_obj_locs_labels]
    TARGET_LOCATIONS = [convert_to_global(rel_pos, Theta) for rel_pos in sampled_obj_locs]
    
    # Default color order
    COLORS = ['R', 'G', 'B']
    # Sample colors of objects e.g. [blue, red, green]
    sampled_chosen_colors = random.sample(COLORS, num_objects)
    
    CUSTOM_OBJECT_LOCATION = {
        "colors": sampled_chosen_colors,
        "locations": TARGET_LOCATIONS
    }

    with open(os.path.join(sim_dir, 'locs.txt'), 'w') as f:
        f.write(str("".join(sampled_obj_locs_labels)))
    # with open(os.path.join(sim_dir, 'instr.txt'), 'w') as f:
    #     f.write(str("".join(sampled_chosen_colors)))
    
    # Starting drone locations
    rel_drone_locs = [(0.5 * (i - (num_drones - 1)), 0) for i in range(num_drones)]

    INIT_XYZS = np.array([[*convert_to_global(rel_pos, Theta), H] for rel_pos in rel_drone_locs])
    INIT_RPYS = np.array([[0, 0, Theta0] for d in range(num_drones)])


    # if i % (STEPS // 2) == 0:
    #     ran_val = np.random.choice([-1, 1])
    color_map = {
        'R': [1, 0, 0],
        'G': [0, 1, 0],
        'B': [0, 0, 1],
    }
    if num_drones > 1:
        value = np.array(color_map[sampled_chosen_colors[0]] + color_map[sampled_chosen_colors[1]])
    else:
        value = np.array(color_map[sampled_chosen_colors[0]])
    value = value[None,:]
    print(f"instr: {value}")
    LABELS = []

    AGGR_PHY_STEPS = int(simulation_freq_hz / control_freq_hz) if aggregate else 1
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
    LABELS = []
    vel_state_world = [[] for _ in range(num_drones)]
    vel_cmds = [[] for _ in range(num_drones)]
    for i in trange(0, int(STEPS), AGGR_PHY_STEPS):

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

            inputs = [*imgs, value, *hiddens]
            out = single_step_model.predict(inputs)
            if normalize_path is not None:
                out[0][0] = out[0][0] * np_std + np_mean
            vel_cmd = out[0][0]  # shape: 1 x 8
            vel_cmds[0] = copy.deepcopy(vel_cmd[:4])
            if num_drones > 1:
                vel_cmds[1] = copy.deepcopy(vel_cmd[4:])
                comms = out[1]
                comm_trash = out[2]
                all_comms.append(comms)
                all_comm_trash.append(comm_trash)
                hiddens = out[3:]  # list num_hidden long, each el is batch x hidden_dim
            else:
                hiddens = out[1:]
            # print([hiddens[i].shape for i in range(len(hiddens))])
            vel_cmd_world = copy.deepcopy(vel_cmds)
            LABELS.append("".join(sampled_chosen_colors))

            for d in range(num_drones):
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

    # save all comms as npy
    if num_drones > 1:
        all_comms = np.array(all_comms).squeeze()
        np.save(sim_dir + '/comms.npy', all_comms)
        all_comm_trash = np.array(all_comm_trash).squeeze()
        np.save(sim_dir + '/comms_trash.npy', all_comm_trash)

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
    for target in CUSTOM_OBJECT_LOCATION["locations"]:
        ax.plot(target[0], target[1], 'ro')
    # t = np.linspace(0, 1, len(x_data))  # time variable
    # plot_color_line(fig, ax, x_data[0], y_data[0], t)
    # plot_color_line(fig, ax, x_data[1], y_data[1], t, alpha=0.5, color="plasma")

    ax.legend(["Leader", "Follower"])
    ax.set_title(f"XY Positions @ {DEFAULT_SAMPLING_FREQ_HQ}Hz")
    fig.savefig(sim_dir + "/path.jpg")

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

    # save labels as csv
    with open(sim_dir + "/labels.csv", 'wb') as out_file:
        np.savetxt(out_file, np.array(LABELS), delimiter=",", fmt="%s")

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
