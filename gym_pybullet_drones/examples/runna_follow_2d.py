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
DEFAULT_OUTPUT_FOLDER = f'cl_o2_aug4_rand_norm_1rnn_3e4_{DEFAULT_SAMPLING_FREQ_HQ}_300_800'
# DEFAULT_OUTPUT_FOLDER = f'cl_o2_single_aug2_hold_2step_2rnn_{DEFAULT_SAMPLING_FREQ_HQ}_300_100'
DEFAULT_COLAB = False
aligned_follower = True
normalize_path = None
normalize_path = '/home/makramchahine/repos/drone_multimodal/clean_train_o2_aug4_rand_norm_8/mean_std.csv'
# normalize_path = '/home/makramchahine/repos/drone_multimodal/clean_train_o2_single_aug2_hold_2step_norm_8/mean_std.csv'
DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_rand_norm_1rnn_3e4_8_300_800/val/params.json'
DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_rand_norm_1rnn_3e4_8_300_800/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000300_epoch-799_val-loss:0.0038_train-loss:0.0016_mse:0.0016_2023:08:10:15:03:24.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_rand_norm_1rnn_dense_300_800/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_rand_norm_1rnn_dense_300_800/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-498_val-loss:0.0044_train-loss:0.0018_mse:0.0018_2023:08:10:15:14:17.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_rand_norm_1rnn_8_300_800/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_rand_norm_1rnn_8_300_800/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-678_val-loss:0.0033_train-loss:0.0030_mse:0.0030_2023:08:08:17:01:39.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_rand_norm_1rnn_8_300_400/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_rand_norm_1rnn_8_300_400/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-323_val-loss:0.0051_train-loss:0.0035_mse:0.0035_2023:08:10:00:07:39.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_rand_norm_1rnn_6e4_8_300_400/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_rand_norm_1rnn_6e4_8_300_400/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000600_epoch-262_val-loss:0.0054_train-loss:0.0037_mse:0.0037_2023:08:09:10:15:59.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_noback_norm_c1rnn_8_300_800_2/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_noback_norm_c1rnn_8_300_800_2/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-793_val-loss:0.0033_train-loss:0.0028_mse:0.0028_2023:08:08:05:09:53.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_noback_norm_c1rnn_8_300_800/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_noback_norm_c1rnn_8_300_800/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-255_val-loss:0.0029_train-loss:0.0034_mse:0.0034_2023:08:08:05:27:11.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_noback_norm_2rnn_8_300_800/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_noback_norm_2rnn_8_300_800/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-156_val-loss:0.0023_train-loss:0.0018_mse:0.0018_2023:08:06:22:59:51.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_noback_norm_c1rnn_8_300_800/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_noback_norm_c1rnn_8_300_800/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-390_val-loss:0.0043_train-loss:0.0050_mse:0.0050_2023:08:07:14:32:02.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_noback_norm_c1rnn_8_300_400/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_noback_norm_c1rnn_8_300_400/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-381_val-loss:0.0038_train-loss:0.0048_mse:0.0048_2023:08:07:14:44:19.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_noback_norm_1rnn_fix_8_300_800/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_noback_norm_1rnn_fix_8_300_800/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-387_val-loss:0.0051_train-loss:0.0043_mse:0.0043_2023:08:06:22:34:10.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_noback_norm_1rnn_8_300_800/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_noback_norm_1rnn_8_300_800/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-646_val-loss:0.0033_train-loss:0.0027_mse:0.0027_2023:08:05:23:43:23.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_noback_norm_2rnn_8_300_800/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_noback_norm_2rnn_8_300_800/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-798_val-loss:0.0031_train-loss:0.0030_mse:0.0030_2023:08:05:23:43:32.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_noback_norm_2rnn_8_300_500/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_noback_norm_2rnn_8_300_500/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-500_val-loss:0.0046_train-loss:0.0042_mse:0.0042_2023:08:05:01:30:47.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug3_single_noback_norm_2rnn_8_300_500/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug3_single_noback_norm_2rnn_8_300_500/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-473_val-loss:0.0113_train-loss:0.0111_mse:0.0111_2023:08:05:01:24:11.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_single_noback_norm_2rnn_8_300_500/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_single_noback_norm_2rnn_8_300_500/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-500_val-loss:0.0169_train-loss:0.0163_mse:0.0163_2023:08:04:15:11:18.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_single_noback_norm_2rnn_8_300_300/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_single_noback_norm_2rnn_8_300_300/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-167_val-loss:0.0436_train-loss:0.0449_mse:0.0449_2023:08:04:15:10:48.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_noback_norm_2rnn_8_300_100/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_noback_norm_2rnn_8_300_100/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-100_val-loss:0.0175_train-loss:0.0220_mse:0.0220_2023:08:04:13:06:43.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_single_noback_norm_2rnn_8_300_100/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug4_single_noback_norm_2rnn_8_300_100/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-098_val-loss:0.0814_train-loss:0.0883_mse:0.0883_2023:08:04:13:32:59.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug3_noback_norm_2rnn_comm1_8_300_200/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug3_noback_norm_2rnn_comm1_8_300_200/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-199_val-loss:0.0052_train-loss:0.0050_mse:0.0050_2023:08:04:01:02:21.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug3_noback_norm_1rnn_comm_8_300_300/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug3_noback_norm_1rnn_comm_8_300_300/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-297_val-loss:0.0032_train-loss:0.0030_mse:0.0030_2023:08:04:01:00:44.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug3_noback_norm_1rnn_comm_8_300_200/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug3_noback_norm_1rnn_comm_8_300_200/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-199_val-loss:0.0045_train-loss:0.0044_mse:0.0044_2023:08:03:17:41:53.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug3_noback_norm_2rnn_comm_8_300_200/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug3_noback_norm_2rnn_comm_8_300_200/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-193_val-loss:0.0064_train-loss:0.0061_mse:0.0061_2023:08:03:17:30:57.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug3_noback_norm_2rnn_comm_8_300_100/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug3_noback_norm_2rnn_comm_8_300_100/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-099_val-loss:0.0113_train-loss:0.0088_mse:0.0088_2023:08:03:14:11:24.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug3_noback_nc_norm_2rnn_8_300_200/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug3_noback_nc_norm_2rnn_8_300_200/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-197_val-loss:0.0044_train-loss:0.0043_mse:0.0043_2023:08:03:10:52:51.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug3_noback_nc_norm_2rnn_8_300_100/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug3_noback_nc_norm_2rnn_8_300_100/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-098_val-loss:0.0098_train-loss:0.0084_mse:0.0084_2023:08:03:10:51:21.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug3_norm_2rnn_8_300_300/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug3_norm_2rnn_8_300_300/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-299_val-loss:0.0065_train-loss:0.0035_mse:0.0035_2023:08:03:01:56:46.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug3_noback_norm_2rnn_8_300_300/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug3_noback_norm_2rnn_8_300_300/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-290_val-loss:0.0034_train-loss:0.0024_mse:0.0024_2023:08:03:02:28:22.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug3_norm_2rnn_8_300_100/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_aug3_norm_2rnn_8_300_100/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-098_val-loss:0.0139_train-loss:0.0191_mse:0.0191_2023:08:02:22:05:11.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug3_norm_2rnn_8_300_200/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug3_norm_2rnn_8_300_200/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-196_val-loss:0.0447_train-loss:0.0346_mse:0.0346_2023:08:02:21:21:30.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug3_noback_norm_2rnn_8_300_100/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug3_noback_norm_2rnn_8_300_100/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-091_val-loss:0.0176_train-loss:0.0261_mse:0.0261_2023:08:02:17:48:04.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug2_hold_2step_2rnn_norm_8_300_100/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug2_hold_2step_2rnn_norm_8_300_100/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-099_val-loss:0.1232_train-loss:0.2553_mse:0.2553_2023:08:02:14:39:06.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug2_hold_2step_2rnn_8_300_100/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug2_hold_2step_2rnn_8_300_100/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-099_val-loss:0.0001_train-loss:0.0001_mse:0.0001_2023:08:02:14:38:44.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug2_hold_noback_norm_2rnn_cap_multi_8_600_400/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug2_hold_noback_norm_2rnn_cap_multi_8_600_400/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-397_val-loss:0.0112_train-loss:0.0089_mse:0.0089_2023:08:02:01:11:58.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug2_hold_noback_norm_2rnn_cap_8_300_200/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug2_hold_noback_norm_2rnn_cap_8_300_200/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-195_val-loss:0.0193_train-loss:0.0225_mse:0.0225_2023:08:01:08:44:28.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug2_hold_noback_norm_2rnn_8_300_600/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug2_hold_noback_norm_2rnn_8_300_600/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-587_val-loss:0.0180_train-loss:0.0151_mse:0.0151_2023:07:31:23:23:33.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug2_hold_noback_norm_2rnn_8_1200_100/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug2_hold_noback_norm_2rnn_8_1200_100/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-098_val-loss:0.0295_train-loss:0.0271_mse:0.0271_2023:07:31:16:01:42.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug2_hold_noback_norm_2rnn_8_400/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug2_hold_noback_norm_2rnn_8_400/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-395_val-loss:0.0108_train-loss:0.0094_mse:0.0094_2023:07:31:01:03:25.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug2_early_hold_noback_2step_norm_2rnn_8_200/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug2_early_hold_noback_2step_norm_2rnn_8_200/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-200_val-loss:0.1802_train-loss:0.0819_mse:0.0819_2023:07:30:16:57:44.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug2_early_hold_noback_2step_norm_2rnn_8_100/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug2_early_hold_noback_2step_norm_2rnn_8_100/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-099_val-loss:0.1891_train-loss:0.1614_mse:0.1614_2023:07:30:13:50:18.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug_hold_2step_norm_2rnn_8_300/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug_hold_2step_norm_2rnn_8_300/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-283_val-loss:0.1191_train-loss:0.2991_mse:0.2991_2023:07:28:03:41:16.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug2_early_hold_noback_norm_2rnn_8/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug2_early_hold_noback_norm_2rnn_8/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-100_val-loss:0.0295_train-loss:0.0429_mse:0.0429_2023:07:28:14:26:22.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug2_hold_noback_norm_2rnn_8/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug2_hold_noback_norm_2rnn_8/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-094_val-loss:0.0442_train-loss:0.0452_mse:0.0452_2023:07:28:11:43:30.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug_hold_2step_norm_2rnn_8_300/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug_hold_2step_norm_2rnn_8_300/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-283_val-loss:0.1191_train-loss:0.2991_mse:0.2991_2023:07:28:03:41:16.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug_hold_2step_norm_2rnn_relu_8/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug_hold_2step_norm_2rnn_relu_8/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-097_val-loss:0.1197_train-loss:0.3089_mse:0.3089_2023:07:27:22:24:06.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug_hold_2step_norm_2rnn_8/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_aug_hold_2step_norm_2rnn_8/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-100_val-loss:0.1569_train-loss:0.5670_mse:0.5670_2023:07:27:20:53:38.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_hold_2step_norm_2rnn_proc_8/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_hold_2step_norm_2rnn_proc_8/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-047_val-loss:0.1129_train-loss:0.4695_mse:0.4695_2023:07:27:17:56:48.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_hold_2step_norm_2rnn_8/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_single_hold_2step_norm_2rnn_8/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-092_val-loss:0.0940_train-loss:0.4597_mse:0.4597_2023:07:27:16:07:48.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_hold_noback_2step_norm_2rnn_8_comb_nobleed/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_hold_noback_2step_norm_2rnn_8_comb_nobleed/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-098_val-loss:0.1464_train-loss:0.1164_mse:0.1164_2023:07:27:11:07:55.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_hold_noback_2step_norm_2rnn_8_comb/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_hold_noback_2step_norm_2rnn_8_comb/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-058_val-loss:0.1786_train-loss:0.1008_mse:0.1008_2023:07:27:02:22:37.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_hold_noback_2step_norm_2rnn_8_half_300/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o2_hold_noback_2step_norm_2rnn_8_half_300/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-143_val-loss:0.1612_train-loss:0.1027_mse:0.1027_2023:07:26:21:05:43.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o1_label_norm_2rnn_8/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_o1_label_norm_2rnn_8/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-099_val-loss:0.0194_train-loss:0.0218_mse:0.0218_2023:07:24:23:18:36.hdf5'
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
    OBJ_START_DIST = random.uniform(1, 2)

    leader_goes_blue = True if random.random() < 0.5 else False
    left_right = True if random.random() < 0.5 else False
    # write to file left_right value
    with open(os.path.join(sim_dir, 'left_right.txt'), 'w') as f:
        f.write(str(left_right))
    rel_obj_a = (OBJ_START_DIST, 0.5)
    rel_obj_b = (OBJ_START_DIST, -0.5)
    rel_obj_l = rel_obj_a if left_right else rel_obj_b
    rel_obj_f = rel_obj_b if left_right else rel_obj_a
    rel_drone_f = (0, 0)
    rel_drone_l = (-0.5, 0)
    SPAWN_ORDER = [rel_obj_l, rel_obj_f] if leader_goes_blue else [rel_obj_f, rel_obj_l]

    # if i % (STEPS // 2) == 0:
    #     ran_val = np.random.choice([-1, 1])
    value = np.array([0, 1]) if not leader_goes_blue else np.array([1, 0])
    value = value[None,:]
    labels = []

    if num_drones == 1:
        INIT_XYZS = np.array([[*convert_to_global(rel_pos, Theta), H] for rel_pos in [rel_drone_l]])
        INIT_RPYS = np.array([[0, 0, Theta0] for d in range(num_drones)])
    else:
        INIT_XYZS = np.array([[*convert_to_global(rel_pos, Theta), H] for rel_pos in [rel_drone_l, rel_drone_f]])
        INIT_RPYS = np.array([[0, 0, Theta0], [0, 0, Theta]]) if aligned_follower else np.array([[0, 0, Theta0] for d in range(num_drones)])
    CUSTOM_OBJECT_LOCATIONS = [convert_to_global(rel_pos, Theta) for rel_pos in SPAWN_ORDER]
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
                         custom_obj_location=CUSTOM_OBJECT_LOCATIONS
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
            labels.append(1 if leader_goes_blue else -1)

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
    for target in CUSTOM_OBJECT_LOCATIONS:
        ax.plot(target[0], target[1], 'ro')
    # t = np.linspace(0, 1, len(x_data))  # time variable
    # plot_color_line(fig, ax, x_data[0], y_data[0], t)
    # plot_color_line(fig, ax, x_data[1], y_data[1], t, alpha=0.5, color="plasma")

    ax.legend(["Leader", "Follower"])
    ax.set_title(f"XY Positions @ {DEFAULT_SAMPLING_FREQ_HQ}Hz")
    fig.savefig(sim_dir + "/path.jpg")

    # dist = np.sqrt( (x_data[0]-x_data[1]) **2 + (y_data[0]-y_data[1]) **2 ) 
    # fig, ax = plt.subplots()
    # ax.plot(time_data, dist)
    # ax.set_xlabel("Time (step)")
    # ax.set_ylabel("Distance (arb. units)")
    # ax.set_title(f"Distance between Drones @ {DEFAULT_SAMPLING_FREQ_HQ}Hz")
    # fig.savefig(sim_dir + f"/dist.jpg")
    # # save dist as csv
    # with open(sim_dir + f"/dist.csv", 'wb') as out_file:
    #     np.savetxt(out_file, dist, delimiter=",")

    # plot radius over time
    # radius_leader = np.sqrt(x_data[0] **2 + y_data[0] **2)
    # radius_follower = np.sqrt(x_data[1] **2 + y_data[1] **2)
    # fig, ax = plt.subplots()
    # ax.plot(time_data, radius_leader)
    # ax.plot(time_data, radius_follower)
    # ax.set_xlabel("Time (step)")
    # ax.set_ylabel("Radius (arb. units)")
    # ax.set_title(f"Radius vs time @ {DEFAULT_SAMPLING_FREQ_HQ}Hz")
    # fig.savefig(sim_dir + "/radius.jpg")
    # # save the radius as a csv
    # with open(sim_dir + "/radiusL.csv", 'wb') as out_file:
    #     np.savetxt(out_file, radius_leader, delimiter=",")
    # with open(sim_dir + "/radiusF.csv", 'wb') as out_file:
    #     np.savetxt(out_file, radius_follower, delimiter=",")

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
        np.savetxt(out_file, np.array(labels), delimiter=",")

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
