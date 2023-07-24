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
DEFAULT_DURATION_SEC = 60
random_deviation = True
DEFAULT_OUTPUT_FOLDER = f'cl2_v13_fast_1r_big_fix_rd_{DEFAULT_SAMPLING_FREQ_HQ}_500'
DEFAULT_COLAB = False
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-199_val-loss:0.0009_train-loss:0.0010_mse:0.0010_2023:05:17:12:02:55.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v2/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v2/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-100_val-loss:0.0006_train-loss:0.0006_mse:0.0006_2023:06:08:18:05:37.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v3/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v3/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-099_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:09:12:12:39.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v4/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v4/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-099_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:09:18:03:46.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v5/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v5/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-098_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:10:17:21:33.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v6/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v6/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-100_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:11:00:10:06.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v7/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v7/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-100_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:11:09:01:33.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v8/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v8/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-098_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:12:11:19:34.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v9/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v9/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-100_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:12:16:19:33.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v9_aug/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v9_aug/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-099_val-loss:0.0004_train-loss:0.0009_mse:0.0009_2023:06:13:17:37:22.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v9_aug_split/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v9_aug_split/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-097_val-loss:0.0002_train-loss:0.0005_mse:0.0005_2023:06:13:23:49:00.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v9_aug_split_v2/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v9_aug_split_v2/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-097_val-loss:0.0001_train-loss:0.0003_mse:0.0003_2023:06:14:08:35:57.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v10/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v10/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-099_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:12:23:45:46.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-099_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:14:12:09:26.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_early/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_early/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-098_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:19:11:41:35.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_p/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_p/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-088_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:16:11:46:04.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fp_early/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fp_early/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-089_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:16:23:41:17.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fpp_early_400/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fpp_early_400/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-400_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:19:23:58:40.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fpp_early_800/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fpp_early_800/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-399_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:20:09:18:08.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_big/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_big/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-100_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:14:17:41:01.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fast_8_big/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fast_8_big/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-097_val-loss:0.0000_train-loss:0.0001_mse:0.0001_2023:07:01:11:26:07.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_64/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_64/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-098_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:15:07:51:58.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/random_walk/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/random_walk/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-097_val-loss:0.0006_train-loss:0.0010_mse:0.0010_2023:06:15:19:15:13.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/base/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/base/val/model-ctrnn_wiredcfccell_seq-64_lr-0.001000_epoch-100_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:21:07:34:44.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/base_ss/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/base_ss/val/model-ctrnn_wiredcfccell_seq-1_lr-0.001000_epoch-013_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:21:10:40:38.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/base_ss_1000/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/base_ss_1000/val/model-ctrnn_wiredcfccell_seq-1_lr-0.001000_epoch-334_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:21:11:09:21.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_big/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_big/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-100_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:14:17:41:01.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_early_ss/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_early_ss/val/model-ctrnn_wiredcfccell_seq-1_lr-0.001000_epoch-017_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:21:11:53:06.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fpp_early_ss/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fpp_early_ss/val/model-ctrnn_wiredcfccell_seq-1_lr-0.001000_epoch-097_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:21:12:16:13.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fast/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fast/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-100_val-loss:0.0003_train-loss:0.0006_mse:0.0006_2023:06:22:13:45:15.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fast_init/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fast_init/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-095_val-loss:0.0007_train-loss:0.0008_mse:0.0008_2023:06:22:16:33:55.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fast_init_pp/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fast_init_pp/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-098_val-loss:0.0004_train-loss:0.0007_mse:0.0007_2023:06:22:18:46:02.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fast_init_pp_big/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fast_init_pp_big/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-098_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:23:10:44:01.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fast_init_pp_1s_big/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fast_init_pp_1s_big/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-095_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:26:02:01:40.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fast_init_pp_big/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fast_init_pp_big/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-098_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:23:10:44:01.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fast_init_pp_60hz/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fast_init_pp_60hz/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-070_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:28:14:34:30.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v12/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v12/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-086_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:22:10:27:14.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v12_fast_init_pp_8_big/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v12_fast_init_pp_8_big/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-098_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:30:07:22:32.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v12_fast_init_pp_8/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v12_fast_init_pp_8/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-100_val-loss:0.0005_train-loss:0.0007_mse:0.0007_2023:06:29:17:24:55.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v13_fast_8_big/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v13_fast_8_big/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-099_val-loss:0.0001_train-loss:0.0001_mse:0.0001_2023:07:03:11:57:44.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v13_fast_init_pp_8_big/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v13_fast_init_pp_8_big/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-099_val-loss:0.0011_train-loss:0.0006_mse:0.0006_2023:06:30:12:00:27.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v13_fast_pp_big_8/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v13_fast_pp_big_8/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-098_val-loss:0.0001_train-loss:0.0001_mse:0.0001_2023:07:05:13:32:24.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v13_fast_1r_big_8/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v13_fast_1r_big_8/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-139_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:07:05:23:43:27.hdf5'
DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v13_fast_1r_big_8_500/val/params.json'
DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v13_fast_1r_big_8_500/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-147_val-loss:0.0000_train-loss:0.0001_mse:0.0001_2023:07:07:01:03:22.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v13_fast_1r_big_2/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v13_fast_1r_big_2/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-150_val-loss:0.0001_train-loss:0.0001_mse:0.0001_2023:07:06:09:44:53.hdf5'

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
    R_base = 1.0
    R = R_base * random.uniform(0.5, 1.5) if random_deviation else R_base
    sim_name = "run-flight-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S.%f")
    sim_dir = os.path.join(output_folder, sim_name)
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir + '/')
    if not os.path.exists(f'{sim_dir}/pics'):
        os.makedirs(f'{sim_dir}/pics' + '/')

    # make initial positions and orientations of drones centered around a circle of radius R and height H around the origin (0,0,0)
    Theta = random.random() * 2 * np.pi
    Theta0 = Theta - np.pi
    Delta_Theta0 = random.uniform(-np.pi * 0.175, np.pi * 0.175) if random_deviation else 0
    INIT_XYZS = np.array([[R * np.cos(Theta), R * np.sin(Theta), H]])
    INIT_RPYS = np.array([[0, 0, Theta0 + Delta_Theta0]])
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
    REC_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ / DEFAULT_SAMPLING_FREQ_HQ))
    action = {str(i): np.array([0, 0, 0, 0]) for i in range(num_drones)}
    START = time.time()
    STEPS = CTRL_EVERY_N_STEPS * NUM_WP

    time_data = []
    x_data = []
    y_data = []
    vels_states_body = []
    yaw_states = []
    yaw_rate_states = []
    radius_data = []
    theta_data = []
    labels = []

    for i in trange(0, int(STEPS), AGGR_PHY_STEPS):

        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)
        state = obs[str(0)]["state"]

        #### Compute control at the desired frequency ##############
        if i % REC_EVERY_N_STEPS == 0:
            for j in range(num_drones):
                rgb, dep, seg = env._getDroneImages(j)
                env._exportImage(img_type=ImageType.RGB,
                                 img_input=rgb,
                                 path=f'{sim_dir}/pics',
                                 frame_num=int(i / CTRL_EVERY_N_STEPS),
                                 )

                rgb = rgb[None,:,:,0:3]

                if i % (STEPS // 2) == 0:
                    ran_val = np.random.choice([-1, 1])
                    value = np.array([0, 1]) if ran_val == -1 else np.array([1, 0])
                    value = value[None,:]
                labels.append(ran_val)

                # if first image run it through network a 100 times to get a good estimate of the hidden state
                if i == 0:
                    for k in range(1):
                        out = single_step_model.predict([rgb, value,  *hiddens])
                        hiddens = out[1:]
                    vel_cmd = out[0][0]  # shape: 1 x 4
                    # vel_cmd = np.array([0, 0, 0, 0])
                else:
                    out = single_step_model.predict([rgb, value,  *hiddens])
                    vel_cmd = out[0][0]  # shape: 1 x 4
                    # vel_cmd = np.array([0, 0, 0, 0])
                    hiddens = out[1:]  # list num_hidden long, each el is batch x hidden_dim

                # # write velocity command to a csv
                # with open(sim_dir + '/vel_cmd.csv', mode='a') as vel_cmd_file:
                #     vel_cmd_writer = csv.writer(vel_cmd_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                #     vel_cmd_writer.writerow(i, vel_cmd)

                yaw = state[9]
                yaw_states.append(yaw)
                yaw_rate_states.append(state[15])

                vel_cmd_world = vel_cmd.copy()
                # convert from body_frame to world_frame
                vel_cmd_world[0] = vel_cmd[0] * np.cos(-yaw) + vel_cmd[1] * np.sin(-yaw)
                vel_cmd_world[1] = -vel_cmd[0] * np.sin(-yaw)+ vel_cmd[1] * np.cos(-yaw)
                # vel_cmd[2] = 0 # force vertical stability (z direction)

                # with open(sim_dir + '/vel_cmd_world.csv', mode='a') as vel_cmd_file:
                #     vel_cmd_writer = csv.writer(vel_cmd_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                #     vel_cmd_writer.writerow(i, vel_cmd_world)

                vel_state_world = state[10:13].copy()
                vel_state_body = state[10:13].copy()
                # convert from world_frame to body_frame
                vel_state_body[0] = vel_state_world[0] * np.cos(state[9]) + vel_state_world[1] * np.sin(state[9])
                vel_state_body[1] = -vel_state_world[0] * np.sin(state[9]) + vel_state_world[1] * np.cos(state[9])
                vels_states_body.append(vel_state_body)

        if i % CTRL_EVERY_N_STEPS == 0:
            for j in range(num_drones):
                action[str(j)], _, _ = ctrl[j].computeControl(control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                                                              cur_pos=state[0:3],
                                                              cur_quat=state[3:7],
                                                              cur_vel=state[10:13],
                                                              cur_ang_vel=state[13:16],
                                                              target_pos=state[0:3],  # same as the current position
                                                              target_rpy=np.array([0, 0, state[9]]),  # keep current yaw
                                                              target_vel=vel_cmd_world[0:3],
                                                              target_rpy_rates=np.array([0, 0, vel_cmd[3]])
                                                              )

                time_data.append(CTRL_EVERY_N_STEPS * env.TIMESTEP * i)
                x_data.append(state[0])
                y_data.append(state[1])

                with open(sim_dir + '/state.csv', mode='a') as state_file:
                    state_writer = csv.writer(state_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    state_writer.writerow([i, *state])

                with open(sim_dir + '/vel_cmd.csv', mode='a') as vel_cmd_file:
                    vel_cmd_writer = csv.writer(vel_cmd_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    vel_cmd_writer.writerow([i, *vel_cmd])

                with open(sim_dir + '/vel_cmd_world.csv', mode='a') as vel_cmd_file:
                    vel_cmd_writer = csv.writer(vel_cmd_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    vel_cmd_writer.writerow([i, *vel_cmd_world])

                with open(sim_dir + '/action.csv', mode='a') as action_file:
                    action_writer = csv.writer(action_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    action_writer.writerow([i, *action[str(j)]])
                # radius_data.append(vel_cmd[4])
                # theta_data.append(vel_cmd[5])

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
    def plot_color_line(fig, ax, x, y, t, color="viridis"):
        # Create a set of line segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(t.min(), t.max())
        lc = LineCollection(segments, cmap=color, norm=norm)
        # Set the values used for colormapping
        lc.set_array(t)
        lc.set_linewidth(2)

        line = ax.add_collection(lc)
        fig.colorbar(line, ax=ax)

    # plot XY data
    time_data, x_data, y_data = np.array(time_data), np.array(x_data), np.array(y_data)
    fig, ax = plt.subplots()
    t = np.linspace(0, 1, len(x_data))  # time variable
    plot_color_line(fig, ax, x_data, y_data, t)

    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(R_base * np.cos(theta), R_base * np.sin(theta), color="red")
    ax.set_aspect('equal', 'box')
    ax.legend(["Actual", "Reference"])
    ax.set_title(f"XY Positions @ {DEFAULT_SAMPLING_FREQ_HQ}Hz")
    fig.savefig(sim_dir + "/path.jpg")

    # plot radius over time
    radius = np.sqrt(x_data **2 + y_data **2)
    fig, ax = plt.subplots()
    ax.plot(time_data, radius)
    ax.set_xlabel("Time (step)")
    ax.set_ylabel("Radius (arb. units)")
    ax.set_title(f"Radius vs time @ {DEFAULT_SAMPLING_FREQ_HQ}Hz")
    fig.savefig(sim_dir + "/radius.jpg")
    # save the radius as a csv
    with open(sim_dir + "/radius.csv", 'wb') as out_file:
        np.savetxt(out_file, radius, delimiter=",")

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

    # plot vel, yaw, and yaw_rate states and predictions
    fig, axs = plt.subplots(2, 2, figsize=(7.5, 5))
    axs = axs.flatten()

    with open(sim_dir + '/vel_cmd.csv', mode='r') as vel_cmd_file:
        vel_cmd_reader = csv.reader(vel_cmd_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        vel_cmd = []
        for row in vel_cmd_reader:
            vel_cmd.append(row)
        vel_cmd = np.array(vel_cmd).astype(np.float)
    
        for i, (title, ax) in enumerate(zip(["vx_pred", "vy_pred", "vz_pred", "yaw_rate_pred"], axs)):
            ax.plot(vel_cmd[:,i+1], label=title)

    vels_states_body = np.array(vels_states_body)
    yaw_states = np.array(yaw_states)
    yaw_rate_states = np.array(yaw_rate_states)

    axs[0].plot(vels_states_body[:, 0], label="vx_obs_body")
    axs[1].plot(vels_states_body[:, 1], label="vy_obs_body")
    axs[2].plot(vels_states_body[:, 2], label="vz_obs")
    axs[3].plot(yaw_states, label="yaw_obs")
    axs[3].plot(yaw_rate_states, label="yaw_rate_obs")

    fig.suptitle(f'Velocity/Yaw Pred. and Obs. @ {DEFAULT_SAMPLING_FREQ_HQ}Hz')

    for ax in axs:
        ax.legend()
    fig.savefig(f'{sim_dir}/vels.png')

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
