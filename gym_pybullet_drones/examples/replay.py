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

from tqdm import trange, tqdm
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
from drone_multimodal.utils.data_utils import image_dir_generator
from drone_multimodal.keras_models import IMAGE_SHAPE

# DEFAULT_DATA_PATH = '/home/makramchahine/repos/drone_multimodal/clean_train_base/save-flight-06.21.2023_07.29.13.667912'
# DEFAULT_DATA_PATH = '/home/makramchahine/repos/drone_multimodal/clean_train_v12/save-flight-06.15.2023_00.32.59.125812'
# DEFAULT_DATA_PATH = '/home/makramchahine/repos/drone_multimodal/clean_replay_debug_60/save-flight-06.27.2023_09.26.12.231414'
DEFAULT_DATA_PATH = '/home/makramchahine/repos/drone_multimodal/clean_train_debug_8_20/save-flight-06.29.2023_09.29.52.256641'
# DEFAULT_DATA_PATH = '/home/makramchahine/repos/drone_multimodal/clean_replay_debug_base/save-flight-06.27.2023_15.07.06.384563'
# DEFAULT_DATA_PATH = '/home/makramchahine/repos/drone_multimodal/clean_train_v11_fast_init_pp/save-flight-06.22.2023_16.37.04.959115'
# DEFAULT_DATA_PATH = '/home/makramchahine/repos/drone_multimodal/clean_train_v11_fast_init/save-flight-06.22.2023_16.05.57.787631'
# DEFAULT_DATA_PATH = '/home/makramchahine/repos/drone_multimodal/clean_train_v11_fast/save-flight-06.22.2023_11.43.22.402338'
# DEFAULT_DATA_PATH = '/home/makramchahine/repos/drone_multimodal/clean_train_v11_early/save-flight-06.19.2023_11.06.38.493638'
# DEFAULT_DATA_PATH = '/home/makramchahine/repos/drone_multimodal/clean_train_v11_fpp_early/save-flight-06.17.2023_09.27.50.397513'
# DEFAULT_DATA_PATH = '/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/sanity_check_black/run_000000'
# DEFAULT_DATA_PATH = '/home/makramchahine/repos/drone_multimodal/clean_train_v11_fp/save-flight-06.16.2023_13.52.38.331658'
# DEFAULT_DATA_PATH = '/home/makramchahine/repos/drone_multimodal/clean_train_v11_p/save-flight-06.16.2023_11.20.13.951611'
# DEFAULT_DATA_PATH = '/home/makramchahine/repos/drone_multimodal/clean_train_v11_p/save-flight-06.16.2023_11.22.44.806711'
# DEFAULT_DATA_PATH = '/home/makramchahine/repos/drone_multimodal/clean_train_RW/save-flight-06.15.2023_16.23.22'
# DEFAULT_DATA_PATH = '/home/makramchahine/repos/drone_multimodal/clean_train_v11/save-flight-06.14.2023_11.16.35.532550'
# DEFAULT_DATA_PATH = '/home/makramchahine/repos/drone_multimodal/clean_train_v10/save-flight-06.12.2023_15.53.14.336564'
# DEFAULT_DATA_PATH = '/home/makramchahine/repos/drone_multimodal/clean_train_v8/save-flight-06.12.2023_10.33.10.122317'
# DEFAULT_DATA_PATH = '/home/makramchahine/repos/drone_multimodal/clean_train_v7/save-flight-06.11.2023_08.38.26.917395'
# DEFAULT_DATA_PATH = '/home/makramchahine/repos/drone_multimodal/clean_train_v6/save-flight-06.10.2023_23.36.44.556644'
# DEFAULT_DATA_PATH = '/home/makramchahine/repos/drone_multimodal/clean_train_v4/save-flight-06.09.2023_17.16.08.900551'
DEFAULT_OUT_DIR = '/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/train_debug_8_20'
# DEFAULT_OUT_DIR = '/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/replay_v11_fast_init_pp_big'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_sanity_check_black_200/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/filtered_sanity_check_black_200/val/model-ctrnn_wiredcfccell_seq-64_lr-0.010000_epoch-200_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:22:04:17:03.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-099_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:14:12:09:26.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fast/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fast/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-100_val-loss:0.0003_train-loss:0.0006_mse:0.0006_2023:06:22:13:45:15.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fast_init/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fast_init/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-095_val-loss:0.0007_train-loss:0.0008_mse:0.0008_2023:06:22:16:33:55.hdf5'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fpp_early_ss/val/model-ctrnn_wiredcfccell_seq-1_lr-0.001000_epoch-097_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:21:12:16:13.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fpp_early/train/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fpp_early/train/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-100_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:19:14:06:27.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fpp_early/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fpp_early/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-100_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:19:14:06:27.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_big/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_big/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-100_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:14:17:41:01.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fpp_early_800/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fpp_early_800/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-399_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:20:09:18:08.hdf5'
DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fast_init_pp_big/val/params.json'
DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v11_fast_init_pp_big/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-098_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:23:10:44:01.hdf5'
# DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v12/val/params.json'
# DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_multimodal/runner_models/init_deviation_v12/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-086_val-loss:0.0000_train-loss:0.0000_mse:0.0000_2023:06:22:10:27:14.hdf5'

def replay(
        data_path = DEFAULT_DATA_PATH,
        out_dir = DEFAULT_OUT_DIR,
        params_path = DEFAULT_PARAMS_PATH,
        checkpoint_path = DEFAULT_CHECKPOINT_PATH
):

    #create out dir if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #### Initialize the model #############################
    # get model params and load model
    model_params = get_params_from_json(params_path, checkpoint_path)
    model_params.no_norm_layer = False
    model_params.single_step = True
    single_step_model = load_model_from_weights(model_params, checkpoint_path)
    hiddens = generate_hidden_list(model=single_step_model, return_numpy=True)
    print('Loaded Model')


    if isinstance(data_path, str):
        data = image_dir_generator(data_path, IMAGE_SHAPE, reverse_channels=False)

    # read data_in csv from data, first row is header
    values = np.genfromtxt(data_path + '/data_in.csv', delimiter=',', skip_header=1)

    #write header to vel_cmd.csv
    with open(out_dir + '/vel_cmd.csv', mode='w') as vel_cmd_file:
        vel_cmd_writer = csv.writer(vel_cmd_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        vel_cmd_writer.writerow(['vx', 'vy', 'vz', 'omega_z'])

    for i, img in tqdm(enumerate(data)):
        img = img[:,:,:,0:3]
        value = np.array([values[i]])

        # if first image run it through network a 100 times to get a good estimate of the hidden state
        if i == 0:
            for k in range(1):
                out = single_step_model.predict([img, value,  *hiddens])
                hiddens = out[1:]
            vel_cmd = out[0][0]  # shape: 1 x 4
        else:
            out = single_step_model.predict([img, value,  *hiddens])
            vel_cmd = out[0][0]  # shape: 1 x 4
            hiddens = out[1:]  # list num_hidden long, each el is batch x hidden_dim
        
        # write velocity command to a csv
        with open(out_dir + '/vel_cmd.csv', mode='a') as vel_cmd_file:
            vel_cmd_writer = csv.writer(vel_cmd_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            vel_cmd_writer.writerow(vel_cmd)




if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    ARGS = parser.parse_args()

    replay(DEFAULT_DATA_PATH, DEFAULT_OUT_DIR, DEFAULT_PARAMS_PATH, DEFAULT_CHECKPOINT_PATH)
