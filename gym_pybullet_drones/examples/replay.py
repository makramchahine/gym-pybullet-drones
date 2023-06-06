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
from drone_communication.utils.model_utils import load_model_from_weights, generate_hidden_list, get_readable_name, \
    get_params_from_json
from drone_communication.utils.data_utils import image_dir_generator
from drone_communication.keras_models import IMAGE_SHAPE

DEFAULT_DATA_PATH = '/home/makramchahine/repos/drone_communication/data_cli/save-flight-05.09.2023_12.00.29'
DEFAULT_OUT_DIR = '/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/replay_results'
DEFAULT_PARAMS_PATH = '/home/makramchahine/repos/drone_communication/runner_models/val/params.json'
DEFAULT_CHECKPOINT_PATH = '/home/makramchahine/repos/drone_communication/runner_models/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000410_epoch-199_val-loss:0.0009_train-loss:0.0010_mse:0.0010_2023:05:17:12:02:55.hdf5'


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
