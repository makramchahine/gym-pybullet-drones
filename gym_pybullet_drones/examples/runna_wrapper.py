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
import copy
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import itertools
import glob
import re

from tqdm import tqdm
from functools import partial
import joblib

import sys
sys.path.append("/home/makramchahine/repos")
sys.path.append("/home/makramchahine/repos/gym-pybullet-drones")
sys.path.append("/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones")
sys.path.append("/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples")

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync, str2bool

from runna_hike import run
from culekta_utils import PERMUTATIONS_COLORS

import tensorflow as tf

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_VISION = False
DEFAULT_GUI = False
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 240
DEFAULT_RECORD_FREQ_HZ = 8
DEFAULT_DURATION_SEC = 15
DEFAULT_COLAB = False

# normalize_path = '/home/makramchahine/repos/drone_multimodal/clean_train_h0f_hr_300/mean_std.csv'
# base_runner_folder = "/home/makramchahine/repos/drone_multimodal/runner_models/filtered_h0f_hr_300_og_600sf"
# normalize_path = "/home/makramchahine/repos/drone_multimodal/clean_train_d3_300/mean_std.csv"
# base_runner_folder = "/home/makramchahine/repos/drone_multimodal/runner_models/filtered_d3_300_srf_600sf"
normalize_path = None
base_runner_folder = "/home/makramchahine/repos/drone_multimodal/runner_models/filtered_d6_nonorm_ss2_600_1_10hzf_bm_px_td_nlsp_gn_nt_pybullet_srf_300sf_irreg2_64_hyp_cfc"
tag_name = "_".join(base_runner_folder.split('/')[-1].split('_')[1:])
multi = True
vanish = False
if multi:
    tag_name = tag_name + '_multi'
    DEFAULT_DURATION_SEC = DEFAULT_DURATION_SEC * 3
# if vanish:
#     tag_name = tag_name + '_vanish_60deg'
tag_name = tag_name + '_05sf_vanish_60deg'
DEFAULT_OUTPUT_FOLDER = f'cl_{tag_name}'
RUNS_PER_MODEL = 10

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


    concurrent_params_paths = []
    concurrent_checkpoint_paths = []
    output_folder_paths = []

    #! Val evaluation
    val_runner_folder = os.path.join(base_runner_folder, 'val')
    hdf5_files = glob.glob(os.path.join(val_runner_folder, '*.hdf5'))
    json_files = glob.glob(os.path.join(val_runner_folder, '*.json'))
    if hdf5_files:
        hdf5_file_path = hdf5_files[0]  # get the first .hdf5 file
    else:
        print("No .hdf5 files found in the directory.")
    if json_files:
        json_file_path = json_files[0]  # get the first .json file
    else:
        print("No .json files found in the directory.")

    if hdf5_file_path and json_file_path and not os.path.exists(os.path.join(DEFAULT_OUTPUT_FOLDER, 'val')):
        for _ in range(RUNS_PER_MODEL):
            concurrent_checkpoint_paths.append(hdf5_file_path)
            concurrent_params_paths.append(json_file_path)
            output_folder_paths.append(os.path.join(DEFAULT_OUTPUT_FOLDER, 'val'))

    #! Recurrent Checkpoints evaluation
    recurrent_folder = os.path.join(base_runner_folder, 'recurrent')
    hdf5_files = glob.glob(os.path.join(recurrent_folder, '*.hdf5'))

    use_epoch_filter = True
    for hdf5_file_path in hdf5_files:
        # parse "epoch-%d" from hdf5 filename
        epoch_num = int(re.findall(r'epoch-(\d+)', hdf5_file_path)[0])
        print(epoch_num)
        epoch_filter = []
        if multi and use_epoch_filter and epoch_num not in epoch_filter:
            continue

        # print(os.path.join(DEFAULT_OUTPUT_FOLDER, f'recurrent{epoch_num}'))
        # print(os.path.exists(os.path.join(DEFAULT_OUTPUT_FOLDER, f'recurrent{epoch_num}')))
        if os.path.exists(os.path.join(DEFAULT_OUTPUT_FOLDER, f'recurrent{epoch_num}')):
            print(f"skipping epoch {epoch_num}")
            continue
        for _ in range(RUNS_PER_MODEL):
            if os.path.exists(os.path.join(base_runner_folder, 'recurrent', f'params{epoch_num}.json')):
                concurrent_checkpoint_paths.append(hdf5_file_path)
                concurrent_params_paths.append(os.path.join(base_runner_folder, 'recurrent', f'params{epoch_num}.json'))
                output_folder_paths.append(os.path.join(DEFAULT_OUTPUT_FOLDER, f'recurrent{epoch_num}'))

    print(output_folder_paths)

    #! Concurrent Run
    futures = []
    returns = []

    NUM_INITIALIZATIONS = 1
    OBJECTS = ["R", "B"]
    # OBJECTS = ["R", "R", "G", "B", "B"] * NUM_INITIALIZATIONS
    if multi:
        # PERMUTATIONS_COLORS = [list(perm) for perm in itertools.combinations_with_replacement(OBJECTS, 100)]
        # OBJECTS = [random.sample(PERMUTATIONS_COLORS, 1)[0] for _ in range(len(output_folder_paths))]
        # OBJECTS = [random.choices(OBJECTS, k=100) for _ in range(len(output_folder_paths))]
        OBJECTS = [['B', 'B', 'R', 'B', 'R', 'R', 'B', 'R', 'B', 'B', 'B', 'R', 'R', 'B', 'R', 'B', 'R', 'R', 'R', 'R', 'B', 'B', 'R', 'B', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'B', 'B', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'R', 'R', 'B', 'B', 'R', 'R', 'R', 'B', 'R', 'B', 'B', 'R', 'B', 'B', 'B', 'B', 'R', 'R', 'R', 'R', 'B', 'B', 'B', 'B', 'B', 'R', 'B', 'R', 'B', 'B', 'R', 'R', 'B', 'R', 'B', 'R', 'B', 'B', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'B', 'B', 'R', 'B'], ['B', 'R', 'R', 'B', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'B', 'R', 'B', 'B', 'R', 'R', 'R', 'B', 'B', 'R', 'R', 'R', 'R', 'B', 'B', 'B', 'R', 'R', 'B', 'R', 'R', 'B', 'R', 'R', 'B', 'B', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'B', 'R', 'B', 'B', 'B', 'B', 'R', 'B', 'R', 'B', 'B', 'B', 'B', 'B', 'R', 'R', 'R', 'B', 'B', 'R', 'R', 'B', 'B', 'R', 'R', 'R', 'B', 'B', 'B', 'B', 'B', 'B', 'R', 'R', 'R', 'R', 'R', 'B', 'R', 'B', 'R', 'B', 'B', 'B', 'B', 'B', 'B', 'R', 'R', 'B', 'R', 'B', 'B', 'B'], ['R', 'R', 'B', 'B', 'R', 'R', 'B', 'R', 'R', 'B', 'R', 'B', 'B', 'R', 'B', 'B', 'R', 'B', 'B', 'B', 'R', 'B', 'B', 'R', 'B', 'R', 'B', 'B', 'B', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'R', 'B', 'B', 'R', 'B', 'B', 'R', 'R', 'B', 'B', 'B', 'B', 'R', 'B', 'R', 'R', 'R', 'B', 'B', 'R', 'R', 'B', 'R', 'B', 'R', 'B', 'B', 'R', 'R', 'R', 'B', 'B', 'B', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'B', 'R', 'B', 'R', 'B', 'R', 'B', 'R', 'B', 'B', 'R', 'B', 'B', 'R', 'B', 'R', 'B', 'B', 'R', 'B', 'R'], ['R', 'R', 'B', 'B', 'B', 'R', 'R', 'B', 'R', 'B', 'B', 'B', 'R', 'R', 'B', 'B', 'B', 'R', 'B', 'B', 'B', 'R', 'B', 'B', 'B', 'R', 'R', 'B', 'B', 'R', 'B', 'R', 'B', 'R', 'R', 'B', 'B', 'B', 'R', 'R', 'B', 'B', 'B', 'R', 'B', 'B', 'B', 'B', 'R', 'B', 'B', 'B', 'R', 'R', 'B', 'B', 'B', 'B', 'R', 'R', 'R', 'R', 'B', 'B', 'R', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'B', 'R', 'B', 'B', 'R', 'R', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'B', 'B', 'R', 'B', 'B', 'R', 'B', 'B', 'B', 'B', 'B'], ['R', 'R', 'B', 'R', 'B', 'B', 'R', 'R', 'B', 'R', 'R', 'R', 'B', 'B', 'R', 'R', 'B', 'R', 'R', 'B', 'B', 'B', 'B', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'R', 'B', 'B', 'B', 'R', 'B', 'B', 'R', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'B', 'B', 'R', 'R', 'B', 'B', 'R', 'R', 'B', 'R', 'R', 'R', 'B', 'B', 'B', 'R', 'R', 'B', 'B', 'R', 'R', 'B', 'R', 'R', 'B', 'R', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'R', 'R', 'B', 'B', 'R', 'R', 'B', 'B', 'R', 'R', 'B', 'R', 'R', 'B', 'R', 'R', 'R', 'R'], ['R', 'B', 'B', 'B', 'R', 'R', 'B', 'B', 'R', 'R', 'R', 'R', 'B', 'B', 'B', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'B', 'B', 'B', 'R', 'R', 'B', 'B', 'B', 'B', 'R', 'R', 'B', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'B', 'B', 'R', 'B', 'B', 'B', 'B', 'B', 'R', 'B', 'R', 'R', 'B', 'B', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'B', 'B', 'B', 'B', 'R', 'B', 'R', 'B', 'R', 'B', 'R', 'B', 'R', 'R', 'R', 'R', 'B', 'B', 'B', 'R', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'R', 'B', 'R', 'R', 'B', 'B', 'B', 'B', 'R', 'R'], ['B', 'B', 'R', 'B', 'R', 'R', 'R', 'R', 'R', 'B', 'B', 'B', 'B', 'R', 'B', 'R', 'B', 'B', 'B', 'B', 'R', 'B', 'B', 'B', 'R', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'B', 'B', 'R', 'B', 'R', 'R', 'R', 'B', 'B', 'B', 'B', 'B', 'R', 'B', 'R', 'B', 'B', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'B', 'B', 'R', 'B', 'R', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'B', 'B', 'B', 'B', 'R', 'R', 'R', 'R', 'B', 'B', 'B', 'R', 'R', 'B', 'B', 'B', 'R', 'B', 'B', 'R', 'R', 'R', 'B', 'R', 'B', 'B', 'R', 'R', 'R', 'B', 'B', 'R'], ['B', 'B', 'R', 'R', 'B', 'B', 'R', 'R', 'R', 'R', 'B', 'B', 'R', 'B', 'R', 'B', 'R', 'R', 'R', 'B', 'B', 'B', 'R', 'B', 'B', 'R', 'B', 'B', 'B', 'R', 'R', 'B', 'B', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'B', 'R', 'B', 'B', 'R', 'B', 'R', 'R', 'R', 'B', 'B', 'R', 'R', 'B', 'B', 'B', 'R', 'B', 'R', 'R', 'B', 'R', 'R', 'B', 'B', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'B', 'R', 'R', 'B', 'B', 'R', 'B', 'R', 'B', 'B', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'R', 'B'], ['B', 'B', 'B', 'B', 'R', 'B', 'R', 'R', 'R', 'R', 'B', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'B', 'B', 'R', 'R', 'R', 'B', 'B', 'R', 'B', 'R', 'R', 'B', 'B', 'B', 'B', 'R', 'B', 'B', 'R', 'R', 'R', 'R', 'R', 'B', 'R', 'B', 'R', 'B', 'B', 'B', 'R', 'B', 'B', 'B', 'R', 'R', 'R', 'R', 'R', 'R', 'B', 'R', 'B', 'B', 'R', 'B', 'B', 'R', 'B', 'B', 'B', 'R', 'R', 'R', 'B', 'B', 'R', 'R', 'R', 'R', 'B', 'B', 'R', 'B', 'R', 'B', 'R', 'R', 'B', 'R', 'R', 'B', 'R', 'R', 'B', 'B', 'R', 'R', 'B', 'R', 'R', 'B'], ['R', 'B', 'B', 'B', 'R', 'R', 'B', 'B', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'R', 'B', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'B', 'B', 'B', 'B', 'R', 'R', 'R', 'B', 'B', 'B', 'B', 'B', 'R', 'B', 'B', 'R', 'R', 'B', 'B', 'B', 'R', 'B', 'R', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'B', 'B', 'B', 'B', 'B', 'R', 'B', 'B', 'R', 'B', 'B', 'R', 'R', 'B', 'R', 'B', 'B', 'R', 'B', 'R', 'R', 'B', 'B', 'B', 'R', 'B', 'R', 'R', 'R', 'B', 'R', 'B', 'B', 'R', 'R', 'B', 'B', 'R', 'B', 'B', 'B']]
        LOCATIONS_REL = []
        for targets in OBJECTS:
            print(targets)
            locations = []
            cur_point = (0, 0) #random.uniform(0.75, 1.5)
            cur_direction = 0 
            for target in targets:
                cur_dist = random.uniform(1, 1.75) - 0.2
                target_loc = (cur_point[0] + (cur_dist + 0.2) * math.cos(cur_direction), cur_point[1] + (cur_dist + 0.2) * math.sin(cur_direction))
                cur_point = (cur_point[0] + cur_dist * math.cos(cur_direction), cur_point[1] + cur_dist * math.sin(cur_direction))
                locations.append(target_loc)

                if target[0] == 'R':
                    cur_direction += math.pi / 3
                elif target[0] == 'G':
                    cur_direction += 0
                elif target[0] == 'B':
                    cur_direction += -math.pi / 3
            LOCATIONS_REL.append(locations)
    else:
        OBJECTS = OBJECTS * (len(output_folder_paths) // 3)#[[random.choice(OBJECTS)] for _ in range(len(output_folder_paths))]
        OBJECTS = [[x] for x in OBJECTS]
        LOCATIONS_REL = [[(random.uniform(0.75, 1.5), 0)] for _ in range(len(output_folder_paths))]
    

    total_list = []
    for i, (obj, loc) in enumerate(zip(OBJECTS, LOCATIONS_REL)):
        total_list.append((obj, loc))

    # assert len(total_list) == NUM_INITIALIZATIONS * 16 * 5, f"len(total_list): {len(total_list)}"
    joblib.Parallel(n_jobs=10)(joblib.delayed(run)(d, output_folder=output_folder_path, params_path=params_path, checkpoint_path=checkpoint_path, duration_sec=DEFAULT_DURATION_SEC) for d, params_path, checkpoint_path, output_folder_path in tqdm(zip(total_list, concurrent_params_paths, concurrent_checkpoint_paths, output_folder_paths)))
    # for d, params_path, checkpoint_path, output_folder_path in zip(total_list, concurrent_params_paths, concurrent_checkpoint_paths, output_folder_paths):
    #     run(d, normalize_path=normalize_path, output_folder=output_folder_path, params_path=params_path, checkpoint_path=checkpoint_path, duration_sec=DEFAULT_DURATION_SEC)

    import os
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    video_filename = "rand.mp4"
    label_pics = True
    absolute_paths = []
    for eval_dir in os.listdir(DEFAULT_OUTPUT_FOLDER):
        for run in sorted(os.listdir(os.path.join(DEFAULT_OUTPUT_FOLDER, eval_dir))[:]):
            absolute_path = os.path.join(DEFAULT_OUTPUT_FOLDER, eval_dir, run)
            print(absolute_path)
            try:
                if "rand.mp4" not in os.listdir(absolute_path):
                    if label_pics:
                        # labels = np.loadtxt(os.path.join(absolute_path, "labels.csv"), delimiter=",", dtype=str)
                        os.mkdir(os.path.join(absolute_path, "labeled_pics"))
                        for i, img_path in enumerate(sorted(os.listdir(os.path.join(absolute_path, "pics0")))):
                            img = Image.open(os.path.join(absolute_path, "pics0", img_path))
                            width, height = img.size
                            
                            draw = ImageDraw.Draw(img)
                            font = ImageFont.truetype("/usr/share/fonts/truetype/lato/Lato-Medium.ttf", size=20)
                            if i < 8:
                                draw.text((width - 60, 10), "begin", fill="red", font=font)

                            img.save(os.path.join(absolute_path, "labeled_pics", img_path))

                        os.system(f"ffmpeg -framerate 240 -pattern_type glob -i '{absolute_path}/labeled_pics/0*.png' -c:v libx264 -pix_fmt yuv420p {absolute_path}/rand.mp4 > /dev/null 2>&1")
                        os.system(f"rm -rf {absolute_path}/labeled_pics")
                    else:
                        os.system(f"ffmpeg -framerate 240 -pattern_type glob -i '{absolute_path}/pics0/0*.png' -c:v libx264 -pix_fmt yuv420p {absolute_path}/rand.mp4 > /dev/null 2>&1")
            except Exception as e:
                print(e)

        import subprocess
        video_paths = [os.path.join(DEFAULT_OUTPUT_FOLDER, eval_dir, absolute_path, "rand.mp4") for absolute_path in sorted(os.listdir(os.path.join(DEFAULT_OUTPUT_FOLDER, eval_dir))) if os.path.isdir(os.path.join(DEFAULT_OUTPUT_FOLDER, eval_dir, absolute_path)) and  "rand.mp4" in os.listdir(os.path.join(DEFAULT_OUTPUT_FOLDER, eval_dir, absolute_path))]
        combined_video_filename = "combined_video.mp4"
        # concatenate all videos in video_paths
        with open("input.txt", "w") as f:
            for video_path in video_paths:
                f.write(f"file {video_path}\n")

        subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", "input.txt", "-c", "copy", f"{DEFAULT_OUTPUT_FOLDER}/{eval_dir}/{combined_video_filename}"])