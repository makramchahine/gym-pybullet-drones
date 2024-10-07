import os
import argparse
import math
import random
import numpy as np
import pybullet as p
import glob
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
import joblib
import subprocess
import sys

sys.path.append("/home/makramchahine/repos")
sys.path.append("/home/makramchahine/repos/gym-pybullet-drones")
sys.path.append("/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones")
sys.path.append("/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples")

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync, str2bool

from runna_hike import run_pybullet_only_hike

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

multi = True
vanish = False
RUNS_PER_MODEL = 10

normalize_path = None
base_runner_folders = [
    # "/home/makramchahine/repos/gaussian-splatting/drone_causality/runner_models/filtered_d6_nonorm_ss2_600_1_10hzf_bm_px_td_nlsp_gn_nt_pybullet_srf_300sf_irreg2_64_hyp_cfc_debugmerge_glorot"
   "/home/makramchahine/repos/drone_multimodal/runner_models/filtered_d6_nonorm_ss2_600_1_10hzf_bm_px_td_nlsp_gn_nt_srf_300sf_irreg2_64_hyp_cfc",
#    "/home/makramchahine/repos/drone_multimodal/runner_models/filtered_d6_nonorm_ss2_600_3hzf_bm_px_td_nlsp_gn_nt_srf_300sf_irreg2_64_hyp_cfc",
#    "/home/makramchahine/repos/drone_multimodal/runner_models/filtered_d6_nonorm_ss2_200_9hzf_bm_px_td_nlsp_gn_nt_srf_150sf_irreg2_64_hyp_cfc",
#    "/home/makramchahine/repos/drone_multimodal/runner_models/filtered_d6_nonorm_ss2_600_3hzf_bm_px_td_nlsp_gn_nt_srf_300sf_irreg2_64_hyp_lstm",
#    "/home/makramchahine/repos/drone_multimodal/runner_models/filtered_d6_nonorm_ss2_200_9hzf_bm_px_td_nlsp_gn_nt_srf_150sf_irreg2_64_hyp_lstm",
#    "/home/makramchahine/repos/drone_multimodal/runner_models/filtered_d6_nonorm_ss2_600_1_10hzf_bm_px_td_nlsp_gn_nt_pybullet_srf_300sf_irreg2_64_hyp_cfc",
#    "/home/makramchahine/repos/drone_multimodal/runner_models/filtered_d6_nonorm_ss2_600_3hzf_bm_px_td_nlsp_gn_nt_pybullet_srf_300sf_irreg2_64_hyp_cfc",
#    "/home/makramchahine/repos/drone_multimodal/runner_models/filtered_d6_nonorm_ss2_200_9hzf_bm_px_td_nlsp_gn_nt_pybullet_srf_150sf_irreg2_64_hyp_cfc",
#    "/home/makramchahine/repos/drone_multimodal/runner_models/filtered_d6_nonorm_ss2_600_3hzf_bm_px_td_nlsp_gn_nt_pybullet_srf_300sf_irreg2_64_hyp_lstm",
#    "/home/makramchahine/repos/drone_multimodal/runner_models/filtered_d6_nonorm_ss2_200_9hzf_bm_px_td_nlsp_gn_nt_pybullet_srf_150sf_irreg2_64_hyp_lstm",
]
record_hzs = [
    3,
    # 3,
    # 9,
    # 3,
    # 9
]
variable_timesteps = [
    True,
    # True,
    # True,
    # False,
    # False
]
tag_names = ["_".join(base_runner_folder.split('/')[-1].split('_')[1:]) for base_runner_folder in base_runner_folders]
tag_names = [tag_name + "_multi" if multi else tag_name for tag_name in tag_names]
tag_names = [tag_name + "_vanish_60deg_100_sam_pnorm_at3hz" for tag_name in tag_names] 
default_output_folders = [f'cl_real_{tag_name}' for tag_name in tag_names]

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
    parser.add_argument('--output_folder', default=None, type=str,
                        help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool,
                        help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()


    concurrent_params_paths = []
    concurrent_checkpoint_paths = []
    output_folder_paths = []
    expanded_record_hzs = []
    expanded_variable_timesteps = []

    for base_runner_folder, default_output_folder, record_hz, variable_timestep in zip(base_runner_folders, default_output_folders, record_hzs, variable_timesteps):

        #! Val evaluation
        print(base_runner_folder)
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

        if hdf5_file_path and json_file_path and not os.path.exists(os.path.join(default_output_folder, 'val')):
            for _ in range(RUNS_PER_MODEL):
                concurrent_checkpoint_paths.append(hdf5_file_path)
                concurrent_params_paths.append(json_file_path)
                output_folder_paths.append(os.path.join(default_output_folder, 'val'))
                expanded_record_hzs.append(record_hz)
                expanded_variable_timesteps.append(variable_timestep)

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

            if os.path.exists(os.path.join(default_output_folder, f'recurrent{epoch_num}')):
                print(f"skipping epoch {epoch_num}")
                continue
            for _ in range(RUNS_PER_MODEL):
                if os.path.exists(os.path.join(base_runner_folder, 'recurrent', f'params{epoch_num}.json')):
                    concurrent_checkpoint_paths.append(hdf5_file_path)
                    concurrent_params_paths.append(os.path.join(base_runner_folder, 'recurrent', f'params{epoch_num}.json'))
                    output_folder_paths.append(os.path.join(default_output_folder, f'recurrent{epoch_num}'))
                    expanded_record_hzs.append(record_hz)
                    expanded_variable_timesteps.append(variable_timestep)

    print(output_folder_paths)

    #! Concurrent Run
    futures = []
    returns = []

    NUM_INITIALIZATIONS = 1
    OBJECTS = ["R", "B"]
    if multi:
        # generated with other script
        OBJECTS = [['R', 'R', 'R', 'R', 'R', 'B', 'B', 'B', 'R', 'B', 'R', 'B', 'R', 'R', 'R', 'R', 'R', 'R', 'B', 'B', 'B', 'R', 'R', 'B', 'B', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'B', 'B', 'B', 'B', 'B', 'R', 'B', 'R', 'B', 'R', 'R', 'B', 'B', 'R', 'B', 'B', 'B', 'B', 'R', 'B', 'B', 'R', 'R', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'R', 'B', 'B', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'R', 'B', 'B', 'R', 'R', 'B', 'R', 'R', 'B', 'R', 'R', 'R'], ['R', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'R', 'B', 'R', 'B', 'B', 'B', 'B', 'R', 'B', 'B', 'B', 'B', 'B', 'R', 'B', 'B', 'B', 'R', 'R', 'R', 'R', 'R', 'B', 'B', 'B', 'R', 'R', 'R', 'B', 'B', 'B', 'B', 'B', 'R', 'R', 'B', 'R', 'B', 'B', 'B', 'B', 'B', 'R', 'R', 'B', 'R', 'B', 'R', 'R', 'B', 'R', 'B', 'R', 'R', 'B', 'R', 'B', 'R', 'B', 'B', 'B', 'B', 'B', 'R', 'R', 'R', 'R', 'R', 'B', 'B', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'B', 'B', 'R', 'R', 'B', 'R', 'B', 'R', 'R', 'R', 'B', 'B', 'R', 'R'], ['R', 'R', 'B', 'B', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'B', 'R', 'B', 'R', 'B', 'R', 'R', 'B', 'B', 'R', 'R', 'R', 'R', 'R', 'R', 'B', 'R', 'B', 'B', 'B', 'B', 'B', 'R', 'B', 'B', 'B', 'B', 'R', 'R', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'R', 'B', 'R', 'B', 'B', 'R', 'B', 'B', 'R', 'B', 'B', 'R', 'B', 'B', 'B', 'R', 'B', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'B', 'R', 'R', 'B', 'B', 'R', 'B', 'B', 'R', 'R', 'R', 'R', 'B', 'B', 'R', 'B', 'R', 'B', 'B', 'R', 'B', 'B', 'B', 'B', 'B'], ['R', 'B', 'R', 'R', 'B', 'R', 'R', 'R', 'B', 'R', 'B', 'B', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'B', 'R', 'R', 'B', 'R', 'B', 'B', 'R', 'R', 'B', 'R', 'B', 'R', 'R', 'R', 'R', 'B', 'B', 'B', 'B', 'B', 'B', 'R', 'B', 'R', 'B', 'B', 'R', 'B', 'B', 'B', 'B', 'B', 'B', 'R', 'B', 'B', 'B', 'B', 'B', 'R', 'R', 'B', 'R', 'B', 'R', 'R', 'B', 'R', 'B', 'B', 'R', 'B', 'R', 'R', 'R', 'B', 'R', 'B', 'B', 'B', 'B', 'B', 'R', 'B', 'B', 'R', 'R', 'B', 'B', 'B', 'R', 'R', 'R', 'B', 'R', 'B', 'R', 'R', 'R', 'B'], ['B', 'B', 'R', 'B', 'R', 'R', 'B', 'R', 'B', 'R', 'R', 'R', 'R', 'R', 'B', 'B', 'R', 'B', 'B', 'R', 'B', 'R', 'R', 'R', 'B', 'R', 'B', 'B', 'R', 'R', 'B', 'B', 'B', 'R', 'B', 'B', 'B', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'B', 'R', 'B', 'B', 'R', 'B', 'B', 'B', 'R', 'R', 'R', 'B', 'B', 'B', 'B', 'B', 'R', 'R', 'B', 'B', 'R', 'R', 'R', 'B', 'B', 'B', 'B', 'B', 'B', 'R', 'B', 'R', 'R', 'R', 'B', 'B', 'B', 'R', 'R', 'B', 'R', 'R', 'B', 'R', 'R', 'R', 'B', 'B', 'R', 'R', 'B', 'R', 'B', 'R', 'R', 'R'], ['R', 'B', 'R', 'B', 'B', 'B', 'B', 'R', 'B', 'B', 'R', 'B', 'B', 'B', 'B', 'R', 'R', 'R', 'B', 'B', 'B', 'B', 'B', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'B', 'R', 'B', 'R', 'R', 'R', 'B', 'R', 'B', 'R', 'B', 'R', 'B', 'R', 'R', 'R', 'B', 'R', 'R', 'B', 'B', 'B', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'B', 'R', 'R', 'B', 'B', 'R', 'R', 'R', 'R', 'R', 'B', 'B', 'B', 'R', 'B', 'B', 'B', 'B', 'R', 'B', 'R', 'R', 'B', 'B', 'B', 'R', 'B', 'B', 'B', 'R', 'B', 'B', 'B', 'B', 'R'], ['R', 'B', 'R', 'B', 'B', 'R', 'B', 'B', 'B', 'B', 'B', 'R', 'B', 'B', 'B', 'B', 'B', 'R', 'R', 'B', 'R', 'B', 'B', 'B', 'R', 'R', 'B', 'B', 'B', 'B', 'B', 'R', 'B', 'B', 'B', 'B', 'B', 'B', 'R', 'R', 'R', 'B', 'B', 'R', 'B', 'R', 'R', 'B', 'B', 'R', 'B', 'B', 'B', 'B', 'R', 'B', 'B', 'R', 'B', 'R', 'R', 'R', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'R', 'R', 'R', 'B', 'R', 'B', 'B', 'B', 'R', 'B', 'B', 'B', 'R', 'B', 'R', 'R', 'R', 'B', 'B', 'B', 'B', 'R', 'B', 'B', 'B', 'B', 'R', 'R', 'B', 'B', 'B'], ['R', 'R', 'R', 'B', 'B', 'R', 'R', 'R', 'B', 'B', 'B', 'B', 'B', 'B', 'R', 'B', 'B', 'R', 'R', 'R', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'R', 'B', 'B', 'B', 'B', 'B', 'B', 'R', 'R', 'B', 'B', 'R', 'B', 'R', 'B', 'R', 'R', 'R', 'R', 'R', 'B', 'B', 'B', 'B', 'R', 'B', 'R', 'R', 'B', 'R', 'R', 'R', 'B', 'R', 'R', 'B', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'B', 'R', 'B', 'B', 'B', 'R', 'B', 'B', 'R', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'R', 'B', 'B', 'B', 'B', 'B', 'R', 'B', 'R', 'R', 'B'], ['R', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'B', 'B', 'B', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'B', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'B', 'R', 'B', 'R', 'B', 'R', 'R', 'B', 'R', 'R', 'B', 'R', 'R', 'R', 'B', 'B', 'B', 'B', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'B', 'B', 'R', 'R', 'B', 'B', 'R', 'B', 'B', 'B', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'R', 'B', 'R', 'B', 'B', 'R', 'B', 'B', 'R', 'R', 'B', 'B', 'B', 'R', 'B', 'B', 'R', 'B', 'B', 'B', 'R', 'R', 'R', 'B', 'B', 'R', 'B', 'B'], ['R', 'R', 'R', 'B', 'B', 'R', 'B', 'B', 'R', 'B', 'B', 'R', 'B', 'R', 'B', 'B', 'R', 'B', 'B', 'R', 'B', 'B', 'B', 'R', 'B', 'B', 'R', 'R', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'R', 'R', 'R', 'B', 'B', 'B', 'B', 'R', 'B', 'B', 'R', 'B', 'R', 'B', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'B', 'B', 'R', 'R', 'B', 'R', 'B', 'R', 'B', 'R', 'R', 'B', 'R', 'R', 'R', 'B', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'R']]
        LOCATIONS_REL = []
        for targets in OBJECTS:
            print(targets)
            locations = []
            cur_point = (0, 0)
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
    for default_output_folder in default_output_folders:
        for i, (obj, loc) in enumerate(zip(OBJECTS, LOCATIONS_REL)):
            total_list.append((obj, loc))

    joblib.Parallel(n_jobs=10)(joblib.delayed(run_pybullet_only_hike)(d, output_folder=output_folder_path, params_path=params_path, checkpoint_path=checkpoint_path, duration_sec=DEFAULT_DURATION_SEC, record_hz=record_hz) for d, params_path, checkpoint_path, output_folder_path, record_hz, variable_timestep in tqdm(zip(total_list, concurrent_params_paths, concurrent_checkpoint_paths, output_folder_paths, expanded_record_hzs, expanded_variable_timesteps)))


    video_filename = "rand.mp4"
    label_pics = True

    print(f"default_output_folders: {default_output_folders}")
    
    for default_output_folder in default_output_folders:
        print(f"default_output_folder: {default_output_folder}")
        absolute_paths = []
        try:
            for eval_dir in os.listdir(default_output_folder):
                success_array = []
                for run_pybullet_only_hike in sorted(os.listdir(os.path.join(default_output_folder, eval_dir))[:]):
                    absolute_path = os.path.join(default_output_folder, eval_dir, run_pybullet_only_hike)
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
                    
                    with open(os.path.join(absolute_path, "success.txt"), "r") as f:
                        lines = f.readlines()
                        success_array.append(int(lines[0].strip()))

                # save success_array as csv
                np.savetxt(os.path.join(default_output_folder, eval_dir, "success.csv"), success_array, delimiter=",", fmt="%d")

                video_paths = [os.path.join(default_output_folder, eval_dir, absolute_path, "rand.mp4") for absolute_path in sorted(os.listdir(os.path.join(default_output_folder, eval_dir))) if os.path.isdir(os.path.join(default_output_folder, eval_dir, absolute_path)) and  "rand.mp4" in os.listdir(os.path.join(default_output_folder, eval_dir, absolute_path))]
                combined_video_filename = "combined_video.mp4"
                # concatenate all videos in video_paths
                with open("input.txt", "w") as f:
                    for video_path in video_paths:
                        f.write(f"file {video_path}\n")

                subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", "input.txt", "-c", "copy", f"{default_output_folder}/{eval_dir}/{combined_video_filename}"])
        except Exception as e:
            print(e)
