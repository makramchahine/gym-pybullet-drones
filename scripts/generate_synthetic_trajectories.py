
import os
import random
import joblib
from tqdm import tqdm
from datetime import datetime
import numpy as np
import json
import argparse

from simulator.simulator_base import DEFAULT_NUM_DRONES
from simulator.simulator_train import TrainSimulator
from simulator.culekta_utils import setup_folders
from path_templates.trajectory_templates import get_init_conditions_func


def generate_one_training_trajectory(output_folder, obj_color, record_hz, task_tag: str):
    sim_name = "save-flight-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S.%f") # include milliseconds in save name for parallel runs
    sim_dir = os.path.join(output_folder, sim_name)
    setup_folders(sim_dir, DEFAULT_NUM_DRONES)

    generate_init_conditions_func = get_init_conditions_func(task_tag)
    init_conditions = generate_init_conditions_func(obj_color)
    with open(os.path.join(sim_dir, 'init_conditions.json'), 'w') as f:
        json.dump(init_conditions, f)

    sim = TrainSimulator(sim_dir, init_conditions, record_hz, task_tag)
    
    sim.precompute_trajectory()
    sim.run_simulation_to_completion()

    sim.export_plots()
    sim.logger.save_as_csv(sim_name, sim.custom_timesteps if sim.custom_timesteps else None)  # Optional CSV save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Provide base directory.')
    parser.add_argument('--base_dir', type=str, default="./generated_paths/train_fly_and_turn", help='Base directory for the script')
    parser.add_argument("--samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--record_hz", type=str, default=3, help="Recording frequency")
    parser.add_argument("--task_tag", type=str, choices=["2choice", "fly_and_turn"], default="fly_and_turn", help="Task tag")
    args = parser.parse_args()
    
    base_dir = args.base_dir
    samples = args.samples
    record_hz = args.record_hz # ints or "1-10"
    task_tag = args.task_tag
    
    OBJECTS = ["R", "B"]
    NUM_INITIALIZATIONS = samples // len(OBJECTS)
    total_list = OBJECTS * NUM_INITIALIZATIONS
    random.shuffle(total_list)

    joblib.Parallel(n_jobs=16)(joblib.delayed(generate_one_training_trajectory)(base_dir, d, record_hz, task_tag) for d in tqdm(total_list))