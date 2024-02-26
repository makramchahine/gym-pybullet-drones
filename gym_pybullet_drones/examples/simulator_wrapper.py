
import os
import random
import joblib
from tqdm import tqdm
from datetime import datetime
import numpy as np
import json
import argparse

from schemas import InitConditionsSchema
from gym_pybullet_drones.examples.simulator_base import DEFAULT_NUM_DRONES
from gym_pybullet_drones.examples.simulator_train import TrainSimulator
from culekta_utils import setup_folders


def generate_init_conditions_fly_and_turn(object_color) -> InitConditionsSchema:
    """
    Specific implementation with weighted probabilities

    Task: Single Object -- Approach and Turn

    """
    max_yaw_offset = 0.175 * np.pi
    if random.random() < 0.75:
        start_heights = [0.1 + random.uniform(0, 1)]
        theta_offset = random.choice([max_yaw_offset, -max_yaw_offset])
    else:
        start_heights = [0.1 + random.choice([0, 1])]
        theta_offset = random.uniform(max_yaw_offset, -max_yaw_offset)

    target_heights = [0.1 + 0.5]
    theta_environment = random.random() * 2 * np.pi
    objects_relative = [(random.uniform(1, 2), 0)]
    
    init_conditions_schema = InitConditionsSchema()
    init_conditions = {
        "task_name": "fly_and_turn",
        "start_heights": start_heights,
        "target_heights": target_heights,
        "start_dist": objects_relative[0][0],
        "theta_offset": theta_offset,
        "theta_environment": theta_environment,
        "objects_relative": objects_relative,
        "objects_color": [object_color],
        "objects_relative_target": objects_relative,
        "objects_color_target": [object_color],
    }
    init_conditions = init_conditions_schema.load(init_conditions)

    return init_conditions

def generate_init_conditions_2choice(object_color) -> InitConditionsSchema:
    """
    Specific implementation with weighted probabilities

    Task: Single Object -- Fly to Correct Choice

    """
    max_yaw_offset = 0.1 * np.pi
    if random.random() < 0.75:
        start_heights = [0.1 + random.uniform(0, 1)]
        theta_offset = random.choice([max_yaw_offset, -max_yaw_offset])
    else:
        start_heights = [0.1 + random.choice([0, 1])]
        theta_offset = random.uniform(max_yaw_offset, -max_yaw_offset)

    target_heights = [0.1 + 0.5]
    theta_environment = random.random() * 2 * np.pi
    
    start_dist = random.uniform(1, 2)
    orthogonal_dist = 0.2
    if random.random() < 0.5:
        correct_side = "R"
        objects_relative = [(start_dist, -orthogonal_dist), (start_dist, orthogonal_dist)]
    else:
        correct_side = "L"
        objects_relative = [(start_dist, orthogonal_dist), (start_dist, -orthogonal_dist)]
    
    objects_color = [object_color, "B" if object_color == "R" else "R"]
    objects_color_target = [object_color]
    objects_relative_target = objects_relative[0:1]
    
    init_conditions_schema = InitConditionsSchema()
    init_conditions = {
        "task_name": "2choice",
        "start_heights": start_heights,
        "target_heights": target_heights,
        "start_dist": objects_relative[0][0],
        "theta_offset": theta_offset,
        "theta_environment": theta_environment,
        "objects_relative": objects_relative,
        "objects_color": objects_color,
        "objects_relative_target": objects_relative_target,
        "objects_color_target": objects_color_target,
        "correct_side": correct_side
    }
    init_conditions = init_conditions_schema.load(init_conditions)

    return init_conditions

function_map = {
    "fly_and_turn": generate_init_conditions_fly_and_turn,
    "2choice": generate_init_conditions_2choice
}

def generate_one_training_trajectory(output_folder, obj_color, record_hz, task_tag: str):
    sim_name = "save-flight-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S.%f") # include milliseconds in save name for parallel runs
    sim_dir = os.path.join(output_folder, sim_name)
    setup_folders(sim_dir, DEFAULT_NUM_DRONES)

    generate_init_conditions_func = function_map[task_tag]
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