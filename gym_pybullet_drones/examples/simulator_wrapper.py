
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
    This function generates initial conditions for the "Fly and Turn" task. 
    The task involves a single object that the drone needs to approach and then turn left or right from. 

    The function uses weighted probabilities to determine the starting heights and the theta offset of the drone. 
    - starting heights and theta offset are randomly generated with a 75% chance of being a uniform random value between 0.1 and 1.1 for the height and 
        either the maximum yaw offset or its negative for the theta offset. Otherwise, the starting height is either 0.1 or 1.1, and 
    - the theta offset is a uniform random value between the maximum yaw offset and its negative.

    The target height is set to 0.6, and the theta environment (the angle of the environment in radians) is a random value between 0 and 2π. 
    The relative position of the objects is a random value between 1 and 2 for the x-coordinate and 0 for the y-coordinate.

    The function then creates an instance of the InitConditionsSchema and loads the generated conditions into it. 

    Parameters:
        object_color (str): The color of the object.

    Returns:
        InitConditionsSchema: The loaded initial conditions.
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

def generate_four_init_conditions_turn_only() -> InitConditionsSchema:
    """
    Task: Stay Stationary and Turn to Intended Ball

    Samples the Ball Positions [(x,y), (x,y)]
    Then, mixes [Red, Blue] ball color positions x [Left, Right] trajectory direction

    Returns:
        Array of four `InitConditionsSchema`

    """
    start_dist = 1
    start_heights = [0.1 + 0.5]
    target_heights = [0.1 + 0.5]
    theta_environment = random.random() * 2 * np.pi

    fov_range = [-0.175 * np.pi, 0.175 * np.pi]
    y_in_frame = start_dist * np.tan(random.uniform(fov_range[0], fov_range[1]))
    object_location = (start_dist, y_in_frame)

    # Ensure the second object does not overlap with the first by considering the object radius and a safe margin
    safe_margin = 0.1  # Additional margin for safety
    total_object_width = 2 * 0.1 + safe_margin  # Total width considering both objects' radii and the safe margin
    
    # Sample the y-coordinate for the second object independently of the first object
    y_in_frame_second = start_dist * np.tan(random.uniform(fov_range[0], fov_range[1]))
    # Ensure the second object does not overlap with the first by checking the distance
    while abs(y_in_frame_second - y_in_frame) < total_object_width:
        y_in_frame_second = start_dist * np.tan(random.uniform(fov_range[0], fov_range[1]))
        
    object_location_second = (start_dist, y_in_frame_second)
    
    # Choose a random yaw within a range that ensures both objects are still in frame
    max_yaw_for_frame = min(abs(np.arctan(y_in_frame / start_dist)), abs(np.arctan(y_in_frame_second / start_dist)))
    # theta_offset = random.uniform(-max_yaw_for_frame, max_yaw_for_frame)
    theta_offset=0

    
    objects_relative = [object_location, object_location_second]

    color_order_choices = [["R", "B"], ["B", "R"]]
    target_indexes = range(2)

    init_conditions_list = []
    for color_order in color_order_choices:
        for target_index in target_indexes:
            objects_color = color_order
            objects_color_target = [color_order[target_index]]
            objects_relative_target = [objects_relative[target_index]]
            
            init_conditions_schema = InitConditionsSchema()
            init_conditions = {
                "task_name": "4turn",
                "start_heights": start_heights,
                "target_heights": target_heights,
                "start_dist": start_dist,
                "theta_offset": theta_offset,
                "theta_environment": theta_environment,
                "objects_relative": objects_relative,
                "objects_color": objects_color,
                "objects_relative_target": objects_relative_target,
                "objects_color_target": objects_color_target,
                "correct_color": color_order[target_index]
            }
            init_conditions = init_conditions_schema.load(init_conditions)
            init_conditions_list.append(init_conditions)

    return init_conditions_list

function_map = {
    "fly_and_turn": generate_init_conditions_fly_and_turn,
    "2choice": generate_init_conditions_2choice,
    "4turn": generate_four_init_conditions_turn_only
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

def generate_multiple_training_trajectory(output_folder, obj_color, record_hz, task_tag: str):
    sim_name = "save-flight-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S.%f") # include milliseconds in save name for parallel runs
    sim_dir = os.path.join(output_folder, sim_name)
    for i in range(4):
        setup_folders(f"{sim_dir}_{i}", DEFAULT_NUM_DRONES)


    generate_init_conditions_func = function_map[task_tag]
    init_conditions_list = generate_init_conditions_func()
    for i, init_conditions in enumerate(init_conditions_list):
        with open(os.path.join(f"{sim_dir}_{i}", 'init_conditions.json'), 'w') as f:
            json.dump(init_conditions, f)

        sim = TrainSimulator(f"{sim_dir}_{i}", init_conditions, record_hz, task_tag)
        
        sim.precompute_trajectory(turn_only=True)
        sim.run_simulation_to_completion()

        sim.export_plots()
        sim.logger.save_as_csv(f"{sim_name}_{i}", sim.custom_timesteps if sim.custom_timesteps else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Provide base directory.')
    parser.add_argument('--base_dir', type=str, default="./generated_paths/train_turn_only", help='Base directory for the script')
    parser.add_argument("--samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--record_hz", type=str, default=1, help="Recording frequency")
    parser.add_argument("--task_tag", type=str, choices=["2choice", "fly_and_turn", "4turn"], default="4turn", help="Task tag")
    args = parser.parse_args()
    
    base_dir = args.base_dir
    samples = args.samples
    record_hz = args.record_hz # ints or "1-10"
    task_tag = args.task_tag
    
    OBJECTS = ["R", "B"]
    NUM_INITIALIZATIONS = samples // len(OBJECTS)
    total_list = OBJECTS * NUM_INITIALIZATIONS
    random.shuffle(total_list)

    joblib.Parallel(n_jobs=16)(joblib.delayed(generate_multiple_training_trajectory)(base_dir, d, record_hz, task_tag) for d in tqdm(total_list))