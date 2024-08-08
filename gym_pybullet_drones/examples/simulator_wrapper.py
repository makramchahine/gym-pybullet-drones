
import os
import random
from functools import partial
import joblib
from tqdm import tqdm
from datetime import datetime
import numpy as np

from gym_pybullet_drones.examples.simulator import BaseSimulator, DEFAULT_NUM_DRONES
from gym_pybullet_drones.examples.simulator_train import TrainSimulator
from culekta_utils import setup_folders

def generate_init_conditions_and_save_to_folder(sim_dir):
    if random.random() < 0.75:
        start_H = 0.1 + random.uniform(0, 1)
        Theta_offset = random.choice([0.175 * np.pi, -0.175 * np.pi])
    else:
        start_H = 0.1 + random.choice([0, 1])
        Theta_offset = random.uniform(0.175 * np.pi, -0.175 * np.pi)
    target_Hs = [0.1 + 0.5]
    Theta = random.random() * 2 * np.pi
    # rel_obj = [(1, 0)]
    rel_obj = [(random.uniform(1, 2), 0)]
    
<<<<<<< Updated upstream

    with open(os.path.join(sim_dir, 'theta.txt'), 'w') as f:
        f.write(str(Theta_offset))
    with open(os.path.join(sim_dir, 'start_h.txt'), 'w') as f:
        f.write(str(start_H))
    with open(os.path.join(sim_dir, 'start_dist.txt'), 'w') as f:
        f.write(str(rel_obj[0][0]))

    return start_H, target_Hs, Theta, Theta_offset, rel_obj

def generate_one_dynamic_training_trajectory(output_folder, obj_color, record_hz):
=======
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

def generate_init_conditions_fly_and_turn_halfwindow(object_color) -> InitConditionsSchema:
    """
    Specific implementation with weighted probabilities

    Task: Single Object -- Approach and Turn

    """
    max_yaw_offset = 0.175 * np.pi / 2
    if random.random() < 0.75:
        start_heights = [0.1 + 0.25 + random.uniform(0, 0.5)]
        theta_offset = random.choice([max_yaw_offset, -max_yaw_offset])
    else:
        start_heights = [0.1 + 0.25 + random.choice([0, 0.5])]
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
    "fly_and_turn_halfwindow": generate_init_conditions_fly_and_turn_halfwindow,
    "2choice": generate_init_conditions_2choice
}

def generate_one_training_trajectory(output_folder, obj_color, record_hz, task_tag: str):
>>>>>>> Stashed changes
    sim_name = "save-flight-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S.%f") # include milliseconds in save name for parallel runs
    sim_dir = os.path.join(output_folder, sim_name)
    setup_folders(sim_dir, DEFAULT_NUM_DRONES)
    
    start_H, target_Hs, Theta, Theta_offset, rel_obj = generate_init_conditions_and_save_to_folder(sim_dir)

    with open(os.path.join(sim_dir, 'colors.txt'), 'w') as f:
        f.write(str("".join(obj_color)))

    sim = TrainSimulator(obj_color, rel_obj, sim_dir, start_H, target_Hs, Theta, Theta_offset, record_hz)
    
    sim.precompute_trajectory()

    print("Running simulation")
    sim.run_simulation_to_completion()

    sim.export_plots()
    sim.logger.save_as_csv(sim_name, sim.custom_timesteps if sim.custom_timesteps else None)  # Optional CSV save


if __name__ == "__main__":
<<<<<<< Updated upstream
    samples = 400
    record_hz = 3 # not the actual hz
    output_folder = f'train_d6_ss2_{samples}_3hzf_bm_px_td_nlsp_gn_nt'
=======
    parser = argparse.ArgumentParser(description='Provide base directory.')
    parser.add_argument('--base_dir', type=str, default="./generated_paths/train_fly_and_turn_halfwindow", help='Base directory for the script')
    parser.add_argument("--samples", type=int, default=600, help="Number of samples")
    parser.add_argument("--record_hz", type=str, default="1-10", help="Recording frequency")
    parser.add_argument("--task_tag", type=str, choices=["2choice", "fly_and_turn", "fly_and_turn_halfwindow"], default="fly_and_turn_halfwindow", help="Task tag")
    args = parser.parse_args()
    
    base_dir = args.base_dir
    samples = args.samples
    record_hz = args.record_hz # ints or "1-10"
    task_tag = args.task_tag
    
>>>>>>> Stashed changes
    OBJECTS = ["R", "B"]
    TOTAL_OBJECTS = OBJECTS
    NUM_INITIALIZATIONS = samples // len(OBJECTS)

    TOTAL_OBJECTS = OBJECTS * NUM_INITIALIZATIONS
    # LOCATIONS_REL = [[(random.uniform(0.5, 1.75), 0)] for _ in range(len(TOTAL_OBJECTS))]
    # LOCATIONS_REL = [[(random.uniform(0.25, 0.25), 0)] for _ in range(len(TOTAL_OBJECTS))]


    total_list = []
    for i, obj in enumerate(zip(TOTAL_OBJECTS)):
        total_list.append(obj)
    assert len(total_list) == NUM_INITIALIZATIONS * (len(OBJECTS)), f"len(total_list): {len(total_list)}"
    random.shuffle(total_list)
    record_hz = "1-10"

    # run_func = partial(run, **vars(ARGS))

    futures = []
    returns = []
    joblib.Parallel(n_jobs=16)(joblib.delayed(generate_one_dynamic_training_trajectory)(output_folder, d, record_hz) for d in tqdm(total_list))