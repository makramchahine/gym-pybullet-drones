
import os
import random
from functools import partial
import joblib
from tqdm import tqdm
from datetime import datetime
import numpy as np

from gym_pybullet_drones.examples.simulator_base import BaseSimulator, DEFAULT_NUM_DRONES
from gym_pybullet_drones.examples.choice_simulator_train import TrainSimulator
from culekta_utils import setup_folders

max_yaw_offset = 0.1 * np.pi

def generate_init_conditions_and_save_to_folder(sim_dir):
    if random.random() < 0.75:
        start_H = 0.1 + random.uniform(0, 1)
        Theta_offset = random.choice([max_yaw_offset, -max_yaw_offset])
    else:
        start_H = 0.1 + random.choice([0, 1])
        Theta_offset = random.uniform(max_yaw_offset, -max_yaw_offset)
    target_Hs = [0.1 + 0.5]
    Theta = random.random() * 2 * np.pi
    dist = random.uniform(1, 2)
    orthogonal_dist = 0.2
    if random.random() < 0.5:
        correct = "R"
        rel_obj = [(dist, -orthogonal_dist), (dist, orthogonal_dist)]
    else:
        correct = "L"
        rel_obj = [(dist, orthogonal_dist), (dist, -orthogonal_dist)]
    
    with open(os.path.join(sim_dir, 'correct.txt'), 'w') as f:
        f.write(str(correct))
    with open(os.path.join(sim_dir, 'theta.txt'), 'w') as f:
        f.write(str(Theta_offset))
    with open(os.path.join(sim_dir, 'start_h.txt'), 'w') as f:
        f.write(str(start_H))
    with open(os.path.join(sim_dir, 'start_dist.txt'), 'w') as f:
        f.write(str(rel_obj[0][0]))
    with open(os.path.join(sim_dir, 'rand_theta.txt'), 'w') as f:
        f.write(str(Theta))

    return start_H, target_Hs, Theta, Theta_offset, rel_obj

def generate_one_dynamic_training_trajectory(output_folder, obj_color, record_hz):
    sim_name = "save-flight-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S.%f") # include milliseconds in save name for parallel runs
    sim_dir = os.path.join(output_folder, sim_name)
    setup_folders(sim_dir, DEFAULT_NUM_DRONES)
    
    start_H, target_Hs, Theta, Theta_offset, rel_obj = generate_init_conditions_and_save_to_folder(sim_dir)

    with open(os.path.join(sim_dir, 'colors.txt'), 'w') as f:
        f.write(str("".join(obj_color)))

    if obj_color[0] == "R":
        obj_color = ["R", "B"]
    elif obj_color[0] == "B":
        obj_color = ["B", "R"]
    else:
        raise ValueError(f"obj_color: {obj_color}")

    sim = TrainSimulator(obj_color, rel_obj, sim_dir, start_H, target_Hs, Theta, Theta_offset, record_hz)
    
    sim.precompute_trajectory()

    print("Running simulation")
    sim.run_simulation_to_completion()

    sim.export_plots()
    sim.logger.save_as_csv(sim_name, sim.custom_timesteps if sim.custom_timesteps else None)  # Optional CSV save


if __name__ == "__main__":
    samples = 6
    record_hz = 3 # ints or "1-10"
    output_folder = f'train_blip_{samples}'
    
    OBJECTS = ["R", "B"]
    TOTAL_OBJECTS = OBJECTS
    NUM_INITIALIZATIONS = samples // len(OBJECTS)
    TOTAL_OBJECTS = OBJECTS * NUM_INITIALIZATIONS


    total_list = []
    for i, obj in enumerate(zip(TOTAL_OBJECTS)):
        total_list.append(obj)
    assert len(total_list) == NUM_INITIALIZATIONS * (len(OBJECTS)), f"len(total_list): {len(total_list)}"
    random.shuffle(total_list)

    futures = []
    returns = []
    joblib.Parallel(n_jobs=16)(joblib.delayed(generate_one_dynamic_training_trajectory)(output_folder, d, record_hz) for d in tqdm(total_list))