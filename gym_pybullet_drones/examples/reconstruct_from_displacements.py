
import os
import random
from functools import partial
import joblib
from tqdm import tqdm
from datetime import datetime
import numpy as np

from gym_pybullet_drones.examples.simulator_base import BaseSimulator, DEFAULT_NUM_DRONES
from gym_pybullet_drones.examples.simulator_train import TrainSimulator
from culekta_utils import setup_folders

def parse_conditions(sim_dir):
    target_Hs = [0.1 + 0.5]
    Theta = random.random() * 2 * np.pi
    
    with open(os.path.join(sim_dir, 'theta.txt'), 'r') as f:
        Theta_offset = float(f.readline().strip())
    with open(os.path.join(sim_dir, 'start_h.txt'), 'r') as f:
        start_H = float(f.readline().strip())
    with open(os.path.join(sim_dir, 'start_dist.txt'), 'r') as f:
        start_dist = float(f.readline().strip())
        rel_obj = [(start_dist, 0)]

    return start_H, target_Hs, Theta, Theta_offset, rel_obj

def generate_reconstructed_folder(recon_folder):
    start_H, target_Hs, Theta, Theta_offset, rel_obj = parse_conditions(recon_folder)
    Theta = 0

    with open(os.path.join(recon_folder, 'colors.txt'), 'r') as f:
        obj_color = [f.readline().strip()]

    sim = TrainSimulator(obj_color, rel_obj, recon_folder, start_H, target_Hs, Theta, Theta_offset, record_hz=3)
    
    print("Running simulation")
    sim.run_recon()


if __name__ == "__main__":
    reconstruct_folder = "/home/makramchahine/repos/gaussian-splatting/train_blip_10"
    
    subfolders = [f.path for f in os.scandir(reconstruct_folder) if f.is_dir()]
    joblib.Parallel(n_jobs=16)(joblib.delayed(generate_reconstructed_folder)(folder) for folder in tqdm(subfolders))