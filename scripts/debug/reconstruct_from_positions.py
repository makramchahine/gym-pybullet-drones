
import os
import joblib
from tqdm import tqdm

from gym_pybullet_drones.examples.simulator_train import TrainSimulator
from path_templates.schemas import parse_init_conditions

def generate_reconstructed_run(recon_folder, task_tag):
    init_conditions = parse_init_conditions(recon_folder)

    sim = TrainSimulator(recon_folder, init_conditions, record_hz=3, task_tag=task_tag)

    sim.run_recon()


if __name__ == "__main__":
    reconstruct_folder = "/home/makramchahine/repos/gaussian-splatting/train_blip_6"
    task_tag = "2choice"
    
    subfolders = [f.path for f in os.scandir(reconstruct_folder) if f.is_dir()]
    joblib.Parallel(n_jobs=16)(joblib.delayed(generate_reconstructed_run)(folder, task_tag) for folder in tqdm(subfolders))