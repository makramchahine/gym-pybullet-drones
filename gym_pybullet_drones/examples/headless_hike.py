import os
import time
from datetime import datetime
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from tqdm import trange

from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

from culekta_utils import *

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
DEFAULT_DURATION_SEC = 8
DEFAULT_COLAB = False
aligned_follower = True


#* Env Params
NUM_OBJECTS = 3
HS = [0.1]

#* Augmentation Params
EARLY_STOP = False
EARLY_STOP_FRAME = random.randint(73, 138)
SWITCH = False
NUM_SWITCHES = 30

CRITICAL_DIST = 0.5
CRITICAL_DIST_BUFFER = 0.1
FINISH_COUNTER_THRESHOLD = 32
# TURN_ANGLE = np.pi / 2


class Simulator():
    def __init__(self, ordered_objs, ordered_locs, sim_dir, theta_offset=None):
        self.num_drones = DEFAULT_NUM_DRONES
        self.sim_dir = sim_dir
        self.simulation_freq_hz = DEFAULT_SIMULATION_FREQ_HZ
        self.control_freq_hz = DEFAULT_CONTROL_FREQ_HZ
        self.record_freq_hz = DEFAULT_RECORD_FREQ_HZ
        self.duration_sec = DEFAULT_DURATION_SEC
        self.Theta_offset = theta_offset if theta_offset is not None else random.choice([0.175 * np.pi, -0.175 * np.pi])
        self.Theta = random.random() * 2 * np.pi
        self.Theta0 = self.Theta
        self.start_H = random.uniform(0.05, 0.25)
        self.TARGET_HS = [0.1]
        self.aggregate = DEFAULT_AGGREGATE
        self.AGGR_PHY_STEPS = int(self.simulation_freq_hz / self.control_freq_hz) if self.aggregate else 1
        self.ordered_objs = ordered_objs
        self.target_index = 0

        self.frame_counter = 0
        self.finish_counter = 0
        self.previously_reached_critical = False
        self.has_precomputed_trajectory = False

        rel_drone_locs = [(0, 0)]
        self._setup_position_info(rel_drone_locs, ordered_locs)

    def _setup_position_info(self, rel_drone_locs, ordered_locs):
        self.FINAL_THETA = [angle_between_two_points(rel_drone, rel_obj) for rel_drone, rel_obj in zip(rel_drone_locs, ordered_locs)]
        self.INIT_XYZS = np.array([[*convert_to_global(rel_pos, self.Theta), self.start_H] for rel_pos in rel_drone_locs])
        self.INIT_RPYS = np.array([[0, 0, self.Theta0 + self.Theta_offset] for d in range(self.num_drones)])

        self.TARGET_POS = [[arry] for arry in self.INIT_XYZS]
        self.TARGET_ATT = [[arry] for arry in self.INIT_RPYS]
        self.INIT_THETA = [init_rpys[2] for init_rpys in self.INIT_RPYS]
        # angular distance between init and final theta
        self.DELTA_THETA = [signed_angular_distance(init_theta, final_theta + self.Theta) for final_theta, init_theta in zip(self.FINAL_THETA, self.INIT_RPYS[:, 2])]

        self.obj_loc_global = [convert_to_global(obj_loc_rel, self.Theta) for obj_loc_rel in ordered_locs]
        self.TARGET_LOCATIONS = self.obj_loc_global
        self.critical_action = self.ordered_objs[self.target_index]
        self.critical_dist = CRITICAL_DIST
        self.critical_dist_buffer = CRITICAL_DIST_BUFFER

    def init_stable_trajectory(self):
        for i in range(240):
            self._step_trajectory(hold=True)

    def _step_trajectory(self, hold=False):
        """
        Modifies:
            self.TARGET_POS, self.TARGET_ATT, self.FINAL_THETA, self.reached_critical
        """
        speeds = []
        for i, (target_pos, target_att, init_theta, final_theta, delta_theta, final_target, height) in enumerate(zip(self.TARGET_POS, self.TARGET_ATT, self.INIT_THETA, self.FINAL_THETA, self.DELTA_THETA, self.TARGET_LOCATIONS, self.TARGET_HS)):
            last_pos = target_pos[-1]    # X, Y, Z
            last_yaw = target_att[-1][2] # R, P, Y
            last_height = target_pos[-1][2]
            dist = distance_to_target(last_pos, final_target)
            yaw_dist = signed_angular_distance(last_yaw, final_theta + self.Theta)
            height_dist = height - last_height

            lift_speed = 0
            if hold:
                speed = 0
                yaw_speed = 0
                new_theta = init_theta
                lift_speed = 0
            elif dist > self.critical_dist + self.critical_dist_buffer and not self.previously_reached_critical:
                speed = DEFAULT_SPEED / self.control_freq_hz
                yaw_speed = DEFAULT_SEARCHING_YAW * np.sign(yaw_dist) / self.control_freq_hz
                lift_speed = 0 if (abs(height_dist) < APPROX_CORRECT_HEIGHT) else STABILIZE_LIFT_SPEED * (height_dist / height) / self.control_freq_hz

                if abs(yaw_dist) < APPROX_CORRECT_YAW:
                    new_theta = final_theta + self.Theta
                else:
                    new_theta = last_yaw + yaw_speed
                self.FINAL_THETA[0] = angle_between_two_points(last_pos[:2], final_target[:2]) - self.Theta
            elif dist > self.critical_dist and not self.previously_reached_critical:
                speed = interpolate_speeds(dist, self.critical_dist, self.critical_dist_buffer, DEFAULT_CRITICAL_SPEED, DEFAULT_SPEED) / self.control_freq_hz
                yaw_speed = interpolate_speeds(dist, self.critical_dist, self.critical_dist_buffer, DEFAULT_CRITICAL_YAW_SPEED, DEFAULT_SEARCHING_YAW) * np.sign(yaw_dist) / self.control_freq_hz
                lift_speed = 0 if (abs(height_dist) < APPROX_CORRECT_HEIGHT) else STABILIZE_LIFT_SPEED * (height_dist / height) / self.control_freq_hz

                if abs(yaw_dist) < APPROX_CORRECT_YAW:
                    new_theta = final_theta + self.Theta
                else:
                    new_theta = last_yaw + yaw_speed
            else:
                speed = DEFAULT_CRITICAL_SPEED / self.control_freq_hz
                yaw_speed = DEFAULT_SEARCHING_YAW * np.sign(yaw_dist) / self.control_freq_hz
                if self.critical_action == 'R':
                    yaw_speed += DEFAULT_CRITICAL_YAW_SPEED / self.control_freq_hz
                elif self.critical_action == 'B':
                    yaw_speed += -DEFAULT_CRITICAL_YAW_SPEED / self.control_freq_hz
                elif self.critical_action == 'G':
                    lift_speed = DEFAULT_LIFT_SPEED / self.control_freq_hz
                    if not self.previously_reached_critical:
                        self.FINAL_THETA[0] = angle_between_two_points(last_pos[:2], final_target[:2]) - self.Theta # continue to face target
                else:
                    assert False, f"critical_action: {self.critical_action}"
                new_theta = last_yaw + yaw_speed


            if self.critical_action == 'G':
                delta_z = lift_speed if not self.previously_reached_critical else -DROP_SPEED / self.control_freq_hz * (last_height / DROP_MAX_HEIGHT)
                self.reached_critical = dist < DEFAULT_DROP_POINT_DIST
                new_height = max(last_height + delta_z, height)
            else:
                delta_z = lift_speed
                self.reached_critical = dist < self.critical_dist
                new_height = (last_height + delta_z) if (lift_speed != 0 or hold) else height
        
            delta_pos = convert_to_global([speed, 0], new_theta)
            target_pos.append([last_pos[0] + delta_pos[0], last_pos[1] + delta_pos[1], new_height])
            target_att.append([0, 0, new_theta])

            speeds.append(speed)
        self.frame_counter += 1
        return speeds

    def check_completed_all_goals(self):
        return self.target_index >= len(self.ordered_objs)
    
    def check_completed_single_goal(self):
        return self.finish_counter >= FINISH_COUNTER_THRESHOLD * 30

    def step_and_evaluate(self) -> bool:
        self._step_trajectory()
        
        if self.reached_critical or self.previously_reached_critical:
            self.finish_counter += 1 if not self.ordered_objs[self.target_index] == "G" else 0.34
            self.previously_reached_critical = True

        if EARLY_STOP and len(self.TARGET_POS[0]) > EARLY_STOP_FRAME * 30: # 73
            return True

        if self.check_completed_single_goal():
            self.increment_target()
        if self.check_completed_all_goals():
            return True
        return False
    
    def precompute_trajectory(self):
        self.init_stable_trajectory()

        finished = False
        while not finished:
            finished = self.step_and_evaluate()

        self.has_precomputed_trajectory = True

    def increment_target(self):
        self.target_index += 1
        if self.target_index < len(self.ordered_objs):
            cur_drone_pos = [convert_to_relative((self.TARGET_POS[d][-1][0], self.TARGET_POS[d][-1][1]), self.Theta) for d in range(self.num_drones)]

            self.FINAL_THETA = [angle_between_two_points(cur_drone_pos[0], self.ordered_locs[self.target_index])]
            print(f"self.ordered_locs[self.target_index]: {self.ordered_locs[self.target_index]}")
            self.TARGET_LOCATIONS = convert_array_to_global([self.ordered_locs[self.target_index]], self.Theta)
            self.INIT_THETA = [target_att[-1][2] for target_att in self.TARGET_ATT]
            self.DELTA_THETA = [signed_angular_distance(init_theta, final_theta + self.Theta) for final_theta, init_theta in zip(self.FINAL_THETA, self.INIT_THETA)]

            self.critical_action = self.ordered_objs[self.target_index] if self.target_index < len(self.ordered_objs) else None

            self.frame_counter = 0
            self.finish_counter = 0
            self.previously_reached_critical = False
            self.start_dropping = False

    def setup_simulation(self, drone=DEFAULT_DRONES):
        if DEFAULT_VISION:
            self.env = VisionAviary(drone_model=drone,
                            num_drones=self.num_drones,
                            initial_xyzs=self.INIT_XYZS,
                            initial_rpys=self.INIT_RPYS,
                            physics=DEFAULT_PHYSICS,
                            neighbourhood_radius=10,
                            freq=self.simulation_freq_hz,
                            aggregate_phy_steps=self.AGGR_PHY_STEPS,
                            gui=DEFAULT_GUI,
                            record=DEFAULT_RECORD_VISION,
                            obstacles=DEFAULT_OBSTACLES,
                            )
        else:
            self.env = CtrlAviary(drone_model=drone,
                            num_drones=self.num_drones,
                            initial_xyzs=self.INIT_XYZS,
                            initial_rpys=self.INIT_RPYS,
                            physics=DEFAULT_PHYSICS,
                            neighbourhood_radius=10,
                            freq=self.simulation_freq_hz,
                            aggregate_phy_steps=self.AGGR_PHY_STEPS,
                            gui=DEFAULT_GUI,
                            record=DEFAULT_RECORD_VISION,
                            obstacles=DEFAULT_OBSTACLES,
                            user_debug_gui=DEFAULT_USER_DEBUG_GUI,
                            custom_obj_location={
                                    "colors": self.ordered_objs,
                                    "locations": self.obj_loc_global
                                }
                            )
        self.env.IMG_RES = np.array([256, 144])

        #### Obtain the PyBullet Client ID from the environment ####
        PYB_CLIENT = self.env.getPyBulletClient()

        if drone in [DroneModel.CF2X, DroneModel.CF2P]:
            self.ctrl = [DSLPIDControl(drone_model=drone) for i in range(self.num_drones)]
        elif drone in [DroneModel.HB]:
            self.ctrl = [SimplePIDControl(drone_model=drone) for i in range(self.num_drones)]

        #! Simulation Params
        self.CTRL_EVERY_N_STEPS = int(np.floor(self.env.SIM_FREQ / self.control_freq_hz)) # 1
        self.REC_EVERY_N_STEPS = int(np.floor(self.env.SIM_FREQ / self.record_freq_hz )) #30 #240
        self.action = {str(i): np.array([0, 0, 0, 0]) for i in range(self.num_drones)}

        if self.has_precomputed_trajectory:
            ACTIVE_WP = len(self.TARGET_POS[0])
            self.TARGET_POS = np.array(self.TARGET_POS)
            self.TARGET_ATT = np.array(self.TARGET_ATT)
            
            self.STEPS = int(self.CTRL_EVERY_N_STEPS * ACTIVE_WP)
        self.env.reset()

        self.record_counter = 0
        self.simulation_counter = 0

        self.pos_array = []
        self.vel_array = []
        self.timestepwise_displacement_array = []
        self.vel_cmds = []

        self.logger = Logger(logging_freq_hz=4,
                num_drones=self.num_drones,
                output_folder="/".join(self.sim_dir.split('/')[:-1]),
                colab=DEFAULT_COLAB,
                )
        
        os.makedirs(os.path.join(self.sim_dir, "pybullet_pics0"), exist_ok=True)

    def check_exausted_steps(self):
        if self.simulation_counter >= self.STEPS:
            self.env.close()
            return True
        return False
    
    def get_x_y_z_yaw_relative_to_base_env(self, state):
        # Convert to relative drone coordinates by factoring in self.Theta
        relative_state = convert_to_relative([state[0], state[1]], self.Theta)
        return np.array([relative_state[0], relative_state[1], state[2], state[9]])
    
    def get_vx_vy_vz_yawrate(self, state):
        global_vx = state[10]
        global_vy = state[11]
        yaw = state[9]

        # convert from world_frame to body_frame
        rel_vx = global_vx * np.cos(yaw) + global_vy * np.sin(yaw)
        rel_vy = -global_vx * np.sin(yaw) + global_vy * np.cos(yaw)

        return np.array([rel_vx, rel_vy, state[12], state[15]])

    def step_simulation(self):
        # print(f"self.simulation_counter: {self.simulation_counter}, and steps: {self.STEPS}")
        recorded_image = False
        while recorded_image == False and self.simulation_counter < self.STEPS:
            obs, reward, done, info = self.env.step(self.action)

            #* Compute step-by-step velocities for 240hz-trajectory; Control Frequency is 240hz
            if self.simulation_counter % self.CTRL_EVERY_N_STEPS == 0:
                for j in range(self.num_drones):
                    state = obs[str(j)]["state"]
                    self.action[str(j)], _, _ = self.ctrl[j].computeControlFromState(
                        control_timestep=self.CTRL_EVERY_N_STEPS * self.env.TIMESTEP,
                        state=state,
                        target_pos = self.TARGET_POS[j, self.simulation_counter],
                        target_rpy = self.TARGET_ATT[j, self.simulation_counter]
                        )

            #* Network Frequency is 30hz
            if self.simulation_counter % self.REC_EVERY_N_STEPS == 0 and self.simulation_counter>self.env.SIM_FREQ:
                for d in range(self.num_drones):
                    rgb, dep, seg = self.env._getDroneImages(d)
                    self.env._exportImage(img_type=ImageType.RGB,
                                    img_input=rgb,
                                    path=self.sim_dir + f"/pybullet_pics{d}",
                                    frame_num=int(self.simulation_counter / self.REC_EVERY_N_STEPS),
                                    )
                    self.logger.log(drone=d,
                           timestamp=int(self.simulation_counter / self.REC_EVERY_N_STEPS),
                           state=obs[str(d)]["state"],
                           control=np.hstack(
                               [self.TARGET_POS[d, self.simulation_counter, 0:2], self.INIT_XYZS[j, 2], self.INIT_RPYS[j, :], np.zeros(6)])
                           )
                recorded_image = True

            self.simulation_counter += 1
        return state

    def dynamic_step_simulation(self, vel_cmd):
        self.vel_cmds.append(vel_cmd)
        if len(self.pos_array) == 0:
            obs, reward, done, info = self.env.step(self.action)
            state = obs[str(0)]["state"]
            self.pos_array.append(self.get_x_y_z_yaw_relative_to_base_env(state))
            self.vel_array.append(self.get_vx_vy_vz_yawrate(state))

        updated_action = False
        while (self.simulation_counter % self.REC_EVERY_N_STEPS) != 0 or not updated_action:
            obs, reward, done, info = self.env.step(self.action)
            state = obs[str(0)]["state"]
            yaw = state[9]

            #* Network Frequency is 30hz
            if self.simulation_counter % self.REC_EVERY_N_STEPS == 0 and self.simulation_counter>self.env.SIM_FREQ:
                for d in range(self.num_drones):
                    rgb, dep, seg = self.env._getDroneImages(d)
                    self.env._exportImage(img_type=ImageType.RGB,
                                    img_input=rgb,
                                    path=self.sim_dir + f"/pybullet_pics{d}",
                                    frame_num=int(self.simulation_counter / self.REC_EVERY_N_STEPS),
                                    )

                self.vel_cmd_world = copy.deepcopy(vel_cmd)
                # convert from body_frame to world_frame
                self.vel_cmd_world[0] = vel_cmd[0] * np.cos(-yaw) + vel_cmd[1] * np.sin(-yaw)
                self.vel_cmd_world[1] = -vel_cmd[0] * np.sin(-yaw)+ vel_cmd[1] * np.cos(-yaw)

                self.action[str(0)], _, _ = self.ctrl[0].computeControl(control_timestep=self.CTRL_EVERY_N_STEPS * self.env.TIMESTEP,
                                                    cur_pos=state[0:3],
                                                    cur_quat=state[3:7],
                                                    cur_vel=state[10:13],
                                                    cur_ang_vel=state[13:16],
                                                    target_pos=state[0:3],  # same as the current position
                                                    target_rpy=np.array([0, 0, state[9]]),  # keep current yaw
                                                    target_vel=self.vel_cmd_world[0:3],
                                                    target_rpy_rates=np.array([0, 0, vel_cmd[3]])
                                                    )
                
                self.pos_array.append(self.get_x_y_z_yaw_relative_to_base_env(state))
                self.vel_array.append(self.get_vx_vy_vz_yawrate(state))
                if len(self.pos_array) > 1:
                    self.timestepwise_displacement_array.append(self.pos_array[-1] - self.pos_array[-2])
                
                updated_action = True
                updated_state = state.copy()
                
            #* Compute step-by-step velocities for 240hz-trajectory; Control Frequency is 240hz
            if self.simulation_counter % self.CTRL_EVERY_N_STEPS == 0:
                self.action[str(0)], _, _ = self.ctrl[0].computeControl(control_timestep=self.CTRL_EVERY_N_STEPS * self.env.TIMESTEP,
                                                    cur_pos=state[0:3],
                                                    cur_quat=state[3:7],
                                                    cur_vel=state[10:13],
                                                    cur_ang_vel=state[13:16],
                                                    target_pos=state[0:3],  # same as the current position
                                                    target_rpy=np.array([0, 0, state[9]]),  # keep current yaw
                                                    target_vel=self.vel_cmd_world[0:3],
                                                    target_rpy_rates=np.array([0, 0, vel_cmd[3]])
                                                    )

            self.simulation_counter += 1
        return updated_state

    def get_latest_displacement(self):
        return self.timestepwise_displacement_array[-1]
                    
    def run_simulation_to_completion(self):
        self.setup_simulation()
        while not self.check_exausted_steps():
            state = self.step_simulation()
            self.pos_array.append(self.get_x_y_z_yaw_relative_to_base_env(state))
            self.vel_array.append(self.get_vx_vy_vz_yawrate(state))

            if len(self.pos_array) > 1:
                self.timestepwise_displacement_array.append(self.pos_array[-1] - self.pos_array[-2])

    def export_plots(self):
        x_data = np.array(self.pos_array)[:, 0]
        y_data = np.array(self.pos_array)[:, 1]
        yaw = np.array(self.pos_array)[:, 3]
        
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(x_data, y_data)
        axs[0].set_aspect('equal', adjustable='box')
        
        for target in self.obj_loc_global:
            rel_target = convert_to_relative(target, self.Theta)
            axs[0].plot(rel_target[0], rel_target[1], 'ro')

        axs[0].legend(["Leader", "Follower"])
        
        axs[1].plot(yaw)
        axs[1].set_ylabel('Yaw')

        fig.savefig(self.sim_dir + "/path.jpg")

        np.savetxt(os.path.join(self.sim_dir, 'pos.csv'), np.array(self.pos_array), delimiter=',')
        np.savetxt(os.path.join(self.sim_dir, 'vel.csv'), np.array(self.vel_array), delimiter=',')
        np.savetxt(os.path.join(self.sim_dir, 'timestepwise_displacement.csv'), np.array(self.timestepwise_displacement_array), delimiter=',')
        np.savetxt(os.path.join(self.sim_dir, 'vel_cmds.csv'), np.array(self.vel_cmds), delimiter=',')

def generate_one_dynamic_training_trajectory(obj_color):
    rel_obj = [(random.uniform(1, 2), 0)]
    
    output_folder = f'train_d0'
    sim_name = "save-flight-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S.%f") # include milliseconds in save name for parallel runs
    sim_dir = os.path.join(output_folder, sim_name)
    setup_folders(sim_dir, DEFAULT_NUM_DRONES)

    with open(os.path.join(sim_dir, 'colors.txt'), 'w') as f:
        f.write(str("".join(obj_color)))

    sim = Simulator(obj_color, rel_obj, sim_dir)
    with open(os.path.join(sim_dir, 'theta.txt'), 'w') as f:
        f.write("P" if sim.Theta_offset > 0 else "N")
    with open(os.path.join(sim_dir, 'start_h.txt'), 'w') as f:
        f.write(str(sim.start_H))
    with open(os.path.join(sim_dir, 'start_dist.txt'), 'w') as f:
        f.write(str(rel_obj[0][0]))
    
    sim.precompute_trajectory()

    print("Running simulation")
    sim.run_simulation_to_completion()

    print("Saving data as csvs")
    np.savetxt(os.path.join(sim_dir, 'pos.csv'), np.array(sim.pos_array), delimiter=',')
    np.savetxt(os.path.join(sim_dir, 'vel.csv'), np.array(sim.vel_array), delimiter=',')
    np.savetxt(os.path.join(sim_dir, 'timestepwise_displacement.csv'), np.array(sim.timestepwise_displacement_array), delimiter=',')

    x_data = np.array(sim.pos_array)[:, 0]
    y_data = np.array(sim.pos_array)[:, 1]
    
    fig, ax = plt.subplots()
    ax.plot(x_data, y_data)
    
    for target in sim.obj_loc_global:
        rel_target = convert_to_relative(target, sim.Theta)
        ax.plot(rel_target[0], rel_target[1], 'ro')

    ax.legend(["Leader", "Follower"])
    fig.savefig(sim_dir + "/path.jpg")

    sim.logger.save_as_csv(sim_name)  # Optional CSV save

def main():
    obj_color = [random.choice(['R', 'B'])]
    generate_one_dynamic_training_trajectory(obj_color)

if __name__ == "__main__":
    main()
