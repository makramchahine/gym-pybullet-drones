import os
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from scipy.stats import norm
from copy import deepcopy

from gym_pybullet_drones.utils.enums import DroneModel, ImageType
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl

from simulator.culekta_utils import *
from simulator.simulator_utils import *
from simulator.simulator_base import BaseSimulator
from simulator.default_pyb_settings import *

FINISH_COUNTER_THRESHOLD = 32
RANDOM_WALK = True
TARGET_NUM_TIMESTEPS_TO_CRITICAL = (55, 90) # this affects the rate of drone control recovery
CONTROL_STEP_NORMALIZATION = 3

class TrainSimulator(BaseSimulator):
    def __init__(self, sim_dir, init_conditions, record_hz, task_tag):
        super().__init__(sim_dir, init_conditions, record_hz)
        self.turn_mode = True if task_tag == "fly_and_turn" else False
        self.num_frames = random.randint(TARGET_NUM_TIMESTEPS_TO_CRITICAL[0], TARGET_NUM_TIMESTEPS_TO_CRITICAL[1])

        self.dist_0_x = self.start_dist
        self.dist_0_yaw = 0 - self.theta_offset
        self.dist_0_z = self.start_height - self.INIT_XYZS[0, 2]

        num_control_steps = self.num_frames / CONTROL_STEP_NORMALIZATION * self.control_freq_hz
        self.eta_x_per_control = (self.dist_0_x - 0.5) / num_control_steps
        self.eta_yaw_per_control = self.dist_0_yaw / num_control_steps
        self.eta_z_per_control = self.dist_0_z / num_control_steps

        # Dynamics intended for generation of training data
        self.I_X = random.uniform(0.025, 0.075)
        self.P_X = 0.18 #random.uniform(0.18, 0.2)
        self.P_YAW = 0.2 #0.1 for 1 dist
        self.P_Z = 0.2
        
        self.custom_timesteps = []
        self.checkpoint_frame = -1
    
    def get_adj_yaw_speed(self, yaw_dist):
        yaw_speed = yaw_dist / self.control_freq_hz * self.P_YAW
        if abs(yaw_dist) > abs(self.eta_yaw_per_control * 0.075):
            yaw_speed += self.eta_yaw_per_control * 0.075
        return yaw_speed
    
    def get_adj_z_speed(self, z_dist):
        z_speed = z_dist / self.control_freq_hz * self.P_Z
        if abs(z_dist) > abs(self.eta_z_per_control * 0.075):
            z_speed += self.eta_z_per_control * 0.075
        return z_speed
    
    def get_adj_x_speed(self, x_dist):
        x_speed = x_dist / self.control_freq_hz * self.P_X
        if abs(x_dist) > abs(self.eta_x_per_control * self.I_X):
            x_speed += self.eta_x_per_control * self.I_X
        return x_speed

    def init_stable_trajectory(self):
        """ Holds still for one second of simulation """
        for i in range(self.simulation_freq_hz):
            self._step_trajectory(hold=True)

    def _compute_turn_effects(self, last_pos, final_target, yaw_speed, lift_speed):
        if self.critical_action == 'R':
            yaw_speed += DEFAULT_CRITICAL_YAW_SPEED / self.control_freq_hz
        elif self.critical_action == 'B':
            yaw_speed += -DEFAULT_CRITICAL_YAW_SPEED / self.control_freq_hz
        elif self.critical_action == 'G':
            lift_speed = DEFAULT_LIFT_SPEED / self.control_freq_hz
            if not self.previously_reached_critical:
                self.FINAL_THETA[0] = angle_between_two_points(last_pos[:2], final_target[:2]) - self.theta_environment # continue to face target
        else:
            assert False, f"critical_action: {self.critical_action}"

        return yaw_speed, lift_speed

    def _get_adj_speeds(self, dist, yaw_dist, height_dist):
        speed = self.get_adj_x_speed(dist)
        yaw_speed = self.get_adj_yaw_speed(yaw_dist)
        lift_speed = self.get_adj_z_speed(height_dist)
        return speed, yaw_speed, lift_speed

    def _step_trajectory(self, hold=False):
        """
        Modifies:
            self.TARGET_POS, self.TARGET_ATT, self.FINAL_THETA, self.reached_critical
        """
        speeds = []
        print(f"self.objects_absolute_target: {self.objects_absolute_target}")
        for i, (target_pos, target_att, init_theta, final_theta, final_target) in enumerate(zip(self.TARGET_POS, self.TARGET_ATT, self.INIT_THETA, self.FINAL_THETA, self.objects_absolute_target)):
            last_pos = target_pos[-1]    # X, Y, Z
            last_yaw = target_att[-1][2] # R, P, Y
            last_height = target_pos[-1][2]
            dist = distance_to_target(last_pos, final_target)
            yaw_dist = signed_angular_distance(last_yaw, final_theta + self.theta_environment)
            height_dist = self.target_height - last_height

            dist_to_crit = dist - 0.49

            lift_speed = 0
            if hold:
                new_theta = init_theta
                speed, yaw_speed, lift_speed = (0, 0, 0)
            elif dist > self.critical_dist + self.critical_dist_buffer and not self.previously_reached_critical:
                speed, yaw_speed, lift_speed = self._get_adj_speeds(dist_to_crit, yaw_dist, height_dist)
                new_theta = final_theta + self.theta_environment if abs(yaw_dist) < APPROX_CORRECT_YAW else last_yaw + yaw_speed
                
                self.FINAL_THETA[0] = angle_between_two_points(last_pos[:2], final_target[:2]) - self.theta_environment
            elif dist > self.critical_dist and not self.previously_reached_critical:
                speed, yaw_speed, lift_speed = self._get_adj_speeds(dist_to_crit, yaw_dist, height_dist)
                new_theta = final_theta + self.theta_environment if abs(yaw_dist) < APPROX_CORRECT_YAW else last_yaw + yaw_speed
            else:
                if self.checkpoint_frame == -1:
                    self.checkpoint_frame = len(target_pos)
                speed = 0
                yaw_speed = DEFAULT_SEARCHING_YAW * np.sign(yaw_dist) / self.control_freq_hz
                if self.turn_mode:
                    yaw_speed, lift_speed = self._compute_turn_effects(last_pos, final_target, yaw_speed, lift_speed)
                new_theta = last_yaw + yaw_speed


            if self.critical_action == 'G':
                delta_z = lift_speed if not self.previously_reached_critical else -DROP_SPEED / self.control_freq_hz * (last_height / DROP_MAX_HEIGHT)
                self.reached_critical = dist < DEFAULT_DROP_POINT_DIST
                new_height = max(last_height + delta_z, self.target_height)
            else:
                delta_z = lift_speed
                self.reached_critical = dist < self.critical_dist
                new_height = (last_height + delta_z) if (lift_speed != 0 or hold) else self.target_height
        
            delta_pos = convert_to_global([speed, 0], new_theta)
            target_pos.append([last_pos[0] + delta_pos[0], last_pos[1] + delta_pos[1], new_height])
            target_att.append([0, 0, new_theta])

            speeds.append(speed)
        self.frame_counter += 1
        return speeds

    def check_completed_all_goals(self):
        return self.target_index >= len(self.objects_color_target)
    
    def check_completed_single_goal(self):
        return self.finish_counter >= FINISH_COUNTER_THRESHOLD * 30

    def check_exausted_steps(self):
        if self.simulation_counter >= self.STEPS:
            self.env.close()
            return True
        return False

    def evaluate_trajectory(self) -> bool:
        if self.check_completed_all_goals():
            if self.frame_counter > (64 + 8) * self.simulation_freq_hz / CONTROL_STEP_NORMALIZATION:
                return True
            return False
        
        if self.reached_critical or self.previously_reached_critical:
            self.finish_counter += 1 if not self.objects_color_target[self.target_index] == "G" else 0.34
            self.previously_reached_critical = True

        if self.check_completed_single_goal():
            self.increment_target()
        if self.check_completed_all_goals() and (self.frame_counter > (64 + 8) * self.simulation_freq_hz / CONTROL_STEP_NORMALIZATION):
            print(f"Finished trajectory: {self.frame_counter}")
            return True
        return False
    
    def precompute_trajectory(self):
        self.init_stable_trajectory()

        finished = False
        while not finished:
            self._step_trajectory()
            finished = self.evaluate_trajectory()

        if RANDOM_WALK:
            print("self.checkpoint_frame", self.checkpoint_frame)
            print("len(self.TARGET_POS[0])", len(self.TARGET_POS[0]))
            self.add_noise_to_targets()
        self.has_precomputed_trajectory = True

    def add_noise_to_targets(self):
        new_mean = 0 #random.uniform(-0.15, 0.15)
        xyz_noise_matrix = np.random.normal(0, 0.01, size=(3, self.checkpoint_frame))
        xyz_noise_matrix -= np.mean(xyz_noise_matrix, axis=1, keepdims=True)
        yaw_noise_matrix = np.random.normal(new_mean, 0.001, size=(1, self.checkpoint_frame))
        yaw_noise_matrix -= np.mean(yaw_noise_matrix, axis=1, keepdims=True)

        self.TARGET_POS = np.array(self.TARGET_POS)
        self.TARGET_ATT = np.array(self.TARGET_ATT)
        self.TARGET_POS[0, :self.checkpoint_frame, 0:3] += xyz_noise_matrix.T
        self.TARGET_ATT[0, :self.checkpoint_frame, 2] += yaw_noise_matrix[0].T

    def add_noise_to_state(self, state):
        vx, vy, vz, yaw_rate = state[4], state[5], state[6], state[12]
        x, y, z, yaw = state[0], state[1], state[2], state[9]

        C = 0.1
        if RANDOM_WALK:
            state[0] += vx * C * np.random.normal(0, 1)
            state[1] += vy * C * np.random.normal(0, 1)
            state[2] += vz * C * np.random.normal(0, 1)
            state[9] += yaw_rate * C * np.random.normal(0, 1)

        return state

    def set_num_steps_until_record(self, hz):
        if type(hz) == int:
            # 79 is 3Hz; 26 is 9Hz
            num_steps_until_record = round(self.control_freq_hz / hz) - 1
        elif hz == "1-10":
            x = np.linspace(0, 216, 217)
            weights = 3 * norm.pdf(x, loc=21, scale=30) + norm.pdf(x, loc=217-21, scale=100)
            weights /= weights.sum()
            num_steps_until_record = np.random.choice(np.arange(24, 241), p=weights)
        else:
            raise ValueError(f"Incorrect hz value: {hz}")
        return num_steps_until_record

    def step_simulation(self, hz):
        num_steps_until_record = self.set_num_steps_until_record(hz)

        recorded_image = False
        step_counter = 0
        while recorded_image == False and self.simulation_counter < self.STEPS:
            obs, reward, done, info = self.env.step(self.action)

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
            if step_counter >= num_steps_until_record and self.simulation_counter>self.env.SIM_FREQ:
                for d in range(self.num_drones):
                    rgb, dep, seg = self.env._getDroneImages(d)
                    self.env._exportImage(img_type=ImageType.RGB,
                                    img_input=rgb,
                                    path=self.sim_dir + f"/pybullet_pics{d}",
                                    frame_num=int(self.simulation_counter),
                                    )
                    self.logger.log(drone=d,
                           timestamp=round(self.simulation_counter),
                           state=obs[str(d)]["state"],
                           control=np.hstack(
                               [self.TARGET_POS[d, self.simulation_counter, 0:2], self.INIT_XYZS[j, 2], self.INIT_RPYS[j, :], np.zeros(6)])
                           )
                recorded_image = True
                self.custom_timesteps.append(self.simulation_counter *  1.0 / self.simulation_freq_hz)

            self.simulation_counter += 1
            step_counter += 1
        return state

    def run_simulation_to_completion(self):
        """ Creates training images with the stored trajectory """
        self.setup_simulation()

        while not self.check_exausted_steps():
            state = self.step_simulation(self.record_freq_hz)
            self.global_pos_array.append(get_x_y_z_yaw_relative_to_base_env(state, self.theta_environment))
            self.vel_array.append(get_vx_vy_vz_yawrate_rel_to_self(state))

            if len(self.global_pos_array) > 1:
                rel_disp = get_relative_displacement(self.global_pos_array[-1], self.global_pos_array[-2], -self.theta_environment)
                self.timestepwise_displacement_array.append(rel_disp)

    
    def run_recon(self):
        self.setup_simulation()
        custom_obj_location = {
            "colors": self.objects_color,
            "locations": self.objects_absolute
        }
        
        positions = np.loadtxt(self.sim_dir + "/sim_pos.csv", delimiter=",")
        os.makedirs(self.sim_dir + f"/recon_pics0", exist_ok=True)

        def get_absolute_position(state, Theta):
            relative_state = convert_to_global([state[0], state[1]], Theta)
            return np.array([relative_state[0], relative_state[1], state[2], state[3]])
        
        for i in range(len(positions)):
            absolute_position = get_absolute_position(positions[i], self.theta_environment)
            pos = absolute_position[:3].reshape(1, 3)
            rpy = np.array([[0, 0, absolute_position[3]]])

            drone = DEFAULT_DRONES
            self.env = CtrlAviary(drone_model=drone,
                num_drones=self.num_drones,
                initial_xyzs=pos,
                initial_rpys=rpy,
                physics=DEFAULT_PHYSICS,
                neighbourhood_radius=10,
                freq=self.simulation_freq_hz,
                aggregate_phy_steps=self.AGGR_PHY_STEPS,
                gui=DEFAULT_GUI,
                record=DEFAULT_RECORD_VISION,
                obstacles=DEFAULT_OBSTACLES,
                user_debug_gui=DEFAULT_USER_DEBUG_GUI,
                custom_obj_location=custom_obj_location
                )
            self.env.IMG_RES = np.array([256, 144])
            PYB_CLIENT = self.env.getPyBulletClient()

            if drone in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = [DSLPIDControl(drone_model=drone) for i in range(self.num_drones)]
            elif drone in [DroneModel.HB]:
                self.ctrl = [SimplePIDControl(drone_model=drone) for i in range(self.num_drones)]

            rgb, dep, seg = self.env._getDroneImages(0)
            self.env._exportImage(img_type=ImageType.RGB,
                                    img_input=rgb,
                                    path=self.sim_dir + f"/recon_pics0",
                                    frame_num=int(i+1),
                                    )
            self.env.close()