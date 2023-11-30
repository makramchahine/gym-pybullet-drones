import os
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import ImageType

from culekta_utils import *
from simulator_utils import *
from simulator import BaseSimulator

FINISH_COUNTER_THRESHOLD = 32
RANDOM_WALK = 0.25

class TrainSimulator(BaseSimulator):
    def __init__(self, ordered_objs, ordered_rel_locs, sim_dir, start_H, target_Hs, Theta, Theta_offset, record_hz):
        super().__init__(ordered_objs, ordered_rel_locs, sim_dir, start_H, target_Hs, Theta, Theta_offset, record_hz)
        self.custom_timesteps = []

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
                yaw_speed = yaw_dist / self.control_freq_hz / 2
                lift_speed = height_dist / self.control_freq_hz / 2
                # yaw_speed = DEFAULT_SEARCHING_YAW * np.sign(yaw_dist) / self.control_freq_hz
                # lift_speed = 0 if (abs(height_dist) < APPROX_CORRECT_HEIGHT) else STABILIZE_LIFT_MULTIPLIER * (height_dist / height) / self.control_freq_hz

                if abs(yaw_dist) < APPROX_CORRECT_YAW:
                    new_theta = final_theta + self.Theta
                else:
                    new_theta = last_yaw + yaw_speed
                self.FINAL_THETA[0] = angle_between_two_points(last_pos[:2], final_target[:2]) - self.Theta
            elif dist > self.critical_dist and not self.previously_reached_critical:
                speed = interpolate_speeds(dist, self.critical_dist, self.critical_dist_buffer, DEFAULT_CRITICAL_SPEED, DEFAULT_SPEED) / self.control_freq_hz
                yaw_speed = yaw_dist / self.control_freq_hz / 2
                lift_speed = height_dist / self.control_freq_hz / 2
                # yaw_speed = interpolate_speeds(dist, self.critical_dist, self.critical_dist_buffer, DEFAULT_CRITICAL_YAW_SPEED, DEFAULT_SEARCHING_YAW) * np.sign(yaw_dist) / self.control_freq_hz
                # lift_speed = 0 if (abs(height_dist) < APPROX_CORRECT_HEIGHT) else STABILIZE_LIFT_MULTIPLIER * (height_dist / height) / self.control_freq_hz

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

    def check_exausted_steps(self):
        if self.simulation_counter >= self.STEPS:
            self.env.close()
            return True
        return False

    def evaluate_trajectory(self) -> bool:
        if self.check_completed_all_goals():
            if self.frame_counter > (64 + 8) * self.simulation_freq_hz / self.record_freq_hz:
                return True
            return False
        
        if self.reached_critical or self.previously_reached_critical:
            self.finish_counter += 1 if not self.ordered_objs[self.target_index] == "G" else 0.34
            self.previously_reached_critical = True

        if self.check_completed_single_goal():
            self.increment_target()
        if self.check_completed_all_goals() and (self.frame_counter > (64 + 8) * self.simulation_freq_hz / self.record_freq_hz):
            print(f"Finished trajectory: {self.frame_counter}")
            return True
        return False
    
    def precompute_trajectory(self):
        self.init_stable_trajectory()

        finished = False
        while not finished:
            self._step_trajectory()
            finished = self.evaluate_trajectory()

        self.has_precomputed_trajectory = True

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

    def step_simulation(self):
        recorded_image = False
        from scipy.stats import norm
        # x = np.linspace(0, 216, 217)
        # weights = 3 * norm.pdf(x, loc=21, scale=30) + norm.pdf(x, loc=217-21, scale=100)
        # weights /= weights.sum()
        # num_frames_until_record = np.random.choice(np.arange(24, 241), p=weights)
        num_frames_until_record = 79 #26#79
        step_counter = 0
        while recorded_image == False and self.simulation_counter < self.STEPS:
            obs, reward, done, info = self.env.step(self.action)

            #* Compute step-by-step velocities for 240hz-trajectory; Control Frequency is 240hz
            if self.simulation_counter % self.CTRL_EVERY_N_STEPS == 0:
                for j in range(self.num_drones):
                    state = obs[str(j)]["state"]
                    random_walk_state = self.add_noise_to_state(state.copy())
                    self.action[str(j)], _, _ = self.ctrl[j].computeControlFromState(
                        control_timestep=self.CTRL_EVERY_N_STEPS * self.env.TIMESTEP,
                        state=random_walk_state if RANDOM_WALK else state,
                        target_pos = self.TARGET_POS[j, self.simulation_counter],
                        target_rpy = self.TARGET_ATT[j, self.simulation_counter]
                        )

            #* Network Frequency is 30hz
            if step_counter >= num_frames_until_record and self.simulation_counter>self.env.SIM_FREQ:
                for d in range(self.num_drones):
                    rgb, dep, seg = self.env._getDroneImages(d)
                    self.env._exportImage(img_type=ImageType.RGB,
                                    img_input=rgb,
                                    path=self.sim_dir + f"/pybullet_pics{d}",
                                    frame_num=int(self.simulation_counter),
                                    )
                    self.logger.log(drone=d,
                           timestamp=round(self.simulation_counter),
                           state=random_walk_state if RANDOM_WALK else obs[str(d)]["state"],
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
            state = self.step_simulation()
            self.global_pos_array.append(get_x_y_z_yaw_relative_to_base_env(state, self.Theta))
            self.vel_array.append(get_vx_vy_vz_yawrate_rel_to_self(state))

            if len(self.global_pos_array) > 1:
                global_disp = self.global_pos_array[-1] - self.global_pos_array[-2]
                rel_disp = get_relative_displacement(self.global_pos_array[-1], self.global_pos_array[-2], -self.Theta)
                # print(global_disp - rel_disp)
                self.timestepwise_displacement_array.append(rel_disp)
