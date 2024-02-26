import os
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import ImageType

from culekta_utils import *
from simulator_utils import *
from gym_pybullet_drones.examples.simulator_base import BaseSimulator

from gym_pybullet_drones.examples.schemas import InitConditionsSchema

FINISH_COUNTER_THRESHOLD = 1.5 # in seconds

class EvalSimulator(BaseSimulator):
    def __init__(self, sim_dir: str, init_conditions: InitConditionsSchema, record_hz: str):
        super().__init__(sim_dir, init_conditions, record_hz)

    def check_completed_all_goals(self):
        return self.target_index >= len(self.objects_relative_target)
    
    def check_completed_single_goal(self):
        return self.finish_counter >= FINISH_COUNTER_THRESHOLD * self.record_freq_hz

    def check_exausted_steps(self):
        if self.simulation_counter >= self.STEPS:
            self.env.close()
            return True
        return False


    def dynamic_step_simulation(self, vel_cmd):
        self.vel_cmds.append(vel_cmd)

        updated_action = False
        while (self.simulation_counter % self.REC_EVERY_N_STEPS) != 0 or not updated_action:
            # print(f"self.simulation_counter: {self.simulation_counter}, and mod: {self.simulation_counter % self.REC_EVERY_N_STEPS}, and updated_action: {updated_action}")
            obs, reward, done, info = self.env.step(self.action)
            state = obs[str(0)]["state"]
            yaw = state[9]

            #* Network Frequency is 30hz
            if self.simulation_counter % self.REC_EVERY_N_STEPS == 0: # and self.simulation_counter>=self.env.SIM_FREQ:
                for d in range(self.num_drones):
                    rgb, dep, seg = self.env._getDroneImages(d)
                    self.env._exportImage(img_type=ImageType.RGB,
                                    img_input=rgb,
                                    path=self.sim_dir + f"/pybullet_pics{d}",
                                    frame_num=int(self.simulation_counter / self.CTRL_EVERY_N_STEPS),
                                    )

                self.vel_cmd_world = convert_vel_cmd_to_world_frame(vel_cmd, yaw)
                
                self.global_pos_array.append(get_x_y_z_yaw_relative_to_base_env(state, self.theta_environment))
                # self.global_pos_array.append(self.get_x_y_z_yaw_global(state))

                self.vel_array.append(get_vx_vy_vz_yawrate_rel_to_self(state))
                if len(self.global_pos_array) > 1:
                    # self.timestepwise_displacement_array.append(self.global_pos_array[-1] - self.global_pos_array[-2])
                    self.timestepwise_displacement_array.append(get_relative_displacement(self.global_pos_array[-1], self.global_pos_array[-2], -self.theta_environment))
                
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
        self.evaluate_completed_single_task(state)
        finished = self.check_completed_all_goals()
        return updated_state, rgb, finished

    def safe_update_target_index(self):
        self.target_index += 1
        if self.target_index >= len(self.objects_relative_target):
            self.target_index = len(self.objects_relative_target) - 1

    # TODO: Fix automated evaluation
    def evaluate_completed_single_task(self, state):
        # extract positions
        x, y, z = state[0], state[1], state[2]
        yaw = state[9]

        print(f"\nTarget: {self.objects_relative_target[self.target_index]}")
        if object_in_view(x, y, yaw, self.objects_relative_target[self.target_index]):
            self.alive_obj_previously_in_view = True
            return

        if not object_in_view(x, y, yaw, self.objects_relative_target[self.target_index]) and self.alive_obj_previously_in_view:
            # Correct turn
            if (self.objects_relative_target[self.target_index] == 'R' and drone_turned_left(x, y, yaw, self.objects_relative_target[self.target_index]) or (self.objects_relative_target[self.target_index] == 'B' and drone_turned_right(x, y, yaw, self.objects_relative_target[self.target_index]))):
                print(f"Correct turn for {self.target_index}")
                # Remove ball then add next object
                # if self.vanish_mode:
                #     self.env.removeObject(self.alive_obj_id)
                #     print(f"Removed Item: {self.alive_obj_id}")
                #     if not (self.target_index > len(self.ordered_objs) - 1):
                #         self.alive_obj_id = self.env.addObject(self.ordered_objs[self.target_index], self.objects_relative_target[self.target_index])
                #         print(f"New Item: {self.alive_obj_id}")

                self.window_outcomes.append(self.objects_relative_target[self.target_index])
                # self.target_index += 1
                self.safe_update_target_index()
            elif self.objects_relative_target[self.target_index] == 'R' or self.objects_relative_target[self.target_index] == 'B':
                self.window_outcomes.append("N")
                # self.target_index += 1
                self.safe_update_target_index()
            self.alive_obj_previously_in_view = False

    def export_plots(self):
        super().export_plots()

        try:
            with open(os.path.join(self.sim_dir, 'finish.txt'), 'w') as f:
                max_len = max(len(self.window_outcomes), len(self.objects_relative_target))
                # pad self.window_outcomes, self.ordered_objs with X's if they are too short
                self.window_outcomes = self.window_outcomes + ['X'] * (max_len - len(self.window_outcomes))
                self.objects_relative_target = self.objects_relative_target + ['X'] * (max_len - len(self.objects_relative_target))

                for window_outcome, color in zip(self.window_outcomes, self.objects_relative_target):
                    f.write(f"{window_outcome},{color}\n")
        except Exception as e:
            print(e)
