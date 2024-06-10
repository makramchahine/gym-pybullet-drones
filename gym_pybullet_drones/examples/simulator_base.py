import os
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from copy import deepcopy

from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.examples.schemas import InitConditionsSchema


from gym_pybullet_drones.examples.culekta_utils import *
from gym_pybullet_drones.examples.simulator_utils import *
from gym_pybullet_drones.examples.default_pyb_settings import *
aligned_follower = True

CRITICAL_DIST = 0.5
CRITICAL_DIST_BUFFER = 0.1

class BaseSimulator():
    def __init__(self, sim_dir: str, init_conditions: InitConditionsSchema, record_hz: str):
        self.num_drones = DEFAULT_NUM_DRONES
        self.sim_dir = sim_dir
        self.simulation_freq_hz = DEFAULT_SIMULATION_FREQ_HZ
        self.control_freq_hz = DEFAULT_CONTROL_FREQ_HZ
        self.record_freq_hz = record_hz
        self.duration_sec = DEFAULT_DURATION_SEC
        self.start_height = init_conditions["start_heights"]
        self.target_height = init_conditions["target_heights"]
        self.theta_offset = init_conditions["theta_offset"]
        self.theta_environment = init_conditions["theta_environment"]
        self.objects_color = init_conditions["objects_color"]
        self.objects_relative = init_conditions["objects_relative"]
        self.objects_relative_target = init_conditions.get("objects_relative_target", init_conditions["objects_relative"])
        
        # TODO: Change objects_absolute/objects_absolute_target to be xyz instead of just xy
        self.objects_absolute = [convert_to_global(obj_loc_rel, self.theta_environment) for obj_loc_rel in self.objects_relative]
        self.objects_absolute = [(obj[0], obj[1], self.target_height) for i, obj in enumerate(self.objects_absolute)]
        self.objects_absolute_target = [convert_to_global(obj_loc_rel, self.theta_environment) for obj_loc_rel in self.objects_relative_target]
        self.objects_absolute_target = [(obj[0], obj[1], self.target_height) for i, obj in enumerate(self.objects_absolute_target)]
        self.objects_color_target = init_conditions.get("objects_color_target", init_conditions["objects_color"])
        self.start_dist = init_conditions["start_dist"]
        
        self.drone_theta_0 = self.theta_environment
        self.aggregate = DEFAULT_AGGREGATE
        self.AGGR_PHY_STEPS = int(self.simulation_freq_hz / self.control_freq_hz) if self.aggregate else 1
        self.target_index = 0
        self.frame_counter = 0
        self.finish_counter = 0
        self.reached_critical = False
        self.previously_reached_critical = False
        self.has_precomputed_trajectory = False
        self.alive_obj_previously_in_view = False

        self.rel_drone_locs = [(0, 0)]
        self._init_trajectory_params()
        self.window_outcomes = []

    def _init_trajectory_params(self):
        self.FINAL_THETA = [angle_between_two_points(rel_drone, rel_obj) for rel_drone, rel_obj in zip(self.rel_drone_locs, self.objects_relative_target)]
        self.INIT_XYZS = np.array([[*convert_to_global(rel_pos, self.theta_environment), self.start_height] for rel_pos in self.rel_drone_locs])
        self.INIT_RPYS = np.array([[0, 0, self.drone_theta_0 + self.theta_offset] for d in range(self.num_drones)])

        self.TARGET_POS = [[arry] for arry in self.INIT_XYZS]
        self.TARGET_ATT = [[arry] for arry in self.INIT_RPYS]
        self.INIT_THETA = [init_rpys[2] for init_rpys in self.INIT_RPYS]
        # angular distance between init and final theta
        self.DELTA_THETA = [signed_angular_distance(init_theta, final_theta + self.theta_environment) for final_theta, init_theta in zip(self.FINAL_THETA, self.INIT_RPYS[:, 2])]

        self.critical_action = self.objects_color_target[self.target_index]
        self.critical_dist = CRITICAL_DIST
        self.critical_dist_buffer = CRITICAL_DIST_BUFFER

        if self.check_completed_all_goals():
            if self.frame_counter > (64 + 8) * self.simulation_freq_hz / self.record_freq_hz:
                return True
            return False
        
        if self.reached_critical or self.previously_reached_critical:
            self.finish_counter += 1 if not self.objects_color_target[self.target_index] == "G" else 0.34
            self.previously_reached_critical = True

        if self.check_completed_single_goal():
            self.increment_target()
        if self.check_completed_all_goals() and (self.frame_counter > (64 + 8) * self.simulation_freq_hz / self.record_freq_hz):
            print(f"Finished trajectory: {self.frame_counter}")
            return True
        return False
        
    def increment_target(self):
        self.target_index += 1
        if self.target_index < len(self.objects_color_target):
            cur_drone_pos = [convert_to_relative((self.TARGET_POS[d][-1][0], self.TARGET_POS[d][-1][1]), self.theta_environment) for d in range(self.num_drones)]

            self.FINAL_THETA = [angle_between_two_points(cur_drone_pos[0], self.objects_relative[self.target_index])]
            self.TARGET_LOCATIONS = convert_array_to_global([self.objects_relative[self.target_index]], self.theta_environment)
            self.INIT_THETA = [target_att[-1][2] for target_att in self.TARGET_ATT]
            self.DELTA_THETA = [signed_angular_distance(init_theta, final_theta + self.theta_environment) for final_theta, init_theta in zip(self.FINAL_THETA, self.INIT_THETA)]

            self.critical_action = self.objects_color_target[self.target_index] if self.target_index < len(self.objects_color_target) else None

            self.frame_counter = 0
            self.finish_counter = 0
            self.previously_reached_critical = False
            self.start_dropping = False

    def setup_simulation(self, drone=DEFAULT_DRONES, custom_obj_location=None):
        if custom_obj_location is None:
            custom_obj_location = {
                "colors": self.objects_color,
                "locations": self.objects_absolute
            }
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
                        custom_obj_location=custom_obj_location
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

        self.global_pos_array = []
        self.vel_array = []
        self.timestepwise_displacement_array = []
        self.vel_cmds = []

        self.logger = Logger(logging_freq_hz=self.simulation_freq_hz,
                num_drones=self.num_drones,
                output_folder="/".join(self.sim_dir.split('/')[:-1]),
                colab=DEFAULT_COLAB,
                )
        
        os.makedirs(os.path.join(self.sim_dir, "pybullet_pics0"), exist_ok=True)

        rgb, dep, seg = self.env._getDroneImages(0)
        self.env._exportImage(img_type=ImageType.RGB,
                img_input=rgb,
                path=self.sim_dir + f"/pybullet_pics{0}",
                frame_num=int(self.simulation_counter),
                )

    def export_plots(self):
        x_data = np.array(self.global_pos_array)[:, 0]
        y_data = np.array(self.global_pos_array)[:, 1]
        yaw = np.array(self.global_pos_array)[:, 3]

        x_data_integrated = np.cumsum(np.array(self.timestepwise_displacement_array)[:, 0])
        y_data_integrated = np.cumsum(np.array(self.timestepwise_displacement_array)[:, 1])
        yaw_integrated = np.cumsum(np.array(self.timestepwise_displacement_array)[:, 3])

        # ! Path Plot
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(x_data, y_data, alpha=0.5)
        axs[0].plot(x_data_integrated, y_data_integrated, alpha=0.5)
        axs[0].set_aspect('equal', adjustable='box')
        
        for target in self.objects_absolute_target:
            rel_target = convert_to_relative(target, self.theta_environment)
            axs[0].plot(rel_target[0], rel_target[1], 'ro')

        legend_titles = ["Sim Position", "Disp Int", "Target"]
        axs[0].legend(legend_titles)
        
        axs[1].plot(yaw)
        axs[1].plot(yaw_integrated)
        axs[1].set_ylabel('Yaw')
        axs[1].legend(["Yaw", "Yaw Disp Int"])

        fig.savefig(self.sim_dir + "/sim_pos.jpg")

        # ! timestepwise_displacement as a 2x2 subplot
        # timestepwise_displacement_data = np.array(self.timestepwise_displacement_array)
        # fig2, axs2 = plt.subplots(2, 2)
        # labels = ['X Displacement', 'Y Displacement', 'Z Displacement', 'Yaw Displacement']
        # for idx, label in enumerate(labels):
        #     axs2.flat[idx].plot(timestepwise_displacement_data[:, idx])
        #     axs2.flat[idx].set_ylabel(label)
        # fig2.savefig(self.sim_dir + "/timestepwise_displacement.jpg")

        # ! Velocity Plot
        vel_data = np.array(self.vel_array)
        fig3, axs3 = plt.subplots(2, 2)
        labels = ['X Velocity', 'Y Velocity', 'Z Velocity', 'Yaw Rate']
        for idx, label in enumerate(labels):
            axs3.flat[idx].plot(vel_data[:, idx])
            axs3.flat[idx].set_ylabel(label)
        fig3.savefig(self.sim_dir + "/sim_velocity.jpg")

        # ! Velocity Commands Plot
        if self.vel_cmds:
            vel_cmd_data = np.array(self.vel_cmds)
            fig4, axs4 = plt.subplots(2, 2)
            labels = ['X Velocity', 'Y Velocity', 'Z Velocity', 'Yaw Rate']
            for idx, label in enumerate(labels):
                axs4.flat[idx].plot(vel_cmd_data[:, idx])
                axs4.flat[idx].set_ylabel(label)
            fig4.savefig(self.sim_dir + "/vel_cmds.jpg")

        np.savetxt(os.path.join(self.sim_dir, 'sim_pos.csv'), np.array(self.global_pos_array), delimiter=',')
        np.savetxt(os.path.join(self.sim_dir, 'sim_vel.csv'), np.array(self.vel_array), delimiter=',')
        np.savetxt(os.path.join(self.sim_dir, 'timestepwise_displacement.csv'), np.array(self.timestepwise_displacement_array), delimiter=',')
        if self.vel_cmds:
            np.savetxt(os.path.join(self.sim_dir, 'vel_cmds.csv'), np.array(self.vel_cmds), delimiter=',')

