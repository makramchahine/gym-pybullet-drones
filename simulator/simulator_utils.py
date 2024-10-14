import numpy as np

from simulator.culekta_utils import convert_to_relative, signed_angular_distance

def convert_vel_cmd_to_world_frame(vel_cmd, yaw):
    # convert from body_frame to world_frame
    vel_cmd_world = vel_cmd.copy()
    vel_cmd_world[0] = vel_cmd[0] * np.cos(-yaw) + vel_cmd[1] * np.sin(-yaw)
    vel_cmd_world[1] = -vel_cmd[0] * np.sin(-yaw)+ vel_cmd[1] * np.cos(-yaw)
    return vel_cmd_world

def get_x_y_z_yaw_relative_to_base_env(state, Theta):
    # Convert to relative drone coordinates by factoring in Theta
    relative_state = convert_to_relative([state[0], state[1]], Theta)
    return np.array([relative_state[0], relative_state[1], state[2], state[9]])

def get_x_y_z_yaw_global(state):
    return np.array([state[0], state[1], state[2], state[9]])

def get_relative_displacement(later_state, earlier_state, environment_theta):
    relative_xy = convert_to_relative([later_state[0] - earlier_state[0], later_state[1] - earlier_state[1]], earlier_state[3] + environment_theta)
    return np.array([relative_xy[0], relative_xy[1], later_state[2] - earlier_state[2], signed_angular_distance(earlier_state[3], later_state[3])])

def get_x_y_z_yaw_rel_to_self(state, previous_yaw):
    # Convert to relative drone coordinates by factoring in self.Theta
    relative_state = convert_to_relative([state[0], state[1]], previous_yaw)
    return np.array([relative_state[0], relative_state[1], state[2], state[9]])

def get_vx_vy_vz_yawrate_rel_to_self(state):
    global_vx = state[10]
    global_vy = state[11]
    yaw = state[9]

    # convert from world_frame to body_frame
    rel_vx = global_vx * np.cos(yaw) + global_vy * np.sin(yaw)
    rel_vy = -global_vx * np.sin(yaw) + global_vy * np.cos(yaw)

    return np.array([rel_vx, rel_vy, state[12], state[15]])