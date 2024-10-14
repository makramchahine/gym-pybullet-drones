import os
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import itertools

DEFAULT_CONTROL_FREQ_HZ = 240
STOPPING_START = 1.0; STOPPING_END = 0.5; END_HOLD_TIME = 0.5 # Hold stop for 0.5 seconds
DEFAULT_SPEED = 0.1 / 2 # TILES / S
DEFAULT_BACKSPEED = 0.03 # TILES / S
MIN_SPEED = 0.01
ANGLE_RECOVERY_TIMESTEPS = DEFAULT_CONTROL_FREQ_HZ * 5 #random.uniform(2, 10)
DEFAULT_SEARCHING_YAW = 0.05
SLOW_YAW_THRESHOLD = 0.05
MIN_SEARCHING_YAW = 0.01
APPROX_CORRECT_YAW = 0.0001
APPROX_CORRECT_HEIGHT = 0.001
STABILIZE_LIFT_MULTIPLIER = 0.25

IN_VIEW_THRESHOLD_RADIAN = 0.2 * np.pi
IN_RANGE_THRESHOLD = 0.7

drop = True
DEFAULT_CRITICAL_SPEED = 0.05 / 1.5  
DEFAULT_CRITICAL_YAW_SPEED = 0.2 * np.pi * 2/3
DEFAULT_SEARCHING_YAW = 0.10 * np.pi * 2/3
DEFAULT_LIFT_SPEED = 0.05
DEFAULT_DROP_POINT_DIST = 0.05
DROP_MAX_HEIGHT = 0.3
DROP_SPEED = 0.1

COLORS = ['R', 'G', 'B']
PERMUTATIONS_COLORS = [list(perm) for perm in itertools.permutations(COLORS, 3)]


def signed_angular_distance(theta1, theta2):
    # source to target
    # If the result is positive, it means you would need to rotate counter-clockwise from theta1 to theta2. If the result is negative, it means you would need to rotate clockwise.
    return ((theta2 - theta1 + np.pi) % (2 * np.pi)) - np.pi

def convert_to_global(rel_pos, theta):
    return (np.cos(theta) * rel_pos[0] - np.sin(theta) * rel_pos[1], np.sin(theta) * rel_pos[0] + np.cos(theta) * rel_pos[1])

def convert_to_relative(global_pos, theta):
    return (np.cos(theta) * global_pos[0] + np.sin(theta) * global_pos[1], -np.sin(theta) * global_pos[0] + np.cos(theta) * global_pos[1])

def angle_between_two_points(source_coord, target_coord):
    return np.arctan2(target_coord[1] - source_coord[1], target_coord[0] - source_coord[0])

def convert_color_array_to_location_array(LCR_obj_colors, LCR_obj_xy, color_array):
    return [LCR_obj_xy[LCR_obj_colors.index(color)] for color in color_array]

def convert_array_to_global(array, Theta):
    return [convert_to_global(rel_pos, Theta) for rel_pos in array]

def setup_folders(sim_dir, num_drones):
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir + '/')
    for d in range(num_drones):
        if not os.path.exists(sim_dir + f"/pics{d}"):
            os.makedirs(sim_dir + f"/pics{d}/")
        if not os.path.exists(sim_dir + f"/pybullet_pics{d}"):
            os.makedirs(sim_dir + f"/pybullet_pics{d}/")

def add_random_targets(target_colors, target_locations, LCR_obj_xy, LCR_obj_colors):
    sampled_order = random.sample(PERMUTATIONS_COLORS, 1)[0]
    return add_target(sampled_order, target_colors, target_locations, LCR_obj_xy, LCR_obj_colors)

def add_target(colors, target_colors, target_locations, LCR_obj_xy, LCR_obj_colors):
    target_colors.append(colors)
    target_locations.append(convert_color_array_to_location_array(colors, LCR_obj_xy, LCR_obj_colors))
    return target_colors, target_locations

def step_with_distance_and_angle(INIT_THETA, FINAL_THETA, DELTA_THETA, TARGET_LOCATIONS, target_index, UNFILTERED_LABELS, TARGET_POS, TARGET_ATT, Theta, control_freq_hz, num_drones, target_colors, HS, hold=False):
    speeds = []
    aligned = [False, False]
    UNFILTERED_LABELS.append(target_colors[target_index][:num_drones])
    for i, (target_pos, target_att, init_theta, final_theta, delta_theta, final_target, height) in enumerate(zip(TARGET_POS, TARGET_ATT, INIT_THETA, FINAL_THETA, DELTA_THETA, TARGET_LOCATIONS, HS)):
        last_pos = target_pos[-1] # X, Y, Z
        last_yaw = target_att[-1][2] # R, P, Y
        dist = np.sqrt((last_pos[0] - final_target[0]) ** 2 + (last_pos[1] - final_target[1]) ** 2)
        yaw_dist = signed_angular_distance(last_yaw, final_theta + Theta)

        if hold:
            speed = 0
        elif dist > STOPPING_START:
            speed = DEFAULT_SPEED / control_freq_hz
        elif dist > STOPPING_END:
            speed = DEFAULT_SPEED * (dist - STOPPING_END) / (STOPPING_START - STOPPING_END)
            speed = np.max([speed, MIN_SPEED]) / control_freq_hz
        elif dist < STOPPING_END - 0.05:
            speed = DEFAULT_BACKSPEED * (STOPPING_END - dist) / STOPPING_END
            speed = -np.max([speed, MIN_SPEED]) / control_freq_hz
        else:
            speed = 0

        if hold:
            yaw_speed = 0
        elif abs(yaw_dist) > SLOW_YAW_THRESHOLD:
            yaw_speed = DEFAULT_SEARCHING_YAW * np.sign(yaw_dist) / control_freq_hz
        elif abs(yaw_dist) >= APPROX_CORRECT_YAW:
            if random.random() < 0.1:
                aligned = [True, True]
            yaw_speed = DEFAULT_SEARCHING_YAW * (yaw_dist / SLOW_YAW_THRESHOLD) / control_freq_hz
        else:
            aligned[i] = True
            yaw_speed = 0

        if abs(yaw_dist) < APPROX_CORRECT_YAW or np.sign(yaw_dist) != np.sign(delta_theta):
            adj_theta = final_theta + Theta - init_theta
        else:
            adj_theta = last_yaw + yaw_speed - init_theta
        
        target_pos.append([0, 0, height])
        target_att.append([0, 0, init_theta + adj_theta])

        speeds.append(speed)
    return speeds, aligned, UNFILTERED_LABELS, TARGET_POS, TARGET_ATT

def getting_farther_away(target_pos, final_target):
    if len(target_pos) < 2:
        return False
    last_pos = target_pos[-1]
    second_to_last_pos = target_pos[-2]
    return np.sqrt((last_pos[0] - final_target[0]) ** 2 + (last_pos[1] - final_target[1]) ** 2) > np.sqrt((second_to_last_pos[0] - final_target[0]) ** 2 + (second_to_last_pos[1] - final_target[1]) ** 2)

def interpolate_speeds(dist, critical_dist, critical_dist_buffer, speed1, speed2):
    # as dist gets closer to critical_dist, speed gets closer to speed1
    return speed1 + (speed2 - speed1) * (dist - critical_dist) / critical_dist_buffer

def distance_to_target(obj_xy, target_xy):
    return np.sqrt((obj_xy[0] - target_xy[0]) ** 2 + (obj_xy[1] - target_xy[1]) ** 2)

def step_hike(critical_dist, critical_dist_buffer, critical_action: str, INIT_THETA, FINAL_THETA, DELTA_THETA, TARGET_LOCATIONS, target_index, TARGET_POS, TARGET_ATT, Theta, control_freq_hz, HS, previously_hit_critical=False, hold=False):
    speeds = []
    for i, (target_pos, target_att, init_theta, final_theta, delta_theta, final_target, height) in enumerate(zip(TARGET_POS, TARGET_ATT, INIT_THETA, FINAL_THETA, DELTA_THETA, TARGET_LOCATIONS, HS)):
        last_pos = target_pos[-1]    # X, Y, Z
        last_yaw = target_att[-1][2] # R, P, Y
        last_height = target_pos[-1][2]
        dist = distance_to_target(last_pos, final_target)
        yaw_dist = signed_angular_distance(last_yaw, final_theta + Theta)
        height_dist = height - last_height
        # print(f"yaw_dist: {yaw_dist}, delta_theta: {delta_theta}")

        lift_speed = 0
        if hold:
            speed = 0
            yaw_speed = 0
            new_theta = init_theta
            lift_speed = 0
        elif dist > critical_dist + critical_dist_buffer and not previously_hit_critical:
            speed = DEFAULT_SPEED / control_freq_hz
            yaw_speed = DEFAULT_SEARCHING_YAW * np.sign(yaw_dist) / control_freq_hz
            lift_speed = 0 if (abs(height_dist) < APPROX_CORRECT_HEIGHT) else STABILIZE_LIFT_MULTIPLIER * (height_dist / height) / control_freq_hz

            if abs(yaw_dist) < APPROX_CORRECT_YAW:
                new_theta = final_theta + Theta
            else:
                new_theta = last_yaw + yaw_speed
            FINAL_THETA[0] = angle_between_two_points(last_pos[:2], final_target[:2]) - Theta
        elif dist > critical_dist and not previously_hit_critical:
            speed = interpolate_speeds(dist, critical_dist, critical_dist_buffer, DEFAULT_CRITICAL_SPEED, DEFAULT_SPEED) / control_freq_hz
            yaw_speed = interpolate_speeds(dist, critical_dist, critical_dist_buffer, DEFAULT_CRITICAL_YAW_SPEED, DEFAULT_SEARCHING_YAW) * np.sign(yaw_dist) / control_freq_hz
            lift_speed = 0 if (abs(height_dist) < APPROX_CORRECT_HEIGHT) else STABILIZE_LIFT_MULTIPLIER * (height_dist / height) / control_freq_hz

            if abs(yaw_dist) < APPROX_CORRECT_YAW:
                new_theta = final_theta + Theta
            else:
                new_theta = last_yaw + yaw_speed
        else:
            speed = DEFAULT_CRITICAL_SPEED / control_freq_hz
            yaw_speed = DEFAULT_SEARCHING_YAW * np.sign(yaw_dist) / control_freq_hz
            if critical_action == 'R':
                yaw_speed += DEFAULT_CRITICAL_YAW_SPEED / control_freq_hz
            elif critical_action == 'B':
                yaw_speed += -DEFAULT_CRITICAL_YAW_SPEED / control_freq_hz
            elif critical_action == 'G':
                lift_speed = DEFAULT_LIFT_SPEED / control_freq_hz
                if not previously_hit_critical:
                    FINAL_THETA[0] = angle_between_two_points(last_pos[:2], final_target[:2]) - Theta # continue to face target
            else:
                assert False, f"critical_action: {critical_action}"
            new_theta = last_yaw + yaw_speed


        if critical_action == 'G':
            delta_z = lift_speed if not previously_hit_critical else -DROP_SPEED / control_freq_hz * (last_height / DROP_MAX_HEIGHT)
            reached_critical = dist < DEFAULT_DROP_POINT_DIST
            new_height = max(last_height + delta_z, height)
        else:
            delta_z = lift_speed
            reached_critical = dist < critical_dist
            new_height = (last_height + delta_z) if (lift_speed != 0 or hold) else height
    
        delta_pos = convert_to_global([speed, 0], new_theta)
        target_pos.append([last_pos[0] + delta_pos[0], last_pos[1] + delta_pos[1], new_height])
        target_att.append([0, 0, new_theta])


        speeds.append(speed)
    return reached_critical, speeds, TARGET_POS, TARGET_ATT, FINAL_THETA

def generate_debug_plots(sim_dir, TARGET_POS, TARGET_LOCATIONS, num_drones):
    # read log_0.csv and plot radius
    data = np.genfromtxt(sim_dir + "/log_0.csv", delimiter=",", skip_header=2)
    fig, ax = plt.subplots()
    # plot with transparency
    ax.plot(data[:, 1], data[:, 2], alpha=0.9)
    # draw arrow with yaw every 10 steps
    if num_drones > 1:
        data2 = np.genfromtxt(sim_dir + "/log_1.csv", delimiter=",", skip_header=2)
        ax.plot(data2[:, 1], data2[:, 2], alpha=0.9)
    ax.plot(TARGET_POS[0, :, 0], TARGET_POS[0, :, 1], alpha=0.9)
    for target in TARGET_LOCATIONS:
        print(f"target: {target}")
        ax.plot(target[0], target[1], 'ro')
    ax.legend(["Leader", "Leader Target", "Obj"])
    for i in range(0, len(data), 10):
        ax.arrow(data[i, 1], data[i, 2], 0.1 * np.cos(data[i, 9]), 0.1 * np.sin(data[i, 9]), head_width=0.005, head_length=0.01, fc='k', ec='k')
    
    # ax.set_aspect('equal', 'box')
    fig.savefig(sim_dir + "/path.jpg")

def is_double_switch(first, second):
    return first[0] != second[0] and first[1] != second[1]

def get_decoupled_paths(first, second):
    intermediate1 = first[0] + second[1] # switch follower first
    intermediate2 = second[0] + first[1] # switch leader first
    return list(intermediate1), list(intermediate2) # returns in form ['R', 'G'] and ['G', 'R']

def decouple_frament(fragment):
    decoupled_frament = []
    for (LCR_env_color, (color1, color2), base_obj_xy) in fragment:
        if is_double_switch(color1, color2):
            intermediate1, intermediate2 = get_decoupled_paths(color1, color2)
            decoupled_frament.append((LCR_env_color, (color1, intermediate1, color2), base_obj_xy))
            decoupled_frament.append((LCR_env_color, (color1, intermediate2, color2), base_obj_xy))
        else:
            decoupled_frament.append((LCR_env_color, (color1, color2), base_obj_xy))

    return decoupled_frament

def get_relative_angle_to_target(x, y, yaw, xy_target):
    target_x, target_y = xy_target[0], xy_target[1]
    angle_to_target = np.arctan2(target_y - y, target_x - x)
    relative_angle = (angle_to_target - yaw) % (2 * np.pi)
    if relative_angle > np.pi:
        relative_angle -= 2 * np.pi
    return relative_angle

def drone_turned_left(x, y, yaw, xy_target):
    relative_angle = get_relative_angle_to_target(x, y, yaw, xy_target)
    return relative_angle < 0

def drone_turned_right(x, y, yaw, xy_target):
    relative_angle = get_relative_angle_to_target(x, y, yaw, xy_target)
    return relative_angle > 0

def object_in_view(x, y, yaw, xy_target):
    relative_angle = get_relative_angle_to_target(x, y, yaw, xy_target)
    return (-IN_VIEW_THRESHOLD_RADIAN <= relative_angle <= IN_VIEW_THRESHOLD_RADIAN)

def object_in_range(x, y, xy_target):
    return (distance_to_target((x, y), xy_target) < IN_RANGE_THRESHOLD)

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