"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` or `VisionAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python fly.py

Notes
-----
The drones move, at different altitudes, along circular trajectories
in the X-Y plane, around point (0, 0).

"""
import os
import copy
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import itertools

from tqdm import trange

from scipy.signal import butter, filtfilt
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 2
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
DEFAULT_OUTPUT_FOLDER = f'train_o3_msu3_s40_fd_e100s138_216' # 'train_v11_fast_init_pp_60hz'
DEFAULT_COLAB = False
aligned_follower = True

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

def run(
        loc_color_tuple,
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        vision=DEFAULT_VISION,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        aggregate=DEFAULT_AGGREGATE,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        record_freq_hz=DEFAULT_RECORD_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB,
):
    LCR_obj_colors, colors_array, LCR_obj_xy = loc_color_tuple

    #! Simulation parameters
    #* Env Params
    H = .1
    sim_name = "save-flight-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S.%f") # include milliseconds in save name for parallel runs

    sim_dir = os.path.join(output_folder, sim_name)
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir + '/')
    for d in range(num_drones):
        if not os.path.exists(sim_dir + f"/pics{d}"):
            os.makedirs(sim_dir + f"/pics{d}/")

    num_objects = 3
    Theta = random.random() * 2 * np.pi
    Theta0 = Theta # drone faces the direction of theta

    #* Augmentation Params
    early_stop = True
    EARLY_STOP_FRAME = random.randint(73, 138)
    two_step = False
    switch = True
    force_single_switch = False
    decoupled_switch = colors_array is not None and len(colors_array) == 3
    uniform_mode = True
    alternating_mode = False

    COLORS = ['R', 'G', 'B']
    PERMUTATIONS_COLORS = [list(perm) for perm in itertools.permutations(COLORS, 3)]
    RAND_SWITCH_TIMES = []

    if colors_array is None:
        colors_array = [random.sample(PERMUTATIONS_COLORS, 1)[0]]

    #! Generate target locations (start and switched)
    #? base_obj arrays are absolutely ordered [Left, Center, Right]
    #? base_obj_colors is therefore the corresponding colors of [Left, Center, Right]
    LCR_obj_loclabels = ['L', 'C', 'R']
    if LCR_obj_xy is None:
        OBJ_START_DIST = random.uniform(2.5, 3)
        LCR_obj_xy = [(OBJ_START_DIST, lateral_obj_dist) for lateral_obj_dist in [random.uniform(0.35, 0.7), random.uniform(-0.1, 0.1), random.uniform(-0.35, -0.7)]]
    # LCR_obj_colors = [color_env[start_obj_loclabels.index(label)] for label in LCR_obj_loclabels]

    def convert_color_array_to_location_array(color_array):
        return [LCR_obj_xy[LCR_obj_colors.index(color)] for color in color_array]
    def convert_array_to_global(array, Theta=Theta):
        return [convert_to_global(rel_pos, Theta) for rel_pos in array]

    #? array indices indicate which drone [Leader, Follower1, Follower2, ...]
    start_obj_locs = convert_color_array_to_location_array(colors_array[0])
    TARGET_LOCATIONS = convert_array_to_global(start_obj_locs)
    
    # ? Targets are encoded as 0-index for original, then other indices are switched locations
    target_index = 0
    target_locations = [start_obj_locs]
    target_colors = [colors_array[0]]

    def add_random_targets(target_colors, target_locations):
        sampled_order = random.sample(PERMUTATIONS_COLORS, 1)[0]
        target_colors.append(sampled_order)
        target_locations.append(convert_color_array_to_location_array(sampled_order))
        return target_colors, target_locations
    
    def add_target(colors, target_colors, target_locations):
        target_colors.append(colors)
        target_locations.append(convert_color_array_to_location_array(colors))
        return target_colors, target_locations

    def add_target_wrapper(target_colors, target_locations):
        if alternating_mode and len(target_colors) > 2:
            return add_target(target_colors[-2], target_colors, target_locations)
        else:
            return add_random_targets(target_colors, target_locations)

    SWITCH_STOP_FRAME = 40
    if decoupled_switch:
        assert len(colors_array) == 3, f"decoupled_switch requires 3 color instructions, colors_array: {colors_array}"
        target_colors, target_locations = add_target(colors_array[1], target_colors, target_locations)
        target_colors, target_locations = add_target(colors_array[2], target_colors, target_locations)
        assert len(target_colors) == 3, "decoupled_switch requires 3 color instructions, target_colors: {target_colors}"

        RAND_SWITCH_TIMES = sorted(random.sample(list(range(10, EARLY_STOP_FRAME // 4)), 2))
    elif force_single_switch:
        target_colors, target_locations = add_target(colors_array[1], target_colors, target_locations)
        RAND_SWITCH_TIMES = [random.randint(15, SWITCH_STOP_FRAME)]
    elif uniform_mode:
        num_switches = random.choices([1, 2, 3], k=1)
        for i in range(num_switches[0]):
            target_colors, target_locations = add_target_wrapper(target_colors, target_locations)
        RAND_SWITCH_TIMES = sorted(random.sample(range(10, SWITCH_STOP_FRAME), num_switches[0]))
    else:
        for i in range(10, SWITCH_STOP_FRAME):
            if random.random() < 0.05:
                target_colors, target_locations = add_target_wrapper(target_colors, target_locations)
                RAND_SWITCH_TIMES.append(i)
    print("RAND_SWITCH_TIMES: ", RAND_SWITCH_TIMES)

    
    CUSTOM_OBJECT_LOCATION = {
        "colors": LCR_obj_colors,
        "locations": convert_array_to_global(LCR_obj_xy)
    }

    #* Save starting env params
    # with open(os.path.join(sim_dir, 'locs.txt'), 'w') as f:
    #     f.write(str("".join(start_obj_loclabels)))
    with open(os.path.join(sim_dir, 'colors.txt'), 'w') as f:
        f.write(str("".join(colors_array[0])))
    
    # Starting drone locations
    rel_drone_locs = [(0.5 * (i - (num_drones - 1)), 0) for i in range(num_drones)]

    FINAL_THETA = [angle_between_two_points(rel_drone, rel_obj) for rel_drone, rel_obj in zip(rel_drone_locs, start_obj_locs)]
    INIT_XYZS = np.array([[*convert_to_global(rel_pos, Theta), H] for rel_pos in rel_drone_locs])
    INIT_RPYS = np.array([[0, 0, Theta0] for d in range(num_drones)])
    AGGR_PHY_STEPS = int(simulation_freq_hz / control_freq_hz) if aggregate else 1

    STOPPING_START = 1.0; STOPPING_END = 0.5; END_HOLD_TIME = 0.5 # Hold stop for 0.5 seconds
    DEFAULT_SPEED = 0.15 # TILES / S
    DEFAULT_BACKSPEED = 0.03 # TILES / S
    MIN_SPEED = 0.01
    ANGLE_RECOVERY_TIMESTEPS = control_freq_hz * 5 #random.uniform(2, 10)
    DEFAULT_SEARCHING_YAW = 0.05
    SLOW_YAW_THRESHOLD = 0.05
    MIN_SEARCHING_YAW = 0.01
    APPROX_CORRECT_YAW = 0.0001

    LABELS = []
    UNFILTERED_LABELS = [colors_array[0]]

    TARGET_POS = [[arry] for arry in INIT_XYZS]
    TARGET_ATT = [[arry] for arry in INIT_RPYS]
    INIT_THETA = [init_rpys[2] for init_rpys in INIT_RPYS]
    # angular distance between init and final theta
    DELTA_THETA = [signed_angular_distance(init_theta, final_theta + Theta) for final_theta, init_theta in zip(FINAL_THETA, INIT_RPYS[:, 2])]

    def step_with_distance_and_angle(counter, INIT_THETA, FINAL_THETA, DELTA_THETA, TARGET_LOCATIONS, target_index, hold=False):
        speeds = []
        UNFILTERED_LABELS.append(target_colors[target_index][:num_drones])
        for target_pos, target_att, init_theta, final_theta, delta_theta, final_target in zip(TARGET_POS, TARGET_ATT, INIT_THETA, FINAL_THETA, DELTA_THETA, TARGET_LOCATIONS):
            last_pos = target_pos[-1] # X, Y, Z
            last_yaw = target_att[-1][2] # R, P, Y
            dist = np.sqrt((last_pos[0] - final_target[0]) ** 2 + (last_pos[1] - final_target[1]) ** 2)
            yaw_dist = signed_angular_distance(last_yaw, final_theta + Theta)
            # print(f"yaw_dist: {yaw_dist}, delta_theta: {delta_theta}")

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
                yaw_speed = DEFAULT_SEARCHING_YAW * (yaw_dist / SLOW_YAW_THRESHOLD) / control_freq_hz
            else:
                yaw_speed = 0

            delta_pos = convert_to_global([speed, 0], final_theta + Theta)

            if abs(yaw_dist) < APPROX_CORRECT_YAW or np.sign(yaw_dist) != np.sign(delta_theta):
                adj_theta = final_theta + Theta - init_theta
            else:
                adj_theta = last_yaw + yaw_speed - init_theta
            
            if two_step and counter < ANGLE_RECOVERY_TIMESTEPS:
                target_pos.append([last_pos[0], last_pos[1], H])
            else:
                target_pos.append([last_pos[0] + delta_pos[0], last_pos[1] + delta_pos[1], H])
            target_att.append([0, 0, init_theta + adj_theta])

            speeds.append(speed)
        return speeds
    
    logging_targets = []
    logging_final_theta = []
    logging_init_theta = []
    logging_delta_theta = []
    # ! Precompute Trajectories
    # usually, computed at 240 hz
    angle_counter = 0 # only updated after Starting Hold

    # * Hold for stabilization
    # for i in range(int(1 * control_freq_hz)):
    #     step_with_distance_and_angle(angle_counter, INIT_THETA, FINAL_THETA, DELTA_THETA, TARGET_LOCATIONS, alternate_index, hold=True)

    # * Main trajectory is at 240 Hz
    speeds = [DEFAULT_SPEED / control_freq_hz for i in range(num_drones)]
    while np.abs(np.array(speeds)).max() > 1e-6:
        speeds = step_with_distance_and_angle(angle_counter, INIT_THETA, FINAL_THETA, DELTA_THETA, TARGET_LOCATIONS, target_index)
        angle_counter += 1

        if angle_counter % 30 == 0:
            logging_targets.append([y for (x, y) in target_locations[target_index]])
            logging_final_theta.append(FINAL_THETA)
            logging_init_theta.append(INIT_THETA)
            logging_delta_theta.append(DELTA_THETA)

        if switch and (np.array(angle_counter) == np.array(RAND_SWITCH_TIMES) * 30).any():
            target_index += 1
            print(f"switching: {angle_counter} {target_index}")
            cur_drone_pos = [convert_to_relative((TARGET_POS[d][-1][0], TARGET_POS[d][-1][1]), Theta) for d in range(num_drones)]

            for d in range(num_drones):
                print(f"cur_drone_pos[d]: {cur_drone_pos[d]}")
                print(f"target_locations[target_index][d]: {target_locations[target_index][d]}")
            print(FINAL_THETA)
            FINAL_THETA = [angle_between_two_points(cur_drone_pos[d], target_locations[target_index][d]) for d in range(num_drones)]
            TARGET_LOCATIONS = convert_array_to_global(target_locations[target_index])
            INIT_THETA = [target_att[-1][2] for target_att in TARGET_ATT]
            DELTA_THETA = [signed_angular_distance(init_theta, final_theta + Theta) for final_theta, init_theta in zip(FINAL_THETA, INIT_THETA)]


        if early_stop and len(TARGET_POS[0]) > EARLY_STOP_FRAME * 30: # 73
            break

    #* Hold for learning to stop
    if not early_stop:
        for i in range(int(END_HOLD_TIME * control_freq_hz)):
            step_with_distance_and_angle(angle_counter, INIT_THETA, FINAL_THETA, DELTA_THETA, TARGET_LOCATIONS, target_index)
            angle_counter += 1

    assert target_index == len(target_locations) - 1, f"target_index: {target_index}, len(target_locations): {len(target_locations)}"

    wp_counters = np.array([0 for i in range(num_drones)])

    #! Simulation Setup
    if vision:
        env = VisionAviary(drone_model=drone,
                           num_drones=num_drones,
                           initial_xyzs=INIT_XYZS,
                           initial_rpys=INIT_RPYS,
                           physics=physics,
                           neighbourhood_radius=10,
                           freq=simulation_freq_hz,
                           aggregate_phy_steps=AGGR_PHY_STEPS,
                           gui=gui,
                           record=record_video,
                           obstacles=obstacles
                           )
    else:
        env = CtrlAviary(drone_model=drone,
                         num_drones=num_drones,
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPYS,
                         physics=physics,
                         neighbourhood_radius=10,
                         freq=simulation_freq_hz,
                         aggregate_phy_steps=AGGR_PHY_STEPS,
                         gui=gui,
                         record=record_video,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         custom_obj_location=CUSTOM_OBJECT_LOCATION
                         )
    env.IMG_RES = np.array([256, 144])

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=4,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]
    elif drone in [DroneModel.HB]:
        ctrl = [SimplePIDControl(drone_model=drone) for i in range(num_drones)]

    #! Simulation Params
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ / control_freq_hz)) # 1
    REC_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ / record_freq_hz )) #30 #240
    action = {str(i): np.array([0, 0, 0, 0]) for i in range(num_drones)}
    START = time.time()
    ACTIVE_WP = len(TARGET_POS[0])
    TARGET_POS = np.array(TARGET_POS)
    TARGET_ATT = np.array(TARGET_ATT)
    STEPS = int(CTRL_EVERY_N_STEPS * ACTIVE_WP)

    env.reset()

    #! Stepping Simulation
    # main loop at 240hz
    actions = []
    for i in trange(0, int(STEPS), AGGR_PHY_STEPS):
        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)

        #* Compute step-by-step velocities for 240hz-trajectory; Control Frequency is 240hz
        if i % CTRL_EVERY_N_STEPS == 0:
            for j in range(num_drones):
                state = obs[str(j)]["state"]
                action[str(j)], _, _ = ctrl[j].computeControlFromState(
                    control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                    state=state,
                    target_pos = TARGET_POS[j, wp_counters[j]],
                    target_rpy = TARGET_ATT[j, wp_counters[j]]
                    )

            #### Go to the next way point and loop #####################
            for j in range(num_drones):
                wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (ACTIVE_WP - 1) else wp_counters[j]

        #* Network Frequency is 30hz
        if i % REC_EVERY_N_STEPS == 0 and i>env.SIM_FREQ:
            for d in range(num_drones):
                rgb, dep, seg = env._getDroneImages(d)
                env._exportImage(img_type=ImageType.RGB,
                                img_input=rgb,
                                path=sim_dir + f"/pics{d}",
                                frame_num=int(i / REC_EVERY_N_STEPS),
                                )
                actions.append(action[str(d)])
                ### Log the simulation ####################################
                logger.log(drone=d,
                           timestamp=int(i / REC_EVERY_N_STEPS),
                           state=obs[str(d)]["state"],
                           control=np.hstack(
                               [TARGET_POS[d, wp_counters[d], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
                           )
                
            LABELS.append(UNFILTERED_LABELS[wp_counters[0]])

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.TIMESTEP)

    with open(sim_dir + "/values.csv", 'wb') as out_file:
        np.savetxt(out_file, LABELS, delimiter=",", fmt='%s')
    processed_switch_times = [t - 8 for t in RAND_SWITCH_TIMES]
    with open(sim_dir + "/switchpoints.csv", 'wb') as out_file:
        np.savetxt(out_file, processed_switch_times, delimiter=",", fmt='%s')
    # with open(sim_dir + "/targets.csv", 'wb') as out_file:
    #     np.savetxt(out_file, logging_targets[8:], delimiter=",", fmt='%f,%f,%f')
    # with open(sim_dir + "/final_theta.csv", 'wb') as out_file:
    #     np.savetxt(out_file, logging_final_theta[8:], delimiter=",", fmt='%f')
    # with open(sim_dir + "/init_theta.csv", 'wb') as out_file:
    #     np.savetxt(out_file, logging_init_theta[8:], delimiter=",", fmt='%f')
    # with open(sim_dir + "/delta_theta.csv", 'wb') as out_file:
    #     np.savetxt(out_file, logging_delta_theta[8:], delimiter=",", fmt='%f')
    
    env.close()
    logger.save_as_csv(sim_name)  # Optional CSV save

    debug_plots = True
    if debug_plots:
        # read log_0.csv and plot radius
        data = np.genfromtxt(sim_dir + "/log_0.csv", delimiter=",", skip_header=2)
        fig, ax = plt.subplots()
        # plot with transparency
        ax.plot(data[:, 1], data[:, 2], alpha=0.9)
        if num_drones > 1:
            data2 = np.genfromtxt(sim_dir + "/log_1.csv", delimiter=",", skip_header=2)
            ax.plot(data2[:, 1], data2[:, 2], alpha=0.9)
        ax.plot(TARGET_POS[0, :, 0], TARGET_POS[0, :, 1], alpha=0.9)
        for target in TARGET_LOCATIONS:
            ax.plot(target[0], target[1], 'ro')
        ax.legend(["Leader", "Follower", "Lead Target", "Obj"])
        
        # ax.set_aspect('equal', 'box')
        fig.savefig(sim_dir + "/path.jpg")

        fig, ax = plt.subplots()
        radius = np.sqrt(data[:, 1] **2 + data[:, 2] **2)
        ax.plot(data[:, 0], radius)
        # radius2 = np.sqrt(data2[:, 1] **2 + data2[:, 2] **2)
        # ax.plot(data2[:, 0], radius2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Radius (arb. units)")
        fig.savefig(sim_dir + "/radius.jpg")

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone', default=DEFAULT_DRONES, type=DroneModel, help='Drone model (default: CF2X)',
                        metavar='', choices=DroneModel)
    parser.add_argument('--num_drones', default=DEFAULT_NUM_DRONES, type=int, help='Number of drones (default: 3)',
                        metavar='')
    parser.add_argument('--physics', default=DEFAULT_PHYSICS, type=Physics, help='Physics updates (default: PYB)',
                        metavar='', choices=Physics)
    parser.add_argument('--vision', default=DEFAULT_VISION, type=str2bool,
                        help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Whether to use PyBullet GUI (default: True)',
                        metavar='')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VISION, type=str2bool,
                        help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot', default=DEFAULT_PLOT, type=str2bool,
                        help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui', default=DEFAULT_USER_DEBUG_GUI, type=str2bool,
                        help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate', default=DEFAULT_AGGREGATE, type=str2bool,
                        help='Whether to aggregate physics steps (default: True)', metavar='')
    parser.add_argument('--obstacles', default=DEFAULT_OBSTACLES, type=str2bool,
                        help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int,
                        help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz', default=DEFAULT_CONTROL_FREQ_HZ, type=int,
                        help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec', default=DEFAULT_DURATION_SEC, type=int,
                        help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
                        help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool,
                        help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    # LOCATIONS = ['L', 'C', 'R']
    # PERMUTATIONS_LOCATIONS = [list(perm) for perm in itertools.permutations(LOCATIONS, 3)]
    # sampled_locations = random.sample(PERMUTATIONS_LOCATIONS, 1)[0]
    
    def is_double_switch(first, second):
        return first[0] != second[0] and first[1] != second[1]
    
    def get_decoupled_paths(first, second):
        intermediate1 = first[0] + second[1] # switch follower first
        intermediate2 = second[0] + first[1] # switch leader first
        return list(intermediate1), list(intermediate2) # returns in form ['R', 'G'] and ['G', 'R']

    # get all 2 color permutations, without replacement e.g. ['RG', 'RB', 'GR', 'GB', 'BR', 'BG']
    COLORS = ['R', 'G', 'B']
    PERMUTATIONS_COLORS = [list(perm) for perm in itertools.permutations(COLORS, 3)]

    num_initializations = 1
    LCR_env_colors = PERMUTATIONS_COLORS * num_initializations
    # LCR_env_colors = random.sample(PERMUTATIONS_COLORS, num_initializations)
    total_list = []
    for i, LCR_env_color in enumerate(LCR_env_colors):
        OBJ_START_DIST = random.uniform(2.5, 3)
        # base_obj_xy = [(OBJ_START_DIST, lateral_obj_dist) for lateral_obj_dist in [random.uniform(0.35, 0.5), random.uniform(-0.1, 0.1), random.uniform(-0.35, -0.5)]]
        base_obj_xy = None

        fragment = [(LCR_env_color, None, base_obj_xy)]
        total_list.extend(fragment)
        # fragment = [(LCR_env_color, (color1, color2), base_obj_xy) for color1 in PERMUTATIONS_COLORS for color2 in PERMUTATIONS_COLORS if color1 != color2]
        # decoupled_frament = []
        # for (LCR_env_color, (color1, color2), base_obj_xy) in fragment:
        #     if is_double_switch(color1, color2):
        #         intermediate1, intermediate2 = get_decoupled_paths(color1, color2)
        #         decoupled_frament.append((LCR_env_color, (color1, intermediate1, color2), base_obj_xy))
        #         decoupled_frament.append((LCR_env_color, (color1, intermediate2, color2), base_obj_xy))
        #     else:
        #         decoupled_frament.append((LCR_env_color, (color1, color2), base_obj_xy))
        # total_list.extend(decoupled_frament)

    total_list = total_list * 36
    assert len(total_list) == num_initializations * 6 * 36, f"len(total_list): {len(total_list)}"
    random.shuffle(total_list)

    from tqdm import tqdm
    from functools import partial
    run_func = partial(run, **vars(ARGS))

    futures = []
    returns = []

    import joblib
    joblib.Parallel(n_jobs=16)(joblib.delayed(run_func)(d) for d in tqdm(total_list))
