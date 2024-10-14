import os
import numpy as np
import matplotlib.pyplot as plt

from simulator.simulator_utils import convert_to_relative

def export_plots(global_pos_array, timestepwise_displacement_array, vel_array, vel_cmds, objects_absolute_target, theta_environment, sim_dir):
    x_data = np.array(global_pos_array)[:, 0]
    y_data = np.array(global_pos_array)[:, 1]
    yaw = np.array(global_pos_array)[:, 3]

    x_data_integrated = np.cumsum(np.array(timestepwise_displacement_array)[:, 0])
    y_data_integrated = np.cumsum(np.array(timestepwise_displacement_array)[:, 1])
    yaw_integrated = np.cumsum(np.array(timestepwise_displacement_array)[:, 3])

    # ! Path Plot
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(x_data, y_data, alpha=0.5)
    axs[0].plot(x_data_integrated, y_data_integrated, alpha=0.5)
    axs[0].set_aspect('equal', adjustable='box')
    
    for target in objects_absolute_target:
        rel_target = convert_to_relative(target, theta_environment)
        axs[0].plot(rel_target[0], rel_target[1], 'ro')

    legend_titles = ["Sim Position", "Disp Int", "Target"]
    axs[0].legend(legend_titles)
    
    axs[1].plot(yaw)
    axs[1].plot(yaw_integrated)
    axs[1].set_ylabel('Yaw')
    axs[1].legend(["Yaw", "Yaw Disp Int"])

    fig.savefig(sim_dir + "/sim_pos.jpg")

    # ! timestepwise_displacement as a 2x2 subplot
    timestepwise_displacement_data = np.array(timestepwise_displacement_array)
    fig2, axs2 = plt.subplots(2, 2)
    labels = ['X Displacement', 'Y Displacement', 'Z Displacement', 'Yaw Displacement']
    for idx, label in enumerate(labels):
        axs2.flat[idx].plot(timestepwise_displacement_data[:, idx])
        axs2.flat[idx].set_ylabel(label)
    fig2.savefig(sim_dir + "/timestepwise_displacement.jpg")

    # ! Velocity Plot
    vel_data = np.array(vel_array)
    fig3, axs3 = plt.subplots(2, 2)
    labels = ['X Velocity', 'Y Velocity', 'Z Velocity', 'Yaw Rate']
    for idx, label in enumerate(labels):
        axs3.flat[idx].plot(vel_data[:, idx])
        axs3.flat[idx].set_ylabel(label)
    fig3.savefig(sim_dir + "/sim_velocity.jpg")

    # ! Velocity Commands Plot
    if vel_cmds:
        vel_cmd_data = np.array(vel_cmds)
        fig4, axs4 = plt.subplots(2, 2)
        labels = ['X Velocity', 'Y Velocity', 'Z Velocity', 'Yaw Rate']
        for idx, label in enumerate(labels):
            axs4.flat[idx].plot(vel_cmd_data[:, idx])
            axs4.flat[idx].set_ylabel(label)
        fig4.savefig(sim_dir + "/vel_cmds.jpg")

    np.savetxt(os.path.join(sim_dir, 'sim_pos.csv'), np.array(global_pos_array), delimiter=',')
    np.savetxt(os.path.join(sim_dir, 'sim_vel.csv'), np.array(vel_array), delimiter=',')
    np.savetxt(os.path.join(sim_dir, 'timestepwise_displacement.csv'), np.array(timestepwise_displacement_array), delimiter=',')
    if vel_cmds:
        np.savetxt(os.path.join(sim_dir, 'vel_cmds.csv'), np.array(vel_cmds), delimiter=',')
