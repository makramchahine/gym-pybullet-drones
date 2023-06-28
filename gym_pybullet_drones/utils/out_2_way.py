# write a function that takes a 4 dimensional vector corresponding to the drone body frame velocities and yaw rate
# as well as its position in the world frame and a time step and returns the next state of the drone in the world frame
import numpy as np


def out2way(output, state, dt):
    """
    :param output: 4 dimensional vector corresponding to the drone body frame velocities and yaw rate
    :param state: 20 dimensional vector corresponding to the drone state in the world frame
    :param dt: time step
    :return: the next state of the drone in the world frame
    """

    # # convert body frame velocities to world frame velocities
    vx_w = output[0] * np.cos(state[9]) - output[1] * np.sin(state[9])
    vy_w = output[0] * np.sin(state[9]) + output[1] * np.cos(state[9])
    # vx_w = output[0] * np.cos(-state[9]) + output[1] * np.sin(-state[9])
    # vy_w = -output[0] * np.sin(-state[9]) + output[1] * np.cos(-state[9])
    #
    # vx_w = output[0]
    # vy_w = output[1]

    # integrate to get the next position
    x = state[0] + vx_w * dt
    y = state[1] + vy_w * dt
    z = state[2] + output[2] * dt
    yaw = state[9] + output[3] * dt

    # return next position and attitude
    return np.array([x, y, z, state[7], state[8], yaw, vx_w, vy_w, output[2], state[17], state[18], output[3]])
