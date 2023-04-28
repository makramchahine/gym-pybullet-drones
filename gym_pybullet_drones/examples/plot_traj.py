# Path: gym_pybullet_drones/examples/plot_traj.py
import numpy as np
import matplotlib.pyplot as plt
import csv

# open log_0.csv and plot the trajectory, the x,y,z coordinates are the second column, third column and fourth column respectively, time is the first column
with open('/Users/makramchahine/PycharmProjects/gym-pybullet-drones/gym_pybullet_drones/examples/results/save-flight-04.14.2023_11.53.37/log_0.csv', 'r') as f:
    reader = csv.reader(f)
    data = np.array(list(reader))
    data = data.astype(np.float32)
    plt.plot(data[:, 0], data[:, 1])
    plt.plot(data[:, 0], data[:, 2])
    plt.plot(data[:, 0], data[:, 3])
    plt.plot(data[:, 0], data[:, 4])
    plt.show()

vx = data[:, 1]
vy = data[:, 2]
vz = data[:, 3]
 #get the yaw angle from the csv file ya0.csv
with open('/Users/makramchahine/PycharmProjects/gym-pybullet-drones/gym_pybullet_drones/examples/results/save-flight-04.14.2023_11.53.37/ya0.csv', 'r') as f:
    reader = csv.reader(f)
    yaw_angle = np.array(list(reader))
    yaw_angle = -yaw_angle[:,1].astype(np.float32)

# # rotate the velocity vector from the body frame to the inertial frame
# # the euler angles are zero, zero and the yaw angle
V_inertial = np.zeros((len(vx), 3))
for i in range(len(vx)):
    V_inertial[i, 0] = vx[i] * np.cos(yaw_angle[i]) - vy[i] * np.sin(yaw_angle[i])
    V_inertial[i, 1] = vx[i] * np.sin(yaw_angle[i]) + vy[i] * np.cos(yaw_angle[i])
    V_inertial[i, 2] = vz[i]

#vplot the velocity vector in the inertial frame
plt.plot(data[:, 0], V_inertial[:, 0])
plt.plot(data[:, 0], V_inertial[:, 1])
plt.plot(data[:, 0], V_inertial[:, 2])
plt.plot(data[:, 0], data[:, 4])
plt.show()

# use ffmpeg to convert images to video
# the images are not zero padded but appear in the correct order with naming according to format frame_{i}.png where i can be any number
# the timestamp of each image is the first column of the csv file


