### script that plots each column of the vel_cmd.csv file vs index
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.signal import butter, filtfilt

def  plottit(csv_file):
    # read csv file into dataframe
    df = pd.read_csv(csv_file)
    # add header to dataframe
    df.columns = ['vx', 'vy', 'vz', 'omega_z']

    # b, a = butter(3, 0.10)
    # df['vx'] = filtfilt(b, a, df['vx'])
    # df['vy'] = filtfilt(b, a, df['vy'])
    # df['vz'] = filtfilt(b, a, df['vz'])
    # df['omega_z'] = filtfilt(b, a, df['omega_z'])


    # plot df.att_x vs df.att_y in a subplot and df.yaw vs df.time_total in another subplot
    fig, axs = plt.subplots(4)
    fig.suptitle('Velocity and angular velocity vs index')
    axs[0].plot(df['vx'][1:])
    axs[0].set_title('vx')
    axs[1].plot(df['vy'][1:])
    axs[1].set_title('vy')
    axs[2].plot(df['vz'][1:])
    axs[2].set_title('vz')
    axs[3].plot(df['omega_z'][1:])
    axs[3].set_title('omega_z')
    for ax in axs.flat:
        ax.set(xlabel='index', ylabel='value')
    # Hide x labels and tick labels for top plots and y ticks for right plots
    for ax in axs.flat:
        ax.label_outer()
    # show plot
    plt.show()
    # save png of plot
    fig.savefig('vel_cmd.png')
    plt.close(fig) # close the figure window to free up memory and avoid plots overlaying each other on
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str)
    args = parser.parse_args()
    plottit(args.csv_file)

