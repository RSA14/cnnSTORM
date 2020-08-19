import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def plot3D(data: pd.DataFrame, colour=None, point_size=1, depthshade=True):
    x, y, z = data['x'], data['y'], data['z']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=colour, s=point_size, depthshade=depthshade)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

    return


test = pd.read_csv('sequence-as-stack-MT1.N1.LD-AS-Exp_truth.csv', header=0)

plot3D(test)
