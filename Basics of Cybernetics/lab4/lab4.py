from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage import measure
from skimage import filters
from skimage.feature import blob_dog, blob_log, blob_doh
from mpl_toolkits.mplot3d import Axes3D

file_path = './breast.txt'


def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        data.append(line.split(' '))
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = int(data[i][j])

    return data


def main():
    data = np.array(read_file(file_path))
    data = data / 10.

    map_x = 30
    map_y = 30
    print("Creating SOM")
    som = MiniSom(map_x, map_y, 9, sigma=0.7, learning_rate=0.8, random_seed=6)  # initialization of 6x6 SOM
    print("Training SOM")
    som.train_random(data, 1000)

    print("Ploting")
    plt.figure(figsize=(7, 7))
    plt.pcolor(som.distance_map().T)
    plt.colorbar()
    plt.show()

    array = np.asarray(som.distance_map().T)

    array[array >= 0.5] = 1
    array[array < 0.5] = 0
    for i in range(map_x):
        array[0][i] = 1
        array[i][0] = 1
        array[map_x - 1][i] = 1
        array[i][map_x - 1] = 1

    array = 1 - array

    print("Ploting 2")
    plt.figure(figsize=(7, 7))
    plt.pcolor(array.T)
    plt.colorbar()
    plt.show()

    x1 = [elem[0] for elem in data]
    y1 = [elem[1] for elem in data]
    z1 = [elem[2] for elem in data]
    x2 = [elem[3] for elem in data]
    y2 = [elem[4] for elem in data]
    z2 = [elem[5] for elem in data]
    x3 = [elem[6] for elem in data]
    y3 = [elem[7] for elem in data]
    z3 = [elem[8] for elem in data]

    print("Labeling clusters")
    labelled = measure.label(array)
    plt.imshow(labelled)
    plt.show()
    label_vector = []

    print("Winners")
    winners = []
    for i, x in enumerate(data):
        w = som.winner(x)
        # plt.plot(w[0], w[1], 'x', markersize=10, markeredgewidth=1)
        winners.append(w)
        curr_label = labelled[w[0]][w[1]]
        label_vector.append(curr_label)
    # plt.imshow(array)
    # plt.show()

    colors = []
    for mark in label_vector:
        if mark == 0:
            colors.append('b')
        elif mark == 1:
            colors.append('g')
        elif mark == 2:
            colors.append('r')
        elif mark == 3:
            colors.append('y')
        elif mark == 4:
            colors.append('k')
        else:
            colors.append('m')

    fig = plt.figure()
    ax = Axes3D(fig)
    c1 = [colors[i] for i in range(len(x1))]
    ax.scatter(x1, y1, z1, c=c1)
    plt.show()

    fig = plt.figure()
    ax = Axes3D(fig)
    c2 = [colors[i] for i in range(len(x2))]
    ax.scatter(x2, y2, z2, c=c2)
    plt.show()

    fig = plt.figure()
    ax = Axes3D(fig)
    c3 = [colors[i] for i in range(len(x3))]
    ax.scatter(x3, y3, z3, c=c3)
    plt.show()

    xx = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    for i, elem in enumerate(data):
        plt.plot(xx, elem, c=colors[i])
    plt.show()


if __name__ == '__main__':
    main()
