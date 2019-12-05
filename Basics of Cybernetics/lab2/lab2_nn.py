from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import csv
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

color_map = {1: 'r', 2: 'g', 3: 'b', 4: "c", 5: "m", 6: "y", 7: (1.0, 0.9, 0.5), 8: (0.1, 0.2, 0.5),
             9: (0.8, 0.8, 0.5), 10: (0.5, 0.9, 0.5), 11: (153 / 255, 1.0, 1.0), 12: (0.5, 0.5, 0.0),
             13: (1.0, 0.6, 0.6),
             14: (204 / 255, 255 / 255, 153 / 255), 0: (255 / 255, 204 / 255, 255 / 255)}


def load_csv():
    with open('./data/dataset.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        x_point = []
        y_point = []
        cluster_point = []
        for row in readCSV:
            xa = row[0]
            ya = row[1]
            clustera = row[2]

            x_point.append(xa)
            y_point.append(ya)
            cluster_point.append(clustera)

    x_point.pop(0)
    y_point.pop(0)
    cluster_point.pop(0)

    x_point = list(map(float, x_point))
    y_point = list(map(float, y_point))
    cluster_point = list(map(float, cluster_point))

    x_point = list(map(int, x_point))
    y_point = list(map(int, y_point))
    cluster_point = list(map(int, cluster_point))

    for i in range(len(cluster_point)):
        cluster_point[i] = cluster_point[i] - 1

    return x_point, y_point, cluster_point


def merge(list1, list2):
    merged_list = [[list1[i], list2[i]] for i in range(0, len(list1))]
    return merged_list


def nn():
    normalized_param = 1000000.

    x, y, clusters = load_csv()
    x = np.asarray(x)
    y = np.asarray(y)
    clusters = np.asarray(clusters)

    c = list(zip(x, y, clusters))
    random.shuffle(c)
    x, y, clusters = zip(*c)

    x = np.divide(x, normalized_param)
    y = np.divide(y, normalized_param)
    points = merge(x, y)

    trainingPoints = points[:int(len(points) * 0.4)]
    validationPoints = points[int(len(points) * 0.4): int(len(points) * 0.5)]
    testingPoints = points[int(len(points) * 0.5):]
    trainingPoints = np.array(trainingPoints)
    validationPoints = np.array(validationPoints)
    testingPoints = np.array(testingPoints)

    trainingClusters = clusters[:int(len(points) * 0.4)]
    validationClusters = clusters[int(len(points) * 0.4): int(len(points) * 0.5)]
    testingClusters = clusters[int(len(points) * 0.5):]

    kolor = testingClusters

    trainingClusters = to_categorical(trainingClusters, num_classes=15)
    validationClusters = to_categorical(validationClusters, num_classes=15)
    testingClusters = to_categorical(testingClusters, num_classes=15)

    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(15, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(trainingPoints, trainingClusters, epochs=60, validation_data=(validationPoints, validationClusters))
    score = model.evaluate(testingPoints, testingClusters, verbose=1, sample_weight=None)
    output = model.predict(testingPoints)

    color_list = []
    for out in output:
        color_list.append(np.argmax(out))

    color_list = np.asarray(color_list)

    # plt.scatter(testingPoints[:,0], testingPoints[:,1], color=color_map[color_list[:]], alpha=0.3)
    for i in range(len(testingPoints)):
        plt.scatter(testingPoints[i][0], testingPoints[i][1], color=color_map[color_list[i]], alpha=0.3)
    plt.show()


def main():
    nn()


if __name__ == '__main__':
    main()
