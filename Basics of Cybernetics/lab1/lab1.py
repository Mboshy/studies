import numpy as np
import matplotlib.pyplot as plt
import random
import math

data = "s1.txt"
centroids_number = 15
colors = {1: (0.2, 0.3, 0.4),
          2: (0.7, 0.3, 0.3),
          3: (0.5, 0.7, 0.9),
          4: (1.0, 1.0, 0),
          5: (0.5, 1.0, 0.5),
          6: (0, 1.0, 1.0),
          7: (1.0, 0.5, 0.5),
          8: (1.0, 0, 1.0),
          9: (0.5, 0.5, 1.0),
          10: (0, 1.0, 0),
          11: (0.3, 0.7, 0.3),
          12: (0.3, 0.3, 0.7),
          13: (1.0, 0, 0),
          14: (0.0, 0.0, 0.0),
          15: (0, 0, 1.0)}
centroids = {
    1: [250000, 850000],
    2: [400000, 800000],
    3: [650000, 870000],
    4: [820000, 730000],
    5: [850000, 550000],
    6: [600000, 600000],
    7: [330000, 560000],
    8: [130000, 560000],
    9: [150000, 350000],
    10: [400000, 400000],
    11: [620000, 400000],
    12: [800000, 300000],
    13: [840000, 160000],
    14: [500000, 170000],
    15: [310000, 170000]
}


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cluster = None
        self.color = None

    def closest_centroid(self, centroid):
        mini = 1000000
        closest = 0
        for i in centroid:
            if mini > math.sqrt((self.x - i.x)**2 + (self.y - i.y)**2):
                mini = math.sqrt((self.x - i.x)**2 + (self.y - i.y)**2)
                closest = i.cluster

        self.cluster = closest
        self.color = colors[closest]


class Centroid:
    def __init__(self, x, y, cluster):
        self.x = x
        self.y = y
        self.cluster = cluster

    def update_pos(self, new_x, new_y):
        self.x = new_x
        self.y = new_y


def load_points():
    with open(data) as content:
        point_pos = []
        coordinates = content.readlines()
        for line in coordinates:
            points = line.split("    ")
            point_pos.append(points[1:])

    return point_pos


def main():
    finished = False
    point_pos = load_points()

    x = [int(elem[0]) for elem in point_pos]
    y = [int(elem[1]) for elem in point_pos]
    cent = []
    poi = []

    for i in centroids.keys():
        centroid = Centroid(centroids[i][0], centroids[i][1], i)
        cent.append(centroid)

    for elem in point_pos:
        point = Point(int(elem[0]), int(elem[1]))
        poi.append(point)

    for pt in poi:
        pt.closest_centroid(cent)

    color_s = [pt.color for pt in poi]
    plt.scatter(x, y, color=color_s, alpha=0.3)
    for c in cent:
        plt.scatter(c.x, c.y, color=colors[c.cluster], marker="o", s=200, edgecolors='k')
    plt.show()

    x_sum = 0
    y_sum = 0
    ending = 0

    while not finished:
        for cluster in cent:
            i = 0
            for ptk in poi:
                if cluster.cluster == ptk.cluster:
                    x_sum = x_sum + ptk.x
                    y_sum = y_sum + ptk.y
                    i += 1
            try:
                new_x = x_sum / i
                new_y = y_sum / i
            except:
                pass
            if abs(cluster.x - new_x) < 1000 and abs(cluster.y - new_y) < 1000:
                ending += 1
            cluster.update_pos(new_x, new_y)
            x_sum = 0
            y_sum = 0

        if ending == 13:
            break

        for pt in poi:
            pt.closest_centroid(cent)

        color_s = [pt.color for pt in poi]
        plt.scatter(x, y, color=color_s, alpha=0.3)

        for c in cent:
            plt.scatter(c.x, c.y, color=colors[c.cluster], marker="o", s=200, edgecolors='k')
        plt.show()
        ending = 0





if __name__ == "__main__":
    main()
