import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import csv


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
          15: (0, 0, 1.0),
          20: (0, 0, 0)}


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

    return x_point, y_point, cluster_point


def main():
    xp, yp, cp = load_csv()
    xp = np.asarray(xp)
    yp = np.asarray(yp)
    cp = np.asarray(cp)

    xpos = ctrl.Antecedent(np.arange(0, 1000000, 1000), 'xpos')
    ypos = ctrl.Antecedent(np.arange(0, 1000000, 1000), 'ypos')
    cluster = ctrl.Consequent(np.arange(0, 15, 1), 'cluster')

    xpos['0'] = fuzz.trimf(xpos.universe, [0, 0, 235000])
    xpos['1'] = fuzz.trimf(xpos.universe, [235000, 280000, 325000])
    xpos['2'] = fuzz.trimf(xpos.universe, [325000, 360000, 407000])
    xpos['3'] = fuzz.trimf(xpos.universe, [407000, 455000, 513000])
    xpos['4'] = fuzz.trimf(xpos.universe, [513000, 550000, 605000])
    xpos['5'] = fuzz.trimf(xpos.universe, [605000, 660000, 733000])
    xpos['6'] = fuzz.trimf(xpos.universe, [733000, 870000, 1000000])

    ypos['0'] = fuzz.trimf(ypos.universe, [0, 0, 217000])
    ypos['1'] = fuzz.trimf(ypos.universe, [217000, 240000, 263000])
    ypos['2'] = fuzz.trimf(ypos.universe, [263000, 330000, 413000])
    ypos['3'] = fuzz.trimf(ypos.universe, [413000, 450000, 490000])
    ypos['4'] = fuzz.trimf(ypos.universe, [490000, 560000, 635000])
    ypos['5'] = fuzz.trimf(ypos.universe, [635000, 660000, 687000])
    ypos['6'] = fuzz.trimf(ypos.universe, [687000, 755000, 835000])
    ypos['7'] = fuzz.trimf(ypos.universe, [835000, 970000, 1000000])

    cluster['0'] = fuzz.trimf(cluster.universe, [0, 0, 2])
    cluster['1'] = fuzz.trimf(cluster.universe, [1, 1.6, 3])
    cluster['2'] = fuzz.trimf(cluster.universe, [2, 2.6, 4])
    cluster['3'] = fuzz.trimf(cluster.universe, [3, 3.6, 5])
    cluster['4'] = fuzz.trimf(cluster.universe, [4, 4.6, 6])
    cluster['5'] = fuzz.trimf(cluster.universe, [5, 5.6, 7])
    cluster['6'] = fuzz.trimf(cluster.universe, [6, 6.6, 8])
    cluster['7'] = fuzz.trimf(cluster.universe, [7, 7.6, 9])
    cluster['8'] = fuzz.trimf(cluster.universe, [8, 8.6, 10])
    cluster['9'] = fuzz.trimf(cluster.universe, [9, 9.6, 11])
    cluster['10'] = fuzz.trimf(cluster.universe, [10, 10.6, 12])
    cluster['11'] = fuzz.trimf(cluster.universe, [11, 11.6, 13])
    cluster['12'] = fuzz.trimf(cluster.universe, [12, 12.6, 14])
    cluster['13'] = fuzz.trimf(cluster.universe, [13, 13.6, 15])
    cluster['14'] = fuzz.trimf(cluster.universe, [14, 14.6, 16])

    rule1 = ctrl.Rule((xpos['0'] | xpos['1']) & (ypos['6'] | ypos['7']), cluster['0'])
    rule2 = ctrl.Rule((xpos['2'] | xpos['3']) & (ypos['6'] | ypos['7']), cluster['1'])
    rule3 = ctrl.Rule((xpos['4'] | xpos['5']) & (ypos['6'] | ypos['7']), cluster['2'])
    rule4 = ctrl.Rule((xpos['6']) & (ypos['5'] | ypos['6'] | ypos['7']), cluster['3'])
    rule5 = ctrl.Rule((xpos['5'] | xpos['6']) & (ypos['3'] | ypos['4']), cluster['4'])
    rule6 = ctrl.Rule((xpos['4'] | xpos['5']) & (ypos['4'] | ypos['5']), cluster['5'])
    rule7 = ctrl.Rule((xpos['1'] | xpos['2'] | xpos['3']) & (ypos['4'] | ypos['5']), cluster['6'])
    rule8 = ctrl.Rule((xpos['0']) & (ypos['4'] | ypos['5']), cluster['7'])
    rule9 = ctrl.Rule((xpos['0']) & (ypos['1'] | ypos['2'] | ypos['3']), cluster['8'])
    rule10 = ctrl.Rule((xpos['1'] | xpos['2'] | xpos['3']) & (ypos['1'] | ypos['2'] | ypos['3']), cluster['9'])
    rule11 = ctrl.Rule((xpos['4'] | xpos['5']) & (ypos['1'] | ypos['2'] | ypos['3']), cluster['10'])
    rule12 = ctrl.Rule((xpos['6']) & (ypos['1'] | ypos['2'] | ypos['3']), cluster['11'])
    rule13 = ctrl.Rule((xpos['5'] | xpos['6']) & (ypos['0']), cluster['12'])
    rule14 = ctrl.Rule((xpos['3'] | xpos['4']) & (ypos['0'] | ypos['1']), cluster['13'])
    rule15 = ctrl.Rule((xpos['0'] | xpos['1'] | xpos['2']) & (ypos['0'] | ypos['1']), cluster['14'])

    # cluster.view()

    cluster_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8,
                                       rule9, rule10, rule11, rule12, rule13, rule14, rule15])
    clustering = ctrl.ControlSystemSimulation(cluster_ctrl)

    similarity = 0
    color_fuzz = []
    for x, y, label in zip(xp, yp, cp):
        clustering.input['xpos'] = x
        clustering.input['ypos'] = y
        try:
            clustering.compute()
            # print(int(round(clustering.output['cluster'])), '  ', label)
            color_fuzz.append(int(round(clustering.output['cluster'])))
            if label == int(round(clustering.output['cluster'])):
                similarity += 1
        except Exception as e:
            color_fuzz.append(20)
    print("% podobienstwa:", similarity/5000.*100)

    coloror = [colors[das] for das in color_fuzz]
    plt.scatter(xp, yp, color=coloror)
    plt.show()


if __name__ == '__main__':
    main()
