import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


a = 1.1         # Parametr z zadania
x_0 = 100       # Parametr z zadania
individuals_number = 100        # Parametr do zabawy - iloś osobników na pokolenie
population_quantity = 40000     # Parametr do zabawy - ilość pokoleń


def generate(N):
    """
    Generuje ładnie listę wejść

    :param N: ilość genów na osobnika
    :return: lista wejść u
    """
    x = np.empty((individuals_number, N+1))
    x[:, 0] = x_0
    x[:, N] = x_0
    u = np.random.uniform(5, 6, size=(individuals_number, N))
    # Generowanie stanów
    for i in range(1, u.shape[1]):
        x[:, i] = a*x[:, i-1] - u[:, i-1]
    u[:, N-1] = a*x[:, N-1] - x[:, N]

    return u


def fitness_calculate(u):
    """
    Oblicza przystosowanie

    :param u: Lista wejść
    :return: lista wartości przystosowań poszczególnych osobników
    """
    fitness = np.sqrt(u)
    fitness = np.sum(fitness, axis=1)
    return fitness


def mutation(u, N):
    """
    Mutuje każdy z genów każdego osobnika

    :param u: Lista wejść
    :param N: ilość genów na osobnika
    :return: nowa lista wejść u
    """
    x = np.empty((individuals_number, N + 1))
    x[:, 0] = x_0
    x[:, N] = x_0

    eps = np.random.uniform(-1, 1, size=(individuals_number, N))
    sig = np.random.uniform(0, 0.25, size=(individuals_number, N))

    new_u = u.copy()
    new_u[:, :] = new_u[:, :] + sig[:, :] * eps[:, :]
    # Jak coś jest ujemnego to wysypuje program. Dlatego dodano to coś na dole.
    if (new_u < 0).any():
        for i, j in zip(np.where(new_u < 0)[0], np.where(new_u < 0)[1]):
            aux = new_u[i][j]
            new_u[i][j] = 0
            new_u[i][np.argmax(new_u[i])] += aux
    # Generowanie stanów
    for i in range(1, new_u.shape[1]):
        x[:, i] = a*x[:, i-1] - new_u[:, i-1]
    new_u[:, N-1] = a*x[:, N-1] - x[:, N]

    return new_u


def select_parents(pop, fitness, num_parents):
    """
    Selekcja najlepszych osobników

    :param pop: lista wejść
    :param fitness: przystosowanie każdego z osobników
    :param num_parents: ilość nowych rodziców
    :return: nowa lista rodziców (lista wejść) z najwięszym przystosowaniem
    """
    rank = np.zeros(len(fitness))
    for i in range(int(individuals_number/8)):
        fitness_aux = fitness.copy()
        aux = np.random.choice(range(individuals_number*2), int(individuals_number/2), replace=False)   # losowo wybrane indeksy
        aux = np.sort(aux)
        auxi = fitness_aux[aux]
        for index, j in enumerate(aux):
            rank_value = (auxi[index] > auxi).sum()
            if rank[j] < rank_value:
                rank[j] = rank_value

    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(rank == np.max(rank))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        rank[max_fitness_idx] = -1

    return parents

# TA FUNKCJA NA DOLE DZIAŁA NA TEJ SAMEJ ZASADZIE CO TA U GÓRY, JEST SZYBSZA. ALE W ALGORYTMIE TRZEBA WYZNACZAĆ JESZCZE
#   RANGĘ NA PODSTAWIE PRZYSTOSOWANIA, ZAMIAST OD RAZU CISNĄĆ Z PRZYSTOSOWANIA NAJLEPSZYCH OSOBNIKÓW
# def select_parents(pop, fitness, num_parents):
#     """
#     Selekcja najlepszych osobników
#
#     :param pop: lista wejść
#     :param fitness: przystosowanie każdego z osobników
#     :param num_parents: ilość nowych rodziców
#     :return: nowa lista rodziców (lista wejść) z najwięszym przystosowaniem
#     """
#
#     parents = np.empty((num_parents, pop.shape[1]))
#     for parent_num in range(num_parents):
#         max_fitness_idx = np.where(fitness == np.max(fitness))
#         max_fitness_idx = max_fitness_idx[0][0]
#         parents[parent_num, :] = pop[max_fitness_idx, :]
#         fitness[max_fitness_idx] = -1000
#
#     return parents


def main():
    index = 0   # Przydatne do rysowania
    for N in (2, 4, 10, 20, 45):
        best_outputs = []       # lista najlepszych przystosowań z każdego pokolenia
        J_star = sqrt((x_0 * (a ** N - 1) ** 2) / (a ** (N - 1) * (a - 1)))     # analityczna wartość
        print("Analitycznie: ", "%.6f" % J_star)
        u_0 = generate(N)
        new_u = u_0.copy()
        u = new_u.copy()

        for i in range(population_quantity):
            new_u = mutation(u, N)          # tworzenie potomków
            u = np.append(u, new_u, axis=0)  # rodzice i dzieciaki do jednego worka
            fitness_mut = fitness_calculate(u)  # wyliczanie przystosowania
            u = select_parents(u, fitness_mut, new_u.shape[0])  # wybór rodziców
            best_outputs.append(np.amax(fitness_mut))   # najlepsze przystosowanie do listy
            if abs(np.amax(fitness_mut) - J_star) < 0.000004:
                break
        print("Genetyczniee: ", "%.6f" % np.amax(fitness_mut))

        with plt.style.context('seaborn-darkgrid'):
            plt.figure()
            plt.plot(best_outputs)
            plt.xlabel('Iterations')
            plt.ylabel('Fitness')

        index += 1
    plt.show()


if __name__ == '__main__':
    main()
