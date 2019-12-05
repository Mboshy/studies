import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

individuals_number = 100
population_quantity = 10000


def generate(N):
    """
    Generuje ładnie listę wejść

    :param N: ilość genów na osobnika
    :return: lista wejść u
    """
    u = np.random.uniform(0, 1, size=(individuals_number, N))
    x1 = np.zeros((individuals_number, N))
    x2 = np.zeros((individuals_number, N))
    for i in range(1, u.shape[1]):
        x1[:, i] = x2[:, i-1]
        x2[:, i] = 2*x2[:, i-1] - x1[:, i-1] + (u[:, i-1]/(N**2))

    return u, x1, x2


def fitness_calculate(u, x1, N):
    """
    Oblicza przystosowanie

    :param u: Lista wejść
    :return: lista wartości przystosowań poszczególnych osobników
    """
    u_2 = u[:, :N-1]**2
    u_sum = np.asarray(u_2)
    u_sum = np.sum(u_sum, axis=1)
    fitness = x1[:, N-1] - (u_sum / (2*N))
    return fitness


def mutation(u, N):
    """
    Mutuje każdy z genów każdego osobnika

    :param u: Lista wejść
    :param N: ilość genów na osobnika
    :return: nowa lista wejść u
    """
    x1 = np.zeros((individuals_number, N))
    x2 = np.zeros((individuals_number, N))

    eps = np.random.uniform(-1, 1, size=(individuals_number, N))
    sig = np.random.uniform(0, 0.001, size=(individuals_number, N))

    new_u = u.copy()
    new_u[:, :] = new_u[:, :] + sig[:, :] * eps[:, :]

    for i in range(1, u.shape[1]):
        x2[:, i] = 2*x2[:, i-1] - x1[:, i-1] + (u[:, i-1]/(N**2))
        x1[:, i] = x2[:, i-1]

    return new_u, x1


def select_parents(pop, x1, fitness, num_parents):
    """
    Selekcja najlepszych osobników

    :param pop: lista wejść
    :param x1: lista stanów
    :param fitness: lista przystosowań
    :param num_parents: liczba nowych rodziców
    :return: nowa lista rodziców (lista wejść u wraz z listą stanów x1) z najwięszym przystosowaniem
    """
    rank = np.zeros(len(fitness))
    for i in range(int(individuals_number/4)):
        fitness_aux = fitness.copy()
        aux = np.random.randint(low=0, high=individuals_number*2, size=int(individuals_number/2))
        aux = np.sort(aux)
        auxi = fitness_aux[aux]
        for index, j in enumerate(aux):
            rank_value = (auxi[index] > auxi).sum()
            if rank[j] < rank_value:
                rank[j] = rank_value

    parents = np.zeros((num_parents, pop.shape[1]))
    x = np.zeros((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(rank == np.max(rank))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        x[parent_num, :] = x1[max_fitness_idx, :]
        rank[max_fitness_idx] -= 100

    return parents, x


# TA FUNKCJA NA DOLE DZIAŁA NA TEJ SAMEJ ZASADZIE CO TA U GÓRY, JEST SZYBSZA. ALE W ALGORYTMIE TRZEBA WYZNACZAĆ JESZCZE
#   RANGĘ NA PODSTAWIE PRZYSTOSOWANIA, ZAMIAST OD RAZU CISNĄĆ Z PRZYSTOSOWANIA NAJLEPSZYCH OSOBNIKÓW
# def select_parents(pop, x1, fitness, num_parents):
#     """
#     Selekcja najlepszych osobników
#
#     :param pop: lista wejść
#     :param x1: lista stanów
#     :param fitness: lista przystosowań
#     :param num_parents: liczba nowych rodziców
#     :return: nowa lista rodziców (lista wejść u wraz z listą stanów x1) z najwięszym przystosowaniem
#     """
#     fitness_aux = fitness.copy()
#     parents = np.zeros((num_parents, pop.shape[1]))
#     x = np.zeros((num_parents, pop.shape[1]))
#     for parent_num in range(num_parents):
#         max_fitness_idx = np.where(fitness_aux == np.max(fitness_aux))
#         max_fitness_idx = max_fitness_idx[0][0]
#         parents[parent_num, :] = pop[max_fitness_idx, :]
#         x[parent_num, :] = x1[max_fitness_idx, :]
#         fitness_aux[max_fitness_idx] -= 100
#
#     return parents, x


def main():
    for N in (5, 10, 15, 20, 25, 30, 35, 40, 45):
        # Analityczna wartość
        k_sum = [elem**2 for elem in range(N)]
        k_sum = np.asarray(k_sum)
        k_sum = np.sum(k_sum)
        J_star = 1/3 - (3*N - 1)/(6 * N**2) - (1/(2 * N**3)*k_sum)
        print("Analitycznie: ", "%.6f" % J_star)

        best_outputs = []
        u0, x1_0, x2_0 = generate(N)
        u = u0.copy()
        x1 = x1_0.copy()

        for i in range(population_quantity):
            new_u, new_x1 = mutation(u, N)   # tworzenie potomków
            u = np.append(u, new_u, axis=0)  # rodzice i dzieciaki do jednego worka
            x1 = np.append(x1, new_x1, axis=0)  # rodzice i dzieciaki do jednego worka
            fitness_mut = fitness_calculate(u, x1, N)   # wyliczanie przystosowania
            u, x1 = select_parents(u, x1, fitness_mut, new_u.shape[0])  # wybór rodziców
            fittest = fitness_calculate(u, x1, N)   # wyliczanie nowego przystosowania
            best_outputs.append(np.amax(fittest))

        print("AAAlitAAznie: ", "%.6f" % np.amax(fittest))

        with plt.style.context('seaborn-darkgrid'):
            plt.figure()
            plt.plot(best_outputs)
            plt.xlabel('Iterations')
            plt.ylabel('Fitness')
    plt.show()


if __name__ == '__main__':
    main()
