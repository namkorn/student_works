import operator
import math
import random
import numpy as np
from deap import algorithms, base, creator, tools, gp
from functools import partial


def division_operator(numenator, denumenator):
    return numenator / denumenator if denumenator != 0 else 1


def eval_func(individual):
    x, y, z = individual
    mse = 1 / (1 + (x - 2) ** 2 + (y + 1) ** 2 + (z - 1) ** 2)

    return mse,


def create_toolbox():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -10, 10)  # Діапазон для x, y, z
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_func)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


if __name__ == "__main__":
    random.seed(7)
    toolbox = create_toolbox()

    population = toolbox.population(n=2)

    # Записуємо історію максимумів
    max_history = []

    for gen in range(50):  # Кількість поколінь (ngen)
        # Запускаємо еволюцію для одного покоління
        algorithms.eaMuPlusLambda(population, toolbox, mu=10, lambda_=40, cxpb=0.7, mutpb=0.2, ngen=1, stats=None,
                                  halloffame=None, verbose=False)

        # Знаходимо і виводимо максимум для поточного покоління
        best_individual = tools.selBest(population, k=1)[0]
        best_values = best_individual.fitness.values
        max_fitness = 1 / best_values[0]
        max_history.append(max_fitness)

        print(f"Покоління {gen + 1}, Знайдений максимум: {max_fitness}")
        print("Параметри x, y, z:", best_individual)

