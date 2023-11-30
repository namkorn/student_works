#символічна регресія ст231
import operator
import math
import random

import numpy as np
from deap import algorithms, base, creator, tools, gp
from functools import partial

#визначення нових функцій
def division_operator(numenator, denumenator):
    if denumenator == 0:
        return 1
    return numenator / denumenator

#Визначення оціночної функії
def eval_func(individual, points, toolbox):
    #Переробка дерева виражень в викличену функцію
    func = toolbox.compile(expr=individual)
    #Визначення середньоквадратичної помилки
    mse = sum((func(x/100.0, y/100.0, z/100.0) - (1 / (1 + (x - 2)**2 + (y + 1)**2 + (z - 1)**2)))**2
              for x, y, z in points)

    return mse / len(points),

#функція для створення набору інструментів
def create_toolbox():
    pset = gp.PrimitiveSet("MAIN", 3)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(division_operator, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.sin, 1)

    pset.addEphemeralConstant("rand101", partial(random.randint, -1, 1))

    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", eval_func, points=[(x/10., y, z) for x, y, z in zip(range(-100, 100),
                                                                                     [random.uniform(-1, 1) for _ in range(200)],
                                                                                     [random.uniform(-1, 1) for _ in range(200)])],
                                                                                        toolbox=toolbox)

    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    return toolbox

if __name__ == "__main__":
    random.seed(7)
    toolbox = create_toolbox()

    population = toolbox.population(n=450)
    hall_of_fame = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda x: x.fitness.values)
    stats_size = tools.Statistics(len)

    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    probab_crossover = 0.4
    probab_mutate = 0.2
    num_generations = 60

    population, log = algorithms.eaSimple(population, toolbox, probab_crossover, probab_mutate, num_generations,
                                          stats=mstats, halloffame=hall_of_fame, verbose=True)
