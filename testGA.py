from deap import base, creator, tools, algorithms
import random
import pandas as pd
import numpy as np
import copy

import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# Fixing the random seed
RANDOM_SEED = 2023
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Parameters and their ranges
# PARAMS = {
#     't_c_num': (1, 5),
#     't_c_dist': (0.00005, 0.00015),
#     't_w_num': (1, 5),
#     't_w_dist': (0.00005, 0.00015),
#     't_w_st_accumuled_length': (20.0, 30.0),
#     't_w_ns_accumuled_length': (10.0, 30.0)
# }

PARAMS = {
    't_c_num': (10, 50),
    't_c_dist': (0.00005, 0.00015),
    't_w_num': (-20, -5),
    't_w_dist': (0.1, 0.2),
    't_w_st_accumuled_length': (20.0, 30.0),
    't_w_ns_accumuled_length': (10.0, 30.0)
}
S_PROBLEM = len(PARAMS.keys())
S_POPULATION = 50
N_GENERATION = 10

def extract_low_up_from_params(params):
    low = [value[0] for value in params.values()]
    up = [value[1] for value in params.values()]
    return low, up

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator
def random_attr(param_name):
    return random.uniform(*PARAMS[param_name])
    
for param_name in PARAMS:
    toolbox.register(param_name, random_attr, param_name)

# Structure initializers
toolbox.register("individual", tools.initCycle, creator.Individual,(
    toolbox.t_c_num,
    toolbox.t_c_dist,
    toolbox.t_w_num,
    toolbox.t_w_dist,
    toolbox.t_w_st_accumuled_length,
    toolbox.t_w_ns_accumuled_length
    ), n=S_PROBLEM)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Placeholder fitness function - to be replaced with actual evaluation logic
def evaluate(individual):
    # Example: Simply summing the parameters - Replace with the real logic
    return (sum(individual),)

toolbox.register("evaluate", evaluate)

# Crossover.
# alpha: 0.5 means that offspring can inherit any value between the values of their parents, giving equal weight to both parents.
toolbox.register("mate", tools.cxBlend, alpha=0.5)

# Mutation.
# mu: The mean of the Gaussian distribution
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0, indpb=0.2)
# toolbox.register("mutate", tools.mutShuffleIndexes,indpb=0.2)

polybound_eta = 2
polybound_indpb = 0.1

# low/up: the lower/up bound of the search space.
polybound_low, polybound_up = extract_low_up_from_params(PARAMS)
toolbox.register("mutate", tools.mutPolynomialBounded, low=polybound_low, up=polybound_up, eta=polybound_eta, indpb=polybound_indpb)

# Selection.
# tournsize: The size of the tournament
toolbox.register("select", tools.selTournament, tournsize=3)

def ga_optimizer():
    initial_pop = toolbox.population(n=S_POPULATION)
    pop = copy.deepcopy(initial_pop)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.0, ngen=N_GENERATION, 
                                   stats=stats, halloffame=hof, verbose=True)
    # Identifying the best solution
    best_ind = tools.selBest(pop, 1)[0]
    print(f"Best individual is {best_ind}, Fitness: {best_ind.fitness.values}")
    return pop, log, best_ind


# def pop_to_df(pop):
#     return pd.DataFrame(data=[ind for ind in pop],
#                         columns=['t_c_num', 't_c_dist', 't_w_num', 't_w_dist',
#                                  't_w_st_accumulated_length', 't_w_ns_accumulated_length'])

# df_pop = pop_to_df(pop)

# # Plotting using parallel_coordinates
# plt.figure(figsize=(12, 6))
# parallel_coordinates(df_pop, class_column=None, colormap=plt.get_cmap("tab10"))
# plt.title('Parallel Coordinates Plot of the First Generation')
# plt.xlabel('Parameters')
# plt.ylabel('Parameter Values')
# plt.grid(True)
# plt.show()

ga_optimizer()

