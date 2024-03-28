import os
import random
import copy
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from gridGenerator import GridGenerator

import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# Constants and Paths
PROJECT_PATH = r'C:\dev\phd\enrichIFC\enrichIFC'
DATA_FOLDER_PATH = os.path.join(PROJECT_PATH, 'data', 'data_test')
DATA_RES_PATH = os.path.join(PROJECT_PATH, 'res')
RANDOM_SEED = 2023
S_POPULATION = 20
N_GENERATION = 4

# Fixing the random seed for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Parameters and their ranges
PARAMS = {
    't_c_num': (2, 6),
    't_c_dist': (0.00001, 0.0001),
    't_w_num': (1, 5),
    't_w_dist': (0.00001, 0.0001),
    't_w_st_accumuled_length': (2.0, 10.0),
    't_w_ns_accumuled_length': (2.0, 10.0)
}

# Utility Functions
def preparation_of_grid_generation(work_path, ifc_model, info_columns='info_columns.json', info_walls='info_walls.json'):
    generator = GridGenerator(
        os.path.join(work_path, ifc_model),
        os.path.join(work_path, ifc_model, info_columns),
        os.path.join(work_path, ifc_model, info_walls),
    )
    generator.prepare_wall_lengths()
    generator.get_main_storeys_and_directions(num_directions=2)
    generator.enrich_main_storeys()
    return generator

def extract_low_up_from_params(params):
    low = [value[0] for value in params.values()]
    up = [value[1] for value in params.values()]
    return low, up

# DEAP Functions and Setup 
def random_attr(param_name):
    if isinstance(PARAMS[param_name][0], float):
        return random.uniform(*PARAMS[param_name])
    elif isinstance(PARAMS[param_name][0], int):
        return random.randint(*PARAMS[param_name])

def evaluate(grid_generator, individual):
    parameter_values = dict(zip(PARAMS.keys(), individual))
    grid_generator.update_parameters(parameter_values)
    grid_generator.create_grids()
    grid_generator.calculate_cross_lost()
    loss = grid_generator.perct_crossing_ns_wall + grid_generator.perct_crossing_st_wall,
    return loss

model_paths = [filename for filename in os.listdir(DATA_FOLDER_PATH) if os.path.isfile(os.path.join(DATA_FOLDER_PATH, filename))]
model_path = model_paths[0]
init_grid_generator = preparation_of_grid_generation(DATA_RES_PATH, model_path)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

[toolbox.register(param_name, random_attr, param_name) for param_name in PARAMS]

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.t_c_num, toolbox.t_c_dist, toolbox.t_w_num, toolbox.t_w_dist,
                  toolbox.t_w_st_accumuled_length, toolbox.t_w_ns_accumuled_length), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate, init_grid_generator)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
polybound_low, polybound_up = extract_low_up_from_params(PARAMS)
toolbox.register("mutate", tools.mutPolynomialBounded, low=polybound_low, up=polybound_up, eta=80, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Genetic Algorithm Execution
def ga_optimizer(population_size, generation_iteration, crossover_prob, mutation_prob):
    initial_pop = toolbox.population(n=population_size)
    pop = copy.deepcopy(initial_pop)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=crossover_prob, mutpb=mutation_prob, ngen=generation_iteration, 
                                         stats=stats, verbose=True)
    
    best_ind = tools.selBest(pop, 1)[0]
    print(f"Best individual is {best_ind}, Fitness: {best_ind.fitness.values}")
    return initial_pop, pop, log, best_ind

# Uncomment the line below to run the optimizer
ga_optimizer(
    population_size=S_POPULATION,
    generation_iteration=N_GENERATION,
    crossover_prob=0.5,
    mutation_prob=0.5)