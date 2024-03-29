import os
import random
import copy
import logging
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from gridGenerator import GridGenerator

import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates


#===================================================================================================
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants and Paths
PROJECT_PATH = r'C:\dev\phd\enrichIFC\enrichIFC'  # *****************to change to your own local path*****************
DATA_FOLDER_PATH = os.path.join(PROJECT_PATH, 'data', 'data_test')
DATA_RES_PATH = os.path.join(PROJECT_PATH, 'res')
RANDOM_SEED = 2023
S_POPULATION = 10
N_GENERATION = 3

# Parameters and their ranges
PARAMS = {
    't_c_num': (5, 10),
    't_c_dist': (0.00001, 0.0001),
    't_w_num': (2, 5),
    't_w_dist': (0.00001, 0.0001),
    't_w_st_accumuled_length': (20.0, 40.0),
    't_w_ns_accumuled_length': (20.0, 40.0)
}

# Fixing the random seed for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

#===================================================================================================
# Utility Functions
def preparation_of_grid_generation(work_path: str, ifc_model: str, info_columns: str = 'info_columns.json', info_walls: str = 'info_walls.json') -> GridGenerator:
    try:
        generator = GridGenerator(
            os.path.join(work_path, ifc_model),
            os.path.join(work_path, ifc_model, info_columns),
            os.path.join(work_path, ifc_model, info_walls),
        )
        generator.prepare_wall_lengths()
        generator.get_main_storeys_and_directions(num_directions=2)
        generator.enrich_main_storeys()
        return generator
    except Exception as e:
        logging.error(f"Error in grid generation: {e}")
        raise

def extract_low_up_from_params(params: dict) -> tuple:
    low = [value[0] for value in params.values()]
    up = [value[1] for value in params.values()]
    return low, up

#===================================================================================================
# Additional DEAP Functions
def random_attr(param_name: str):
    if isinstance(PARAMS[param_name][0], float):
        return random.uniform(*PARAMS[param_name])
    elif isinstance(PARAMS[param_name][0], int):
        return random.randint(*PARAMS[param_name])

def evaluate(grid_generator: GridGenerator, individual: list) -> tuple:
    try:
        parameter_values = dict(zip(PARAMS.keys(), individual))
        grid_generator.update_parameters(parameter_values)
        grid_generator.create_grids()
        grid_generator.calculate_cross_lost()
        return grid_generator.perct_crossing_ns_wall + grid_generator.perct_crossing_st_wall,
    except Exception as e:
        logging.error(f"Error in evaluation: {e}")
        return float('inf'),  # Return a high loss in case of error

#===================================================================================================
# Get the Grid generator as part of the loss function calculation.
model_paths = [filename for filename in os.listdir(DATA_FOLDER_PATH) if os.path.isfile(os.path.join(DATA_FOLDER_PATH, filename))]
if not model_paths:
    logging.error("No model paths found. Exiting.")
    raise FileNotFoundError("No model files in DATA_FOLDER_PATH")

model_path = model_paths[0]
try:
    init_grid_generator = preparation_of_grid_generation(DATA_RES_PATH, model_path)
except Exception as e:
    logging.error(f"Failed to initialize grid generator: {e}")
    exit()

#===================================================================================================
# DEAP Setup 

# Setup for individual and population 
creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # minimization problem.
creator.create("Individual", list, fitness=creator.FitnessMin) # input is list format.
toolbox = base.Toolbox()

for param_name in PARAMS:
    toolbox.register(param_name, random_attr, param_name)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 [toolbox.__getattribute__(param_name) for param_name in PARAMS], n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Settings for Crossover: cxUniform
cxUniform_indpb = 0.5

# settings for Mutation: mutPolynomialBounded
polybound_low, polybound_up = extract_low_up_from_params(PARAMS)
polybound_eta = 80
polybound_indpb = 0.1

# Setup for GA cores.
toolbox.register("evaluate", evaluate, init_grid_generator)
toolbox.register("mate", tools.cxUniform, indpb=cxUniform_indpb)
toolbox.register("mutate", tools.mutPolynomialBounded, low=polybound_low, up=polybound_up, eta=polybound_eta, indpb=polybound_indpb)
toolbox.register("select", tools.selTournament, tournsize=3)

#===================================================================================================
# GA Optimization.

def ga_optimizer(population_size: int, generation_iteration: int, crossover_prob: float, mutation_prob: float):
    try:
        initial_pop = toolbox.population(n=population_size)
        pop = copy.deepcopy(initial_pop)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        pop, log = algorithms.eaSimple(
            pop, toolbox, cxpb=crossover_prob, mutpb=mutation_prob, ngen=generation_iteration, stats=stats, verbose=True)
        
        best_ind = tools.selBest(pop, 1)[0]
        logging.info(f"Best individual is {best_ind}, Fitness: {best_ind.fitness.values}")
        return initial_pop, pop, log, best_ind
    
    except Exception as e:
        logging.error(f"Error during GA optimization: {e}")
        raise

# Uncomment the line below to run the optimizer
try:
    ga_optimizer(
        population_size=S_POPULATION,
        generation_iteration=N_GENERATION,
        crossover_prob=0.5,
        mutation_prob=0.5)
    
except Exception as error:
    logging.error(f"Optimization failed with error: {error}")
