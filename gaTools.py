import os
import json
import random
import logging
import numpy as np
from math import ceil, log10
from deap import tools
import matplotlib.pyplot as plt

#===================================================================================================
# IFC related.
def getIfcModelPaths(folder_path: str, only_first: bool=False, path_nr: int=0) -> list:
    
    run_model_path = None
    model_paths = [
        filename for filename in os.listdir(folder_path) if \
            os.path.isfile(os.path.join(folder_path, filename)) and filename.endswith('.ifc')]
    
    if not model_paths:
        raise FileNotFoundError("No model files in the given path.")
    model_paths = sorted(model_paths)
    
    if only_first:
        run_model_path = model_paths[0]
    else:
        run_model_path = model_paths[path_nr]
    return run_model_path

#===================================================================================================
# Parameter related.
def getParameterScales(param_ranges):

    # convert the dict ranges.
    integer_param_ranges = {}
    scale_param_ranges = {}

    for key, (lower, upper) in param_ranges.items():

        # Determine the smallest power of 10 that converts both limits to integers
        lower_decimals = ceil(-log10(lower % 1)) if lower % 1 != 0 else 0
        upper_decimals = ceil(-log10(upper % 1)) if upper % 1 != 0 else 0

        # Choose the maximum number of decimal places to determine scale factor
        max_decimals = max(lower_decimals,upper_decimals)
        
        # Apply the scale factor to convert both limits to integers
        scale_factor = 10 ** max_decimals
        new_lower, new_upper = int(lower * scale_factor), int(upper * scale_factor)

        # store the new dictionarys and the scale values.
        scale_param_ranges[key] = scale_factor
        integer_param_ranges[key] = (new_lower, new_upper)

    return scale_param_ranges, integer_param_ranges

def getParameterVarLimits(param_ranges):
    
    str_lowers, str_uppers = [], []

    for key, (lower, upper) in param_ranges.items():
        # the upper value is excluded.
        upper -= 1
        size_max = len(str(abs(upper)))

        str_lowers.append(str(lower).zfill(size_max))
        str_uppers.append(str(upper).zfill(size_max))

    str_lowers = list(map(int, list(''.join(v for v in str_lowers))))
    str_uppers = list(map(int, list(''.join(v for v in str_uppers))))
    
    return str_lowers, str_uppers

#===================================================================================================
# Generation related.
def generateOneInd(param_ranges):

    # convert the dict param_ranges.
    integer_individual = []

    for key, (lower, upper) in param_ranges.items():
        # the upper value is excluded.
        upper -= 1
        size_max = len(str(abs(upper)))
        integer_v = str(random.randint(lower, upper)).zfill(size_max)
        integer_individual.append(integer_v)
        
    integer_individual = list(''.join(v for v in integer_individual))
    integer_individual = list(map(int, integer_individual))

    return integer_individual

def createInds(n, param_rangs, filename):
    
    int_lists = [generateOneInd(param_ranges=param_rangs) for _ in range(n)]

    with open(filename, 'w') as file:
        for int_list in int_lists:
            file.write(str(int_list) + '\n')

def ga_loadInds(creator, n, filename):
    individuals = []
    try:
        # Since the individual data is expected to be a list of integers:
        with open(filename, 'r') as file:
            for line in file:
                individual = list(map(int, line.strip().strip('[]').split(',')))
                individual = creator(individual)
                individuals.append(individual)
    except FileNotFoundError:
        logging.error(f"The file {filename} was not found.")
        return None
    except Exception as e:
        logging.error(f"An error occurred while loading individuals: {e}")
        return None
    
    if len(individuals) == n:
        return individuals
    else:
        logging.warning(f"Loaded {len(individuals)} individuals, expected {n}.")
        return None
    
# ===================================================================================================
# Storage and Visualization.
def saveLogbook(logbook, log_file):
    logbook_json = {}
    logbook_json = logbook
    
    with open(log_file, "w") as output_file:
        json.dump(logbook_json, output_file, indent = 3)

def save_fitness_data(logbook, json_file):
    data = {
        "generation": logbook.select("gen"),
        "min_fitness": logbook.select("min"),
        "max_fitness": logbook.select("max"),
        "avg_fitness": logbook.select("avg")
    }
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

# def visualizeGenFitness(logbook, fitness_file, show_max=True):
    
#     gen = logbook.select("gen")
#     min_fitness = logbook.select("min")
#     max_fitness = logbook.select("max")
#     avg_fitness = logbook.select("avg")

#     plt.figure(figsize=(12, 6), dpi=300)
#     plt.plot(gen, min_fitness, 'b-', label="Minimum Fitness")
#     plt.plot(gen, avg_fitness, 'g-', label="Average Fitness")

#     if show_max:
#         max_fitness = logbook.select("max")
#         plt.plot(gen, max_fitness, 'r-', label="Maximum Fitness")
    
#     plt.xlabel("Generation")
#     plt.ylabel("Fitness")
#     plt.title("Fitness Over Generations")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()

#     plt.savefig(fitness_file, dpi=300)
#     plt.close()  # Close the figure to free up memory

def visualizeGenFitness(output_file, logbook, ind_file, generation_size, set_violin_filter=True):
    
    def read_floats_from_file(file_path):
        float_list = []
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    # Convert each line to a float and add it to the list
                    try:
                        float_value = float(line.strip())
                        float_list.append(float_value)
                    except ValueError:
                        print(f"Warning: Could not convert '{line.strip()}' to float.")
        except FileNotFoundError:
            print(f"Error: The file {file_path} does not exist.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return float_list
    
    # ------------------------------------ Create the violin plot ------------------------------------
    ind_data = read_floats_from_file(ind_file)
    violin_data = [ind_data[i:i + generation_size] for i in range(0, len(ind_data), generation_size)]
    print ("len of violin_data is", len(violin_data))

    viol_h = 3
    viol_w_per_gen = 3
    plt.figure(figsize=(len(violin_data)*viol_w_per_gen, viol_h), dpi=300)
    fig, ax = plt.subplots()

    # plot the whole violin.
    if not set_violin_filter:

        violin_data_positions = range(len(violin_data))
        parts = ax.violinplot(violin_data, positions=violin_data_positions)
        for partname in ('cbars', 'cmins', 'cmaxes'):
            parts[partname].set_edgecolor('black')
            parts[partname].set_linestyle('--')
            parts[partname].set_linewidth(0.5)
            parts[partname].set_alpha(0.5)
    
    # plot only the filtered violin parts.
    else:

        prev_cmin = None
        filtered_violin_data = []
        filtered_indices = []

        # Filter data based on cmins
        for i, gen_data in enumerate(violin_data):
            cmin = min(gen_data)

            # if it's the initial or the last population or, if it has changes compared to the previous "best" fitness.
            if i == 0  or i == len(violin_data) - 1 or prev_cmin is None or cmin != prev_cmin: 
                
                filtered_violin_data.append(gen_data)
                filtered_indices.append(i)
                
            prev_cmin = cmin

        print("filtered_indices: ", filtered_indices)

        # Customize the violin plot with filtered data
        if filtered_violin_data:
            parts = ax.violinplot(filtered_violin_data, positions=filtered_indices)
            for partname in ('cbars', 'cmins', 'cmaxes'):
                parts[partname].set_edgecolor('black')
                parts[partname].set_linestyle('--')
                parts[partname].set_linewidth(0.5)
                parts[partname].set_alpha(0.5)

            # Setting x-tick labels to show group numbers
            ax.set_xticks(filtered_indices)
            ax.set_xticklabels([f'{i}' for i in filtered_indices])
    
    # ------------------------------------ Create the fitness line plot ------------------------------------
    gen = logbook.select("gen")
    min_fitness = logbook.select("min")
    avg_fitness = logbook.select("avg")

    plt.plot(gen, min_fitness, 'b-', linewidth=0.75, label="Minimum Fitness")
    plt.plot(gen, avg_fitness, 'g-', linewidth=0.75, label="Average Fitness")

    
    # ------------------------------------ Save the plot ------------------------------------
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Over Generations")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

#===================================================================================================
# Visualization for GA population fitnesses.

def clearPopulationFitnesses(file_path):

    if os.path.exists(file_path):
        os.remove(file_path)

def savePopulationFitnesses(file_path, values):

    try:
        # Open the file in append mode
        with open(file_path, 'a') as file:
            for v in values:
                file.write(f"{v}\n")

    except Exception as e:
        print(f"Failed to append to the file: {e}")

# DEAP - varAnd
def varAnd(population, toolbox, cxpb, mutpb):

    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring

# Customized convergence function.
def has_converged(logbook, ngen_converge):
# def has_converged(logbook, ngen_converge, std_converge):
    if len(logbook) < ngen_converge:
        return False
    recent_entries = logbook[-ngen_converge:]
    
    # Check if the min fitness has stayed unchanged
    min_fitnesses = [entry['min'] for entry in recent_entries]
    if len(set(min_fitnesses)) != 1:
        return False
    
    # # Check if all std values are less than std_converge
    # std_fitnesses = [entry['std'] for entry in recent_entries]
    # if all(std < std_converge for std in std_fitnesses):
    #     return True
    
    return True

# Customized convergence function.
def is_local_optimal(logbook, ngen_no_improve):
    if len(logbook) < ngen_no_improve:
        return False
    recent_entries = logbook[-ngen_no_improve:]
    
    # Check if the min fitness has stayed unchanged
    min_fitnesses = [entry['min'] for entry in recent_entries]
    if len(set(min_fitnesses)) != 1:
        return False
    
    return True

# Random restart function
def random_restart(
    creator, toolbox, current_population,
    restart_param_limits, restart_generation_file, restart_round_count, pop_restart=None):

    def random_partial_selection(input_list, number):
        selected_items = random.sample(input_list, number)
        return selected_items
    
    # Change random seed
    new_seed = random.randint(0, 2**32 - 1) + restart_round_count
    random.seed(new_seed)
    
    # Generate new individuals
    createInds(n=len(current_population), param_rangs=restart_param_limits, filename=restart_generation_file)
    toolbox.register("restartpopulation", ga_loadInds, creator.Individual, n=len(current_population), filename=restart_generation_file)
    restart_population = toolbox.restartpopulation()
    
    pop_random_restart = random_partial_selection(restart_population, pop_restart)  
    pop_current_generation = random_partial_selection(current_population, len(current_population)-pop_restart)
    new_population = pop_random_restart + pop_current_generation
    
    return new_population

# DEAP - eaSimple [adjusted.]
def ga_eaSimple(
    population, toolbox,
    cxpb, mutpb, ngen, fitness_file=[], 
    stats=None, halloffame=None, verbose=__debug__):
    
    # *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
    if fitness_file:
        clearPopulationFitnesses(file_path=fitness_file)
    # *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  * 
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
    # Collect the fitness of per population.
    population_fitnesses = [ind.fitness.values[0] for ind in population]
    savePopulationFitnesses(file_path=fitness_file,values=population_fitnesses)
    # *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *

    # Begin the generational process
    for gen in range(1, ngen + 1):
        
        # 
        if has_converged(logbook, gen):
            print(f"Converged at generation {gen}")
            break

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # offspring = toolbox.select(population)

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring
        
        # *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
        # Collect the fitness of per population.
        population_fitnesses = [ind.fitness.values[0] for ind in population]
        savePopulationFitnesses(file_path=fitness_file,values=population_fitnesses)
        # *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}

        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        if verbose:
            print(logbook.stream)

    return population, logbook

# DEAP - eaSimple [adjusted.]
def ga_rr_eaSimple(
    population, creator, toolbox,
    cxpb, mutpb, ngen,
    initial_generation_file=[], fitness_file=[], 
    stats=None, halloffame=None, verbose=__debug__, 
    param_limits = None, ngen_no_improve=5, pop_restart=None,
    ngen_converge=10):
    
    # remove the fitness file it already exist in the target folder.
    if fitness_file:
        clearPopulationFitnesses(file_path=fitness_file)
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Collect the fitness of per population.
    population_fitnesses = [ind.fitness.values[0] for ind in population]
    savePopulationFitnesses(file_path=fitness_file, values=population_fitnesses)

    restart_round_count = 0 # count the num /round of random restart
    best_fitness = min(population_fitnesses) # initialization of the fitness for stagnation analysis.
    set_random_start = False # trigger of random restart.
    
    # ---------------------------------- initial generation (generation 0) created ----------------------------------

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # --------------------------------------------------------------------
        # Check for convergence. if the past iteration reaches the convergence requirements, break the genetic process. 
        if has_converged(logbook, ngen_converge):
            print(f"Converged at generation {gen}")
            break

        # Check for Random Restart.
        if set_random_start:

            # Start the Random Restart: count the random restart for one more round.
            restart_round_count += 1
            print(f"The generation {gen} starts with a random restart (round nr. {restart_round_count})")

            # ------------------------------------ Main part of Random Restat
            restart_ind_file = initial_generation_file.replace(".txt", f"_{restart_round_count}_restart_at_generation_{gen}.txt")
            population = random_restart(creator, toolbox, population, param_limits, restart_ind_file, restart_round_count, pop_restart)
            # ------------------------------------ Main part of Random Restat

            # After the random_restart: 1. refresh the "no_improve_count"; 2. triggle off random restart.
            set_random_start = False

        # --------------------------------------------------------------------
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # --------------------------------------------------------------------
        # Conduct the crossover and mutation processes
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # --------------------------------------------------------------------
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # --------------------------------------------------------------------
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # --------------------------------------------------------------------
        # Replace the current population by the newly generated offspring (which is the population of the current generation)
        population[:] = offspring
        
        # --------------------------------------------------------------------
        # [current population] Collect and Store the fitness statistics.
        population_fitnesses = [ind.fitness.values[0] for ind in population]
        savePopulationFitnesses(file_path=fitness_file,values=population_fitnesses)

        # --------------------------------------------------------------------
        # [current population] Append the statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        # *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
        # Check if a "Random Restart" is needed ?
        # Starting the next generation via a Random Restart.

        if is_local_optimal(logbook, ngen_no_improve):
            print(f"In the generation {gen}, it is detected that a random restart is needed")
            set_random_start = True

        # current_best_fitness = min(population_fitnesses)
        # if current_best_fitness == best_fitness:
        #     no_improve_count += 1
        # else:
        #     no_improve_count = 0
        #     best_fitness = current_best_fitness

        # if no_improve_count >= ngen_no_improve:
        #     print(f"In the generation {gen}, it is detected that a random restart is needed")
        #     set_random_start = True

            # restart_round_count += 1
            # print(f"Random restart at generation {gen}, Restart round {restart_round_count}")
            # current_population = population
            # restart_file_name = initial_generation_file.replace(".txt", f"_restart_{restart_round_count}.txt")
            # population = random_restart(
            #     creator, toolbox, current_population, 
            #     param_limits, restart_file_name, restart_round_count, restart_ratio=0.8)
            # no_improve_count = 0
            #         
            
        if verbose:
            print(logbook.stream)

    return population, logbook