import os
import json
import random
import logging
import pandas as pd
import numpy as np
from math import ceil, log10
from deap import tools
import matplotlib.pyplot as plt
from collections import defaultdict

from toolsQuickUtils import time_decorator

#===================================================================================================
# Basic parameter & Customized Population setup:
PARAMS = {
    'st_c_num': (2, 10), # [3,10)  # min = 3
    'st_w_num': (2, 10), # [3,10)  # min = num_main_floors.
    'ns_w_num': (2, 10), # [2,10)  # min = 2.
    'st_w_accumuled_length_percent': (0.0001, 0.05), # should be more "dependent" on the average length.
    'ns_w_accumuled_length_percent': (0.0001, 0.05), # should be more "dependent" on the average length.
    'st_st_merge': (0.1, 1.00), # ....god sick differentiate between merge
    'ns_st_merge': (0.1, 1.00), # ....god sick differentiate between merge
    'ns_ns_merge': (0.1, 1.00), # ....god sick differentiate between merge
    # 'st_c_align_dist': (0.0001, 0.1), # fixed : 0.001
    'st_w_align_dist': (0.0001, 0.1), # fixed?
    'ns_w_align_dist': (0.0001, 0.1), # fixed?
}

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
# Get the additional Constant Values.
PARAMS_SCALE, PARAMS_INTEGER = getParameterScales(param_ranges=PARAMS)
MinVals, MaxVals = getParameterVarLimits(param_ranges=PARAMS_INTEGER)

# ===================================================================================================
# Basic Functions of GA
def ga_decodeInteger_x(individual: list) -> list:

    decoded_xs = []
    count_parameter_placeholders = []
    for key, (lower, upper) in PARAMS_INTEGER.items():
        upper -= 1
        l = len(str(abs(upper)))
        count_parameter_placeholders.append(l)

    if sum(count_parameter_placeholders) == len(individual):
        
        start_i = 0
        for c in count_parameter_placeholders:
            decoded_x = int(''.join(str(digit) for digit in individual[int(start_i):int(start_i+c)]))
            decoded_xs.append(decoded_x)
            start_i += c
    else:
        raise ValueError("The integer deocder is wrong")
    return decoded_xs

def ga_adjustReal_x(decoded_x):

    decoded_parameters = dict(zip(PARAMS.keys(), decoded_x))
    
    for key, value in list(decoded_parameters.items()):
        
        # rescale back to the real values.
        if PARAMS_SCALE[key] > 1:
            decoded_parameters[key] /= PARAMS_SCALE[key]

        # correct the integer values.
        if '_num' in key:
            decoded_parameters[key] = int(value)

    return decoded_parameters

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

# ===================================================================================================
# Storage and Visualization.

def clearPopulationFile(file_path):

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

def savePopulationThresholds(file_path, values):

    try:
        # Open the file in append mode
        with open(file_path, 'a') as file:
            for v in values:
                file.write(f"{v}\n")

    except Exception as e:
        print(f"Failed to append to the file: {e}")

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

def visualizeGenFitness(
    output_file, logbook, restart_rounds, ind_file, generation_size, set_violin_filter=True):
    
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
    
    # ------------------------------------ Set up the plotting ------------------------------------
    ind_data = read_floats_from_file(ind_file)
    violin_data = [ind_data[i:i + generation_size] for i in range(0, len(ind_data), generation_size)]

    viol_h = 3
    viol_w_per_gen = 3
    plt.figure(figsize=(len(violin_data)*viol_w_per_gen, viol_h), dpi=300)
    fig, ax = plt.subplots()

    # ------------------------------------ Create the violin plot ------------------------------------
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

        # Customize the violin plot with filtered data
        if filtered_violin_data:
            parts = ax.violinplot(filtered_violin_data, positions=filtered_indices)
            for partname in ('cbars', 'cmins', 'cmaxes'):
                parts[partname].set_edgecolor('black')
                parts[partname].set_linestyle('--')
                parts[partname].set_linewidth(0.5)
                parts[partname].set_alpha(0.5)

            axis_x_ticks = filtered_indices + restart_rounds if restart_rounds else filtered_indices
            # Setting x-tick labels to show group numbers
            ax.set_xticks(axis_x_ticks)
            ax.set_xticklabels([f'{i}' for i in axis_x_ticks])
 
    # ------------------------------------ Create the fitness line plot ------------------------------------
    gen = logbook.select("gen")
    min_fitness = logbook.select("min")
    avg_fitness = logbook.select("avg")

    # Plot the minimum and average fitness curves.
    plt.plot(gen, min_fitness, 'b-', linewidth=0.75, label="Minimum Fitness")
    plt.plot(gen, avg_fitness, 'g-', linewidth=0.75, label="Average Fitness")
    
    # Highlight the random restart generations. 
    if restart_rounds:
        rr_avg_fitness = [avg_fitness[gen.index(r_round)] for r_round in restart_rounds]
        # Highlight points on Average Fitness
        plt.scatter(restart_rounds, rr_avg_fitness, s=5, color='r', label="Average Fitness with Random Restart")
    
    # ------------------------------------ Save the plot ------------------------------------
    ax.set_xlim(-5, 205)
    ax.set_ylim(0.0, avg_fitness[0]+0.1)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Over Generations")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def visualizeGenFitness_multiobjectives(
    output_file, logbook, restart_rounds, ind_file, generation_size):
    
    def read_floats_from_file_multiobjectives(file_path):
        float_list = []
        try:
            with open(file_path, 'r') as file:
                for line in file:

                    # Convert each line to a float and add it to the list
                    try:
                        cleaned_line = line.strip('()\n').split(',')
                        float_values = list(map(float, map(str.strip, cleaned_line)))
                        float_list.append(float_values)

                    except ValueError:
                        print(f"Warning: Could not convert '{line.strip()}' to float.")
        except FileNotFoundError:
            print(f"Error: The file {file_path} does not exist.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return float_list
    
    # ------------------------------------ Set up the plotting ------------------------------------    
    plt.figure(figsize=(20,8), dpi=300)
    fig, ax = plt.subplots()

    ind_data = read_floats_from_file_multiobjectives(ind_file)
    array_ind_data = np.array(ind_data)
    num_all_generations = len(array_ind_data) // generation_size
    fitness_results = {1: {'Min': [], 'Avg': [], 'Max': [], 'Std': []},
                    2: {'Min': [], 'Avg': [], 'Max': [], 'Std': []},}

    for i in range(num_all_generations):
        generation_data = array_ind_data[i*generation_size:(i+1)*generation_size]
        for j in range(2):  # Loop through each fitness column
            fitness_results[j+1]['Min'].append(np.min(generation_data[:, j]))
            fitness_results[j+1]['Avg'].append(np.mean(generation_data[:, j]))
            fitness_results[j+1]['Max'].append(np.max(generation_data[:, j]))
            fitness_results[j+1]['Std'].append(np.std(generation_data[:, j]))

    df_fitness1 = pd.DataFrame(fitness_results[1])
    df_fitness2 = pd.DataFrame(fitness_results[2])

    # violin_data = [ind_data[i:i + generation_size] for i in range(0, len(ind_data), generation_size)]

    # # ------------------------------------ Create the violin plot ------------------------------------
    # # plot the whole violin.
    # if not set_violin_filter:

    #     violin_data_positions = range(len(violin_data))
    #     parts = ax.violinplot(violin_data, positions=violin_data_positions)
    #     for partname in ('cbars', 'cmins', 'cmaxes'):
    #         parts[partname].set_edgecolor('black')
    #         parts[partname].set_linestyle('--')
    #         parts[partname].set_linewidth(0.5)
    #         parts[partname].set_alpha(0.5)
    
    # # plot only the filtered violin parts.
    # else:

    #     prev_cmin = None
    #     filtered_violin_data = []
    #     filtered_indices = []

    #     # Filter data based on cmins
    #     for i, gen_data in enumerate(violin_data):
    #         cmin = min(gen_data)

    #         # if it's the initial or the last population or, if it has changes compared to the previous "best" fitness.
    #         if i == 0  or i == len(violin_data) - 1 or prev_cmin is None or cmin != prev_cmin: 
                
    #             filtered_violin_data.append(gen_data)
    #             filtered_indices.append(i)
                
    #         prev_cmin = cmin

    #     # Customize the violin plot with filtered data
    #     if filtered_violin_data:
    #         parts = ax.violinplot(filtered_violin_data, positions=filtered_indices)
    #         for partname in ('cbars', 'cmins', 'cmaxes'):
    #             parts[partname].set_edgecolor('black')
    #             parts[partname].set_linestyle('--')
    #             parts[partname].set_linewidth(0.5)
    #             parts[partname].set_alpha(0.5)

    #         axis_x_ticks = filtered_indices + restart_rounds if restart_rounds else filtered_indices
    #         # Setting x-tick labels to show group numbers
    #         ax.set_xticks(axis_x_ticks)
    #         ax.set_xticklabels([f'{i}' for i in axis_x_ticks])
 
    # ------------------------------------ Create the fitness line plot ------------------------------------
    
    single_gene = list(range(0, num_all_generations))
    plt.plot(single_gene, df_fitness1['Avg'], '#B8860B', linewidth=0.85, label=r"$f_{unbound}$")
    plt.plot(single_gene, df_fitness2['Avg'], '#bc272d', linewidth=0.85, label=r"$f_{distribution}$")
    
    # Fill the shaded areas for df_fitness2
    plt.fill_between(single_gene, df_fitness1['Min'], df_fitness1['Max'], color='#B8860B', alpha=0.1)
    plt.fill_between(single_gene, df_fitness2['Min'], df_fitness2['Max'], color='#bc272d', alpha=0.1)

    # # Plot the minimum and average fitness curves.
    gen = logbook.select("gen")
    # min_fitness = logbook.select("min")
    # plt.plot(gen, min_fitness, 'blue', linewidth=0.95, label="Global Minimum Fitness")
    avg_fitness = logbook.select("avg")
    plt.plot(gen, avg_fitness, '#2e2eb8', linewidth=0.99, label="Average Fitness")

    # Highlight the random restart generations with points on the curve for global average fitness
    if restart_rounds:
        rr_avg_fitness = [avg_fitness[gen.index(r_round)] for r_round in restart_rounds]
        plt.scatter(restart_rounds, rr_avg_fitness, s=5, color='r', label="Average Fitness with Random Restart")
    
    # ------------------------------------ Save the plot ------------------------------------
    # ax.set_ylim(0.0, avg_fitness[0]+0.1)
    ax.set_xlim(0, len(single_gene)+10)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Over Generations")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

#===================================================================================================
# Generation related functions.
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


#===================================================================================================
# DEAP - varAnd
def varAnd(population, toolbox, cxpb, mutpb):

    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover.
    # To combine building blocks of good solutions from diverse chromosomes.
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
    
    # Apply mutation.
    # To introduces new members into the population that are impossible to create by crossover alone.
    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring

#===================================================================================================
# Convergence related functions.

def has_converged(logbook, ngen_converge):
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

#===================================================================================================
# Random Restart related functions.

# conditon functions
def condition_no_improvements(logbook, ngen_threshold_restart):
    """
    ngen_threshold_restart: number of generation with no improvements to restart.
    """
    if len(logbook) < ngen_threshold_restart:
        return False
    
    min_fitnesses = [entry['min'] for entry in logbook[-ngen_threshold_restart:]]
    return len(set(min_fitnesses)) == 1

# conditon functions
def condition_stay_non_best(logbook, ngen_threshold_restart):

    if len(logbook) < ngen_threshold_restart:
        return False

    previous_best_min_fitness = min(entry['min'] for entry in logbook)
    recent_min_fitnesses = min([entry['min'] for entry in logbook[-ngen_threshold_restart:]])
    
    if recent_min_fitnesses > previous_best_min_fitness:
        return True
    else:
        return False
    
# final conditons.
def meet_random_restart_conditions(logbook, ngen_threshold_restart):
    return condition_no_improvements(logbook, ngen_threshold_restart) or condition_stay_non_best(logbook, ngen_threshold_restart)

# rr random selection functions.
def rr_random_selection(input_population, selection_size):
    """
    randomly select the individuals from the input_population to create the selected_population.
    """
    selected_population = random.sample(input_population, selection_size)
    return selected_population

def rr_find_best_ind(population):
    min_fitness = min(ind.fitness.values[0] for ind in population)
    for ind in population:
        if ind.fitness.values[0] == min_fitness:
            # print("ind,", ind)
            # print("fitness,", min_fitness)
            return ind
        
# rr escape selection functions.
def rr_escape_selection(input_population, selection_size, reference_inds=None):
    """
    Select the individuals from the input_population that are most different from the reference_inds.
    
    """

    if reference_inds is None or not reference_inds:
        return None
    
    euclidean_distances = []
    for sublist in input_population:
        min_distance = float('inf')
        for ref in reference_inds:
            distance = np.linalg.norm(np.array(sublist) - np.array(ref))
            if distance < min_distance:
                min_distance = distance
        euclidean_distances.append((min_distance, sublist))

    euclidean_distances.sort(reverse=True, key=lambda x: x[0])
    selected_population = [sublist for _, sublist in euclidean_distances[:selection_size]]
    
    return selected_population

    # old way: compared to one single individual.
    # the Euclidean distances are calculated between the presently available l_best solution and the previ‐ ously memorized locally best points
    #     euclidean_distances = []
    #     for sublist in input_population:
    #         distance = np.linalg.norm(np.array(sublist) - np.array(reference_ind))
    #         euclidean_distances.append((distance, sublist))
    #     euclidean_distances.sort(reverse=True, key=lambda x: x[0])
    #     selected_population = [sublist for _, sublist in euclidean_distances[:selection_size]]
    # return selected_population


def random_restart(
    creator, toolbox, population,
    restart_param_limits, restart_generation_file, restart_round_count, pop_restart=None):
    
    # Change random seed
    new_seed = random.randint(0, 2**32 - 1) + restart_round_count
    random.seed(new_seed)

    # ---------------------------------------- Generate the new population using distance escaping selection.
    pop_restart_ini = len(population)
    createInds(n=pop_restart_ini, param_rangs=restart_param_limits, filename=restart_generation_file)
    toolbox.register("restartpopulation", ga_loadInds, creator.Individual, n=pop_restart_ini, filename=restart_generation_file)
    restart_population = toolbox.restartpopulation()

    pop_random_restart = rr_escape_selection(restart_population, pop_restart, reference_inds=population)
    pop_current_generation = rr_random_selection(population, len(population)-pop_restart)
    new_population = pop_random_restart + pop_current_generation
    return new_population

#===================================================================================================
# The main GA functions : adjusted  DEAP-eaSimple.
@time_decorator
def ga_rr_eaSimple(
    population, creator, toolbox,
    cxpb, mutpb, ngen, set_random_restart=True,
    initial_generation_file=[], fitness_file=[], threshold_file=[],
    stats=None, halloffame=None, verbose=__debug__,
    param_limits=None, ngen_threshold_restart=10,
    pop_restart=None, ngen_converge=10):
    
    # remove the fitness file it already exist in the target folder.
    if fitness_file:
        clearPopulationFile(file_path=fitness_file)
    if threshold_file:
        clearPopulationFile(file_path=threshold_file)

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
    # population_fitnesses = [ind.fitness.values[0] for ind in population]
    population_fitnesses = [ind.fitness.values for ind in population]
    savePopulationFitnesses(file_path=fitness_file, values=population_fitnesses)

    population_thresholds = [list(ind[:]) for ind in population]
    savePopulationThresholds(file_path=threshold_file, values=population_thresholds)

    all_restart_rounds = []
    restart_rounds = 0 # count the num /round of random restart
    restart_history_count = 0 # count the random restart history to avoid "continuous random restart shots.".
    random_start_in_generation = False # trigger of random restart.
    
    # ---------------------------------- initial generation (generation 0) created ----------------------------------

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # --------------------------------------------------------------------
        # Check for convergence. if the past iteration reaches the convergence requirements, break the genetic process. 
        # first to triggle off the convergence conditions.
        if has_converged(logbook, ngen_converge):
            print(f"Converged at generation {gen}")
            break

        # --------------------------------------------------------------------
        # Select the individuals for creating the next generation.
        offspring = toolbox.select(population, len(population))

        # Check for Random Restart.
        if set_random_restart and random_start_in_generation:

            # Use the Random Restart to replace the mutation. count the random restart for one more round.
            restart_rounds += 1
            all_restart_rounds.append(gen)
            print(f"The generation {gen} starts with a random restart (round nr. {restart_rounds})")

            offspring = varAnd(offspring, toolbox, cxpb, mutpb=0) # crossover.

            # ------------------------------------ Main part of Random Restat
            restart_ind_file = initial_generation_file.replace(".txt", f"_{restart_rounds}_restart_at_generation_{gen}.txt")
            offspring = random_restart(
                creator, toolbox,
                offspring,
                param_limits, restart_ind_file, restart_rounds, pop_restart)

            random_start_in_generation = False # After the random_restart, triggle off random restart.
            # ------------------------------------ Main part of Random Restat

        else:
            
            # No Random Restart is needed for current generation.
            # --------------------------------------------------------------------
            # Conduct the crossover and mutation processes
            offspring = varAnd(offspring, toolbox, cxpb, mutpb) # crossover and mutation.

        # --------------------------------------------------------------------
        # Evaluate the individuals with an invalid fitness
        # [Important] "toolbox.evaluate" will run the whole grid generation step.
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
        # population_fitnesses = [ind.fitness.values[0] for ind in population]
        population_fitnesses = [ind.fitness.values for ind in population]
        savePopulationFitnesses(file_path=fitness_file,values=population_fitnesses)
        
        population_thresholds = [list(ind[:]) for ind in population]
        savePopulationThresholds(file_path=threshold_file, values=population_thresholds)

        # --------------------------------------------------------------------
        # [current population] Append the statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        if verbose:
            print(logbook.stream)

        # Calculate for Random Restart.
        if set_random_restart:
            if meet_random_restart_conditions(logbook, ngen_threshold_restart):
                if restart_history_count <= 0:

                    # print(f"In the generation {gen}, it is detected that a random restart is needed") 
                    restart_history_count = ngen_threshold_restart
                    random_start_in_generation = True
            
            restart_history_count-=1

    return population, logbook, all_restart_rounds


# pareto frontier with DEAP in Python
# https://stackoverflow.com/questions/37000488/how-to-plot-multi-objectives-pareto-frontier-with-deap-in-python
# https://arxiv.org/pdf/2305.08852

#===================================================================================================
# Pareto Frontier 

def load_gen_fit_files(gen_file, fit_file):
    """
    Load fitness values from one file and corresponding genome values from another file,
    group them by unique rounded fitness values.
    
    :param gen_file: Path to the file containing genome values (list of lists).
    :param fit_file: Path to the file containing fitness values (as tuples).
    :return: A dictionary where keys are fitness values (rounded) and values are lists of corresponding genome values.
    """
    fit_gen_dict = defaultdict(list)  # Dictionary to store fitness as keys and genome values as list of lists

    try:
        with open(fit_file, 'r') as fit_f, open(gen_file, 'r') as gen_f:
            
            # Iterate over both files line by line
            for fit_line, gen_line in zip(fit_f, gen_f):
                
                fit_line = fit_line.strip().replace('(', '').replace(')', '')
                
                if fit_line:
                    fit_values = tuple(round(float(val), 3) for val in fit_line.split(','))
                    gen_values = [int(val) for val in gen_line.strip().strip('[]').split(', ')]
                    fit_gen_dict[fit_values].append(gen_values)
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        
    return fit_gen_dict

def check_pareto_solutions(fitnesses):

    fitnesses = np.array(fitnesses)  # Ensure input is a numpy array
    
    is_efficient = np.ones(fitnesses.shape[0], dtype=bool)  # Assume all points are Pareto-efficient initially
    
    for i, c in enumerate(fitnesses):
        if is_efficient[i]:
            # Keep any point that is not dominated by any other point
            is_efficient[is_efficient] = np.any(fitnesses[is_efficient] < c, axis=1)
            is_efficient[i] = True  # Ensure the current point is retained
    return is_efficient

def get_pareto_frontier(fit_gen_data):
    
    # Load points from the file
    fit_solutions = list(fit_gen_data.keys())
    
    # Calculate Pareto-efficient points
    pareto_mask = check_pareto_solutions(fit_solutions)
    
    # Separate Pareto front and other points
    pareto_front_solutions = [fit_solutions[i] for i in range(len(fit_solutions)) if pareto_mask[i]]
    non_pareto_solutions = [fit_solutions[i] for i in range(len(fit_solutions)) if not pareto_mask[i]]
    
    # Display some statistics
    print('Number of original points:', len(fit_solutions))
    print('Number of Pareto front points:', len(pareto_front_solutions))
    print('Number of non-Pareto points:', len(non_pareto_solutions))

    return  pareto_front_solutions, non_pareto_solutions

def stairway_lines_from_pareto_frontier(pareto_solutions, limit_max_main_axes, min_x, min_y, delta_border=0.02):
    
    pareto_stairway_solutions = sorted(pareto_solutions, key=lambda x: x[0])
    
    # starting from x_min
    step_x = [min_x]
    step_y = [limit_max_main_axes]
    
    if min_x != pareto_stairway_solutions[0][0]:
        step_x.append(min_x)
        step_y.append(pareto_stairway_solutions[0][1])
        
    for i in range(len(pareto_stairway_solutions)):
        
        x, y = pareto_stairway_solutions[i]
        step_x.append(x) 
        step_y.append(y)

        if i != (len(pareto_stairway_solutions)-1):
            x_extended = pareto_stairway_solutions[i+1][0]
            step_x.append(x_extended)  # Horizontal step
            step_y.append(y)
        else:
            if min_y != pareto_stairway_solutions[i][1]:
                step_x.append(x)
                step_y.append(min_y)
    
    # ending to y_min
    step_x.append(limit_max_main_axes)
    step_y.append(min_y)
    
    return step_x, step_y

def visualization_pareto_frontier(pareto_solutions, non_pareto_solutions, output_file):

    plt.figure(figsize=(6,6), dpi=300)
    
    # non-Pareto points
    other_x, other_y = zip(*non_pareto_solutions)
    plt.scatter(other_x, other_y, marker='o', s=12, edgecolors='#CF3759', facecolors='none', label='GA population')
    limit_max_main_axes = max(other_x + other_y) + 0.02
    
    # Pareto front points
    front_x, front_y = zip(*pareto_solutions)
    plt.scatter(front_x, front_y, marker='s', s=15, edgecolors='#2E2EB8', facecolors='none', label='Pareto front solutions')
    
    limit_min_x_axis = min(other_x + front_x)
    limit_min_y_axis = min(other_y + front_y)
    
    # Pareto front boundary lines
    step_x, step_y = stairway_lines_from_pareto_frontier(
        pareto_solutions, limit_max_main_axes, limit_min_x_axis, limit_min_y_axis)
    plt.plot(step_x, step_y, color='#6C11B3', linewidth=0.75, linestyle='--')

    plt.xlim([-0.02,limit_max_main_axes])
    plt.ylim([-0.02,limit_max_main_axes])
    plt.xlabel(r"$f_{unbound}$", fontsize=14)  # LaTeX format for the x-axis
    plt.ylabel(r"$f_{distribution}$", fontsize=14)  # LaTeX format for the y-axis
    plt.tight_layout()
    plt.grid(False)
    plt.legend(loc='best')
    plt.savefig(output_file, dpi=300)
    plt.close()

def calculate_pareto_front(
    gen_file_path,
    fit_file_path,
    pareto_front_fig_output_file,
    ):

    fit_gen_data = load_gen_fit_files(gen_file_path, fit_file_path)
    pareto_front_solutions, non_pareto_solutions = get_pareto_frontier(fit_gen_data)

    visualization_pareto_frontier(pareto_front_solutions, non_pareto_solutions, pareto_front_fig_output_file)

    pareto_front_data = {fit: fit_gen_data[fit] for fit in pareto_front_solutions if fit in fit_gen_data}
    non_pareto_front_data = {fit: fit_gen_data[fit] for fit in non_pareto_solutions if fit in fit_gen_data}

    return pareto_front_data, non_pareto_front_data

def meta_visualization_pareto_frontier(json_file, output_file, markers, colors):
    
     # Load the data from JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    plt.figure(figsize=(12, 12), dpi=1000)
    
    # Iterate over each "nr" entry in the data
    for i, (nr, results) in enumerate(data.items()):

        # Select a color and marker for this "nr"
        color = colors[i % len(colors)]  # Use different color for each "nr"
        marker = markers[i % len(markers)]  # Use different marker for each "nr"

        # Extract Pareto and non-Pareto data and their corresponding sizes
        pareto_solutions = [eval(k) for k in results['near_data'].keys()]  # Convert stringified tuples back to tuples
        pareto_sizes = [results['near_size'][k] for k in results['near_size'].keys()]  # Get the sizes for near_data

        non_pareto_solutions = [eval(k) for k in results['rest_data'].keys()]
        non_pareto_sizes = [results['rest_size'][k] for k in results['rest_size'].keys()]  # Get the sizes for rest_data

        # Plot Pareto front.
        pareto_sizes_log = np.log10(np.array(pareto_sizes) + 1) * 4 + 30  # Moderate scaling
        front_x, front_y = zip(*pareto_solutions)
        plt.scatter(front_x, front_y, marker=marker, s=pareto_sizes_log, 
                    alpha=1.0, edgecolors='k', facecolors=color, label=f'Pareto front ({nr})')

        # Plot non-Pareto points.
        non_pareto_sizes_log = np.log10(np.array(non_pareto_sizes) + 1) * 4 + 20
        other_x, other_y = zip(*non_pareto_solutions)
        plt.scatter(other_x, other_y, marker=marker, s=non_pareto_sizes_log, 
                    alpha=0.33, edgecolors=color, facecolors='none', label=f'GA population ({nr})')

        # Limit calculations for the axes
        limit_max_main_axes = 1.05
        limit_min_x_axis = min(other_x + front_x)
        limit_min_y_axis = min(other_y + front_y)

        # Pareto front boundary lines (assuming this function is defined elsewhere)
        step_x, step_y = stairway_lines_from_pareto_frontier(
            pareto_solutions, limit_max_main_axes, limit_min_x_axis, limit_min_y_axis)
        plt.plot(step_x, step_y, color=color, linewidth=0.75, linestyle='--')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    near_handles, rest_handles = handles[::2], handles[1::2]
    near_labels, rest_labels = labels[::2], labels[1::2]
    ordered_handles = near_handles + rest_handles
    ordered_labels = near_labels + rest_labels

    # Adjust plot settings
    plt.xlim([-0.05, limit_max_main_axes])
    plt.ylim([-0.05, limit_max_main_axes])
    plt.xticks(fontsize=12)  # Set fontsize for x-axis ticks
    plt.yticks(fontsize=12)  # Set fontsize for y-axis ticks
    plt.xlabel(r"$f_{unbound}$", fontsize=18)  # LaTeX format for the x-axis
    plt.ylabel(r"$f_{distribution}$", fontsize=18)  # LaTeX format for the y-axis
    plt.tight_layout()
    plt.grid(False)
    plt.legend(ordered_handles, ordered_labels, loc='upper left', ncol=2, fontsize=16)
    
    # Save the output file
    plt.savefig(output_file, dpi=1000)
    plt.close()
    