import random

import numpy as np
import torch

from constants import DEVICE
from models.ANFIS.CNNANFIS import HybridCnnAnfis
from models.clustering.HDBScan import get_num_rules_with_hdbscan


class Individual:
    def __init__(self, gene_config_space, genome=None):
        self.gene_config_space = gene_config_space  # Dict with ranges/choices for each gene

        if genome is None:
            # Randomly initialize genes based on config space
            self.genome = {}
            for param_name, config in gene_config_space.items():
                if config['type'] == 'int':
                    self.genome[param_name] = random.randint(config['min'], config['max'])
                elif config['type'] == 'categorical':
                    self.genome[param_name] = random.choice(config['choices'])
                elif config['type'] == 'float':
                    self.genome[param_name] = random.uniform(config['min'], config['max'])
        else:
            self.genome = genome

        # Initialize metrics that will be stored after evaluation
        self.fitness = -float('inf')  # Initialize fitness to a very low value
        self.rmse = float('inf')
        self.pearson = -float('inf')
        self.r2_score = -float('inf')
        self.lr = -float('inf')
        self.batch_size = -float('inf')
        self.epoch = -float('inf')

    def __repr__(self):
        return (f"Individual(FCF={self.genome['firing_conv_filters']}, "
                f"CCF={self.genome['consequent_conv_filters']}, "
                f"Fitness={self.fitness:.6f}, R2={self.r2_score:.6f})")

    def clone(self):
        # Create a deep copy of the individual
        new_individual = Individual(self.gene_config_space, genome=self.genome.copy())
        new_individual.fitness = self.fitness
        new_individual.rmse = self.rmse
        new_individual.pearson = self.pearson
        new_individual.r2_score = self.r2_score
        new_individual.lr = self.lr
        new_individual.batch_size = self.batch_size
        new_individual.epoch = self.epoch
        return new_individual


def fitness_function_anfis(individual, features_X, target_Y, feature_val_X, target_val_Y, fixed_params, scaler_X,
                           scaler_Y, device):
    params = individual.genome

    # Set seeds for reproducibility within this trial
    seed = random.randint(0, 10000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Extract ALL hyperparameters from individual's genome
    firing_conv_filters = params['firing_conv_filters']
    consequent_conv_filters = params['consequent_conv_filters']
    current_lr = params['lr']
    current_batch_size = params['batch_size']
    current_epochs = params['epochs']

    model_params = {
        'input_dim': fixed_params['input_dim'],
        'num_mfs': fixed_params['num_mfs'],
        'num_rules': fixed_params['num_rules'],
        'firing_conv_filters': firing_conv_filters,
        'consequent_conv_filters': consequent_conv_filters,
        'device': device,
        'feature_scaler': scaler_X,
        'target_scaler': scaler_Y
    }
    model = HybridCnnAnfis(**model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, weight_decay=1e-4)
    model.fit(features_X, target_Y, optimizer, batch_size=current_batch_size, epochs=current_epochs)

    _, rmse_error_scaled, pearson_correlation, r2 = model.predict(feature_val_X, target_val_Y)
    # Validation
    print(
        f"rmse_error_scaled: {rmse_error_scaled:.6f}, r2_score: {r2:.6f}, pearson_correlation: {pearson_correlation:.6f}, lr: {current_lr:.6f}, batch_size: {current_batch_size:d}, epochs: {current_epochs:d}")


    # The goal is to MAXIMIZE fitness.
    # Higher Pearson and R2 are good. Higher RMSE is bad.
    fitness_score = r2

    individual.fitness = fitness_score
    individual.rmse = rmse_error_scaled
    individual.pearson = pearson_correlation
    individual.r2_score = r2
    individual.lr = current_lr
    individual.batch_size = current_batch_size
    individual.epoch = current_epochs

    return fitness_score


def selection(population, num_parents):
    # Tournament Selection
    k = 5  # Tournament size
    parents = []
    for _ in range(num_parents):
        tournament_contenders = random.sample(population, k)
        best_contender = max(tournament_contenders, key=lambda ind: ind.fitness)
        parents.append(best_contender)
    return parents


def crossover(parent1, parent2, gene_config_space):
    # Uniform Crossover for multiple genes (hyperparameters)
    child_genome = {}
    for param_name in gene_config_space:
        if random.random() < 0.5:  # 50% chance to inherit from parent1
            child_genome[param_name] = parent1.genome[param_name]
        else:
            child_genome[param_name] = parent2.genome[param_name]

    return Individual(gene_config_space, genome=child_genome)


def mutate(individual, mutation_rate, gene_config_space):
    for param_name, config in gene_config_space.items():
        if random.random() < mutation_rate:
            if config['type'] == 'int':
                # Small random jump within range or completely re-randomize
                if random.random() < 0.5:  # 50% chance for small step
                    step = random.randint(-config['step'], config['step']) if 'step' in config else random.randint(-5,
                                                                                                                   5)
                    new_val = individual.genome[param_name] + step
                else:  # 50% chance for full re-randomization
                    new_val = random.randint(config['min'], config['max'])
                individual.genome[param_name] = max(config['min'], min(config['max'], new_val))

            elif config['type'] == 'categorical':
                # Pick a new random choice, ensuring it's different if possible
                choices = [c for c in config['choices'] if c != individual.genome[param_name]]
                if choices:
                    individual.genome[param_name] = random.choice(choices)

            elif config['type'] == 'float':
                # Add a small random perturbation
                perturbation = random.uniform(-0.1 * (config['max'] - config['min']),
                                              0.1 * (config['max'] - config['min']))
                new_val = individual.genome[param_name] + perturbation
                individual.genome[param_name] = max(config['min'], min(config['max'], new_val))


def genetic_algorithm_anfis(
        gene_config_space,  # Defines ranges/choices for each gene
        X_train, y_train,
        X_val, y_val,
        DEVICE,
        scaler_X, scaler_Y,
        anfis_fixed_params,
        population_size,
        generations,
        mutation_rate,
        num_parents_for_crossover,
        num_elites=2  # Number of best individuals to carry over directly
):
    # Initialize population
    population = [Individual(gene_config_space) for _ in range(population_size)]

    best_individual_overall = None

    print("--- Starting Genetic Algorithm for ANFIS Hyperparameters ---")

    for gen in range(generations):
        print(f"\n--- Generation {gen + 1}/{generations} ---")
        # 1. Evaluate Fitness for current population
        for i, individual in enumerate(population):
            print(
                f"  Evaluating individual {i + 1}/{population_size}: FCF={individual.genome['firing_conv_filters']}, CCF={individual.genome['consequent_conv_filters']}...")
            fitness_function_anfis(individual, X_train, y_train, X_val, y_val, anfis_fixed_params, scaler_X, scaler_Y,
                                   DEVICE)
            print(f"  Individual {i + 1} Fitness: {individual.fitness:.6f}")

        # Sort population by fitness (descending)
        population.sort(key=lambda ind: ind.fitness, reverse=True)

        # Keep track of the best individual in this generation and overall
        current_best = population[0]
        if best_individual_overall is None or current_best.fitness > best_individual_overall.fitness:
            best_individual_overall = current_best.clone()  # Deep copy

        print(f"\nGeneration {gen + 1} Summary:")
        print(f"  Best Fitness = {current_best.fitness:.6f}")
        print(
            f"  Best Params: FCF={current_best.genome['firing_conv_filters']}, CCF={current_best.genome['consequent_conv_filters']}")
        print(
            f"  Best Metrics: RMSE={current_best.rmse:.6f}, Pearson={current_best.pearson:.6f}, R2={current_best.r2_score:.6f}")

        # 2. Selection: Choose parents for crossover
        parents = selection(population, num_parents_for_crossover)

        # 3. Create next generation (offspring)
        next_generation = []

        # Elitism: Keep the few best individuals
        next_generation.extend([ind.clone() for ind in population[:num_elites]])

        # Fill the rest of the new population with offspring
        while len(next_generation) < population_size:
            p1, p2 = random.sample(parents, 2)  # Randomly pick two parents from selected pool
            offspring = crossover(p1, p2, gene_config_space)
            mutate(offspring, mutation_rate, gene_config_space)
            next_generation.append(offspring)

        population = next_generation  # Replace old population with new

    print("\n--- Genetic Algorithm Finished ---")
    print(
        f"Overall Best Individual's Parameters: FCF={best_individual_overall.genome['firing_conv_filters']}, CCF={best_individual_overall.genome['consequent_conv_filters']}")
    print(f"  - Fitness Score: {best_individual_overall.fitness:.6f}")

    # --- MODIFIED: Correctly report the stored metrics ---
    print(f"  - Corresponding RMSE: {best_individual_overall.rmse:.6f}")
    print(f"  - Corresponding Pearson Correlation: {best_individual_overall.pearson:.6f}")
    print(f"  - Corresponding R2 Score: {best_individual_overall.r2_score:.6f}")
    print(f"  - Corresponding LR: {best_individual_overall.lr:.6f}")
    print(f"  - Corresponding Batch Size: {best_individual_overall.batch_size}")
    print(f"  - Corresponding Epoch: {best_individual_overall.epoch}")


    return best_individual_overall.genome

def run_GA(gene_config_space, population_size, generations, mutation_rate, num_parents,num_elites,X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,scaler_X, scaler_y):
    print("\n --- Running GA to optimize hyper-parameters --- \n")
    num_rules_from_hdbscan = get_num_rules_with_hdbscan(X_train_scaled)
    anfis_fixed_params = {
        'input_dim': X_train_scaled.shape[1],
        'num_mfs': 3,
        'num_rules': num_rules_from_hdbscan
    }

    best_genome = genetic_algorithm_anfis(
        gene_config_space=gene_config_space,
        X_train=X_train_scaled, y_train=y_train_scaled,
        X_val=X_val_scaled, y_val=y_val_scaled,
        scaler_X=scaler_X,
        scaler_Y=scaler_y,
        DEVICE=DEVICE,
        anfis_fixed_params=anfis_fixed_params,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        num_parents_for_crossover=num_parents,
        num_elites=num_elites,
    )

    best_genome['num_rules'] = num_rules_from_hdbscan
    return best_genome