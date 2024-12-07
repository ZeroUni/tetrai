import numpy as np
import multiprocessing as mp
from functools import partial
import random
import json
import os
from datetime import datetime
import torch.multiprocessing as torch_mp

from DDQNonCNN import main as train_ddqn

class GeneticOptimizer:
    def __init__(self, 
                 population_size=20,
                 num_generations=10,
                 mutation_rate=0.1,
                 num_episodes=200,
                 max_moves=1000):
        
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.num_episodes = num_episodes
        self.max_moves = max_moves
        
        self.weight_ranges = {
            'score': (0.1, 1.0),
            'lines_cleared': (50, 500),
            'fill_level': (0.1, 1.0),
            'height': (0.1, 1.0),
            'holes': (10, 100),
            'actions_per_piece': (0.5, 3.0),
            'bumpiness': (0.5, 3.0),
            'game_over': (100, 10000)
        }

        self.out_dir = f'genetic_runs/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(self.out_dir, exist_ok=True)

    def create_individual(self):
        return {k: random.uniform(v[0], v[1]) 
                for k, v in self.weight_ranges.items()}

    def evaluate_individual(self, weights, result_queue):
        try:
            score = train_ddqn(weights=weights, 
                             num_episodes=self.num_episodes,
                             max_moves=self.max_moves,
                             display_enabled=False)
            result_queue.put((weights, score))
        except Exception as e:
            print(f"Error evaluating weights: {e}")
            result_queue.put((weights, 0))

    def crossover(self, parent1, parent2):
        child = {}
        for key in self.weight_ranges:
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child

    def mutate(self, individual):
        for key in individual:
            if random.random() < self.mutation_rate:
                min_val, max_val = self.weight_ranges[key]
                individual[key] = random.uniform(min_val, max_val)
        return individual

    def run(self):
        torch_mp.set_start_method('spawn', force=True)
        
        population = [self.create_individual() 
                     for _ in range(self.population_size)]
        
        best_fitness = 0
        best_weights = None
        
        for generation in range(self.num_generations):
            print(f"\nGeneration {generation + 1}")
            
            # Evaluate population in parallel
            result_queue = mp.Queue()
            processes = []
            
            for individual in population:
                p = mp.Process(target=self.evaluate_individual,
                             args=(individual, result_queue))
                p.start()
                processes.append(p)
            
            # Collect results
            results = []
            for _ in range(len(processes)):
                weights, fitness = result_queue.get()
                results.append((weights, fitness))
            
            # Wait for processes
            for p in processes:
                p.join()
            
            # Sort by fitness
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Track best performer
            if results[0][1] > best_fitness:
                best_fitness = results[0][1]
                best_weights = results[0][0]
                
                with open(f'{self.out_dir}/best_weights_gen{generation}.json', 'w') as f:
                    json.dump(best_weights, f)
            
            print(f"Best fitness: {best_fitness}")
            
            # Select top performers
            top_performers = results[:self.population_size//2]
            
            # Create next generation
            next_population = []
            next_population.extend([x[0] for x in top_performers])
            
            while len(next_population) < self.population_size:
                parent1, parent2 = random.choices([x[0] for x in top_performers], k=2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_population.append(child)
                
            population = next_population
        
        return best_weights, best_fitness

if __name__ == "__main__":
    optimizer = GeneticOptimizer(
        population_size=10,
        num_generations=5,
        mutation_rate=0.1,
        num_episodes=100,
        max_moves=500
    )
    best_weights, best_fitness = optimizer.run()
    print(f"\nOptimization complete!")
    print(f"Best weights found: {best_weights}")
    print(f"Best fitness: {best_fitness}")