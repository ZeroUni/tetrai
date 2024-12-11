import numpy as np
import multiprocessing as mp
from functools import partial
import random
import json
import os
from datetime import datetime
import torch.multiprocessing as torch_mp

import gc
import torch

import argparse

from typing import List, Dict

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
            'lines_cleared': (10, 500),
            'fill_level': (0.1, 1.0),
            'height': (0.1, 1.0),
            'holes': (1, 100),
            'actions_per_piece': (0.1, 3.0),
            'bumpiness': (0.1, 3.0),
            'game_over': (100, 10000)
        }

        self.out_dir = f'genetic_runs/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(self.out_dir, exist_ok=True)

    def create_individual(self):
        return {k: random.uniform(v[0], v[1]) 
                for k, v in self.weight_ranges.items()}

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

    def evaluate_individual(self, weights, result_queue, completion_queue):
        try:
            score = train_ddqn(weights=weights, 
                             num_episodes=self.num_episodes,
                             max_moves=self.max_moves,
                             display_enabled=False)
            print(f"Evaluated weights: {weights}, score: {score}")
            
            if score is None or not np.isfinite(score):
                score = float('-inf')

            result_queue.put((weights, score, None))  # None = no error

            # Cleanup and end this torch process
            print("Closing torch process...")
            completion_queue.put(torch_mp.current_process().pid)
        except Exception as e:
            print(f"Error evaluating weights: {e}")
            result_queue.put((None, None, str(e)))  # Signal error
            completion_queue.put(mp.current_process().pid)

    def run(self):
        torch_mp.set_start_method('spawn', force=True)
        
        population = [self.create_individual() 
                     for _ in range(self.population_size)]
        
        best_fitness = 0
        best_weights = None
        
        try:
            for generation in range(self.num_generations):
                print(f"\nGeneration {generation + 1}")
                
                result_queue = torch_mp.Queue()
                completion_queue = torch_mp.Queue()
                processes = {}
                active_processes = 0
                individual_index = 0
                
                # Process population in batches of 4
                while individual_index < len(population):
                    # Start new processes until we hit limit or run out
                    while active_processes < 3 and individual_index < len(population):
                        p = torch_mp.Process(
                            target=self.evaluate_individual,
                            args=(population[individual_index], result_queue, completion_queue)
                        )
                        p.start()
                        processes[p.pid] = p
                        active_processes += 1
                        individual_index += 1

                    if not completion_queue.empty():
                        pid = completion_queue.get()
                        p = processes[pid]
                        if p.is_alive():
                            p.terminate()
                            p.join()
                        del processes[pid]
                        active_processes -= 1
                    
                    # Wait for a process to finish before starting new ones
                    for p in processes.values():
                        if not p.is_alive():
                            p.join()
                            active_processes -= 1
                            del processes[p.pid]
                            break
                
                # Collect results with error handling
                results = []
                try:
                    for _ in range(len(population)):
                        weights, fitness, error = result_queue.get()
                        if error is not None:
                            raise RuntimeError(f"Child process failed: {error}")
                        results.append((weights, fitness))
                except Exception as e:
                    print(f"Error during evaluation: {e}")
                    self.cleanup_processes(processes)
                    raise
                
                # Clean join remaining processes
                for p in processes.values():
                    if p.is_alive():
                        p.join()
                
                # Rest of generation logic remains the same...
                results.sort(key=lambda x: x[1], reverse=True)
                
                if results[0][1] > best_fitness:
                    best_fitness = results[0][1]
                    best_weights = results[0][0]
                    
                    with open(f'{self.out_dir}/best_weights_gen{generation}.json', 'w') as f:
                        json.dump(best_weights, f)
                
                print(f"Best fitness: {best_fitness}")
                
                top_performers = results[:self.population_size//2]
                next_population = []
                next_population.extend([x[0] for x in top_performers])
                
                while len(next_population) < self.population_size:
                    parent1, parent2 = random.choices([x[0] for x in top_performers], k=2)
                    child = self.crossover(parent1, parent2)
                    child = self.mutate(child)
                    next_population.append(child)
                    
                population = next_population
    
        except KeyboardInterrupt:
            print("\nReceived interrupt, cleaning up...")
            self.cleanup_processes(processes)
            raise
            
        except Exception as e:
            print(f"\nError occurred: {e}")
            self.cleanup_processes(processes)
            raise
            
        return best_weights, best_fitness

    def cleanup_processes(self, processes: Dict[int, mp.Process]):
        """Terminate all processes and join them"""
        for p in processes.values():
            if p.is_alive():
                p.terminate()
                p.join()



if __name__ == "__main__":
    # Check for debug flag
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False)
    args = parser.parse_args()
    if args.debug:
        print("Running in debug mode")
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    optimizer = GeneticOptimizer(
        population_size=10,
        num_generations=20,
        mutation_rate=0.1,
        num_episodes=200,
        max_moves=100
    )
    best_weights, best_fitness = optimizer.run()
    print(f"\nOptimization complete!")
    print(f"Best weights found: {best_weights}")
    print(f"Best fitness: {best_fitness}")