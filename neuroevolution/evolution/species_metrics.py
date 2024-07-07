from math import ceil
from typing import List
from neat.math_util import mean

class SpeciesMetrics:
    
    @staticmethod
    def get_adjusted_mean_fitness(all_fitnesses: List[float], species_fitnesses: List[float]) -> float:
        """Get the adjusted mean fitness of species."""
        if not all_fitnesses or not species_fitnesses:
            return 0.0
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        fitness_range = max(1.0, max_fitness - min_fitness) if max_fitness != min_fitness else 1.0
        return (mean(species_fitnesses) - min_fitness) / fitness_range


    @staticmethod
    def get_reproduction_cutoff(survival_threshold: int, evaluated: int, min_species_size: int) -> float:
        """Get the reproduction cutoff."""
        return max(int(ceil(survival_threshold * evaluated)), min_species_size)

    @staticmethod
    def compute_offspring_count(deficit: int, norm: float, min_size: int): 
        normalized_deficit = int(round(deficit * norm))
        return max(min_size, normalized_deficit)

    @staticmethod
    def compute_species_deficit(expected_species_size: int, evaluated: int):
        """
        Computes the population deficit for the species.
        
        :param expected_species_size: The expected size of the species.
        :return: The population deficit.
        """
        size_diff = (expected_species_size - evaluated) * 0.5
        rounded_diff = int(round(size_diff))
        population_deficit = evaluated
        if abs(rounded_diff) > 0:
            population_deficit += rounded_diff
        elif size_diff > 0:
            population_deficit += 1
        elif size_diff < 0:
            population_deficit -= 1
        #print(f"🧪Species {self.key} has population deficit {population_deficit}")
        return population_deficit
    
    @staticmethod
    def compute_expected_size(total_adjusted_fitness: float, species_adjusted_fitness: float, evaluated: int, min_species_size: int) -> float:
        """Compute the expected size of a species."""
        if total_adjusted_fitness > 0:
            fitness_adjusted_size = species_adjusted_fitness / total_adjusted_fitness * evaluated
            expected_species_size = max(min_species_size, fitness_adjusted_size)
        else:
            expected_species_size = min_species_size
        return expected_species_size