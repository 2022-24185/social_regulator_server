"""Fitness function that calculates the fitness of each genome based on user data."""
from typing import TYPE_CHECKING, Dict
from neat.genome import DefaultGenome
from neuroevolution.evolution.fitness_functions.basic_fitness import BasicFitness

if TYPE_CHECKING:
    from neat.config import Config

Population = Dict[int, DefaultGenome]

class UserEvaluatedFitness(BasicFitness):
    """Fitness function that calculates the fitness of each genome based on user data."""
    def __call__(self, population, config):
        self.calculate_fitness(population, config)
    
    def calculate_fitness(self, population: Population, config: 'Config') -> None:
        """Calculate the fitness of each genome in the population based on user data."""
        times, ratings = zip(*((genome.data.time_since_startup, genome.data.user_rating)
                               for genome in population.values()))
        min_time, max_time = min(times), max(times)
        min_rating, max_rating = min(ratings), max(ratings)

        for genome in population.values():
            normalized_time = (genome.data.time_since_startup - min_time) / (max_time - min_time + 1)
            normalized_rating = (genome.data.user_rating - min_rating) / (max_rating - min_rating + 1)
            genome.fitness = normalized_time + normalized_rating
