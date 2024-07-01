"""Fitness function that calculates the fitness of each genome based on user data."""
from typing import TYPE_CHECKING, Dict, Optional
from neat.genome import DefaultGenome
from neuroevolution.evolution.fitness_functions.basic_fitness import BasicFitness
from neuroevolution.server.models import UserData

if TYPE_CHECKING:
    from neat.config import Config

Population = Dict[int, DefaultGenome]

class UserEvaluatedFitness(BasicFitness):
    """Fitness function that calculates the fitness of each genome based on user data."""
    def __init__(self, config):
        super().__init__(config)
    
    def __call__(self, genome: DefaultGenome, max_alive_time: Optional[int] = 0):
        """
        Calculate the fitness of a genome.
        
        :param genome: The genome to evaluate.
        :param max_alive_time: The maximum time a genome can be alive.
        """
        self.calculate_fitness(genome, max_alive_time)

    def rating_and_time_alive_50_50(self, rating: int, time_alive: int) -> float: 
        """Calculate the fitness of a genome based 50/50 on the rating and time alive."""
        fitness = (rating + time_alive) / 2
        return fitness

    def calculate_fitness(self, genome: DefaultGenome, max_alive_time: int) -> None:
        """Calculate the fitness the genome based on user data."""
        data: UserData = genome.data

        if max_alive_time > 0: 
            alive_time = data.time_since_startup / alive_time
        else: 
            alive_time = data.time_since_startup

        rating = data.user_rating
        fitness = self.rating_and_time_alive_50_50(rating, alive_time)

        genome.fitness = fitness
