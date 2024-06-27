"""Module for handling elitism in the evolution process."""

from typing import List, Tuple, Dict
from neat.genome import DefaultGenome

Member = Tuple[int, DefaultGenome]
Members = List[Member]

class Elites:
    """Module for handling elitism in the evolution process."""
    def __init__(self, elitism_config) -> None:
        """
        Initializes the Elites instance.
        
        :param elitism_config: The number of elites to preserve in the population.
        """
        self.elitism = elitism_config
        self.elitism_count = 0
        self.non_elites = 0

    def set_elitism_stats(self, offspring_count: int):
        """
        Sets the elitism count and the number of non-elites in the population.
        
        :param offspring_count: The number of offspring to generate.
        """
        self.elitism_count = max(offspring_count, self.elitism)
        self.non_elites = max(0, offspring_count - self.elitism_count)

    def preserve(self, sorted_parents: Members, offspring_count: int) -> Dict[int, DefaultGenome]:
        """
        Handle elitism by preserving the best members of the old population.
        
        :param sorted_parents: A list of parent genomes sorted by fitness.
        :param offspring_count: The number of offspring to generate.
        """
        self.set_elitism_stats(offspring_count)
        elites = sorted_parents[:self.elitism_count]
        new_population = {}
        for gid, elite in elites:
            new_population[gid] = elite
        return new_population