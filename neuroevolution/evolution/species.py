"""Contains the MixedGenerationSpecies class, which holds information about a species and its members. """
import random
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Dict

from neat.six_util import iteritems
from neuroevolution.evolution.genome_manager import GenomeManager

if TYPE_CHECKING:
    from neat.genome import DefaultGenome

Member = Tuple[int, 'DefaultGenome']
Members = List[Member]
Population = Dict[int, 'DefaultGenome']

class MixedGenerationSpecies:
    """Holds information about a species and its members."""
    def __init__(self, key: int, generation: int):
        """
        Initializes the species with the given key and generation.
        
        :param key: The unique ID of the species.
        :param generation: The generation number.
        """
        self.key = key
        self.created = generation
        self.adjusted_fitness = None
        self.fitness_history = []
    
    def set_adjusted_fitness(self, adjusted_fitness):
        """Sets the adjusted fitness of the species."""
        self.adjusted_fitness = adjusted_fitness
        self.update_fitness_history(adjusted_fitness)

    def update_fitness_history(self, fitness):
        """Updates the fitness history of the species."""
        self.fitness_history.append(fitness)

    def get_fitness_history(self) -> List[float]:
        """Returns the fitness history of the species."""
        return self.fitness_history
