"""Implements the core evolution algorithm."""
import random
from typing import Dict, List, TYPE_CHECKING

from neat.genome import DefaultGenome

from neuroevolution.evolution.species_set import MixedGenerationSpeciesSet
from neuroevolution.evolution.genome_manager import GenomeManager

if TYPE_CHECKING:
    from neuroevolution.server.models import UserData

# Type aliases for better readability
Population = Dict[int, DefaultGenome]

class CompleteExtinctionException(Exception):
    """Exception to raise when a population has no members."""
    def __init__(self, message):
        super().__init__(message)

class PopulationManager:
    """Manages the members of the population"""

    def __init__(self):
        self.generation = 0
        self.genomes = self.create_genome_manager()
        self.species = self.create_species_manager()
        self.reporter = None

    def create_genome_manager(self) -> GenomeManager:
        """
        Create a new genome manager.

        :return: The new genome manager.
        """
        return GenomeManager()
    
    def create_species_manager(self) -> MixedGenerationSpeciesSet:
        """
        Create a new species manager.

        :param config: The configuration.
        :return: The new species manager.
        """
        return MixedGenerationSpeciesSet()
    
    def reset(self):
        """
        Reset the population manager to its initial state.
        """
        self.generation = 0
        self.genomes.reset()
        self.species.reset()
    
    def update_generation(self) -> int:
        """
        Incorporate offspring into the population and update generation count.
        
        :param offspring: The offspring to add to the population.
        """
        self.generation += 1
        self.species.set_generation(self.generation)
        return self.generation
    
    def add_reporter(self, reporter):
        """
        Add a reporter to the population manager.

        :param reporter: The reporter to add.
        """
        self.reporter = reporter
        self.genomes.add_reporter(reporter)
                    
    def get_random_available_genome(self) -> DefaultGenome:
        """
        Send a random member to the user.
        
        :return: A random genome from the available genomes.
        """
        available = self.genomes.get_available_genomes()
        if not available:
            raise RuntimeError("No more genomes to send.")
        genome = random.choice(available)
        self.genomes.set_unavailable(genome.key)
        return genome

