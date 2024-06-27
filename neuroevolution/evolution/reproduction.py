"""
Handles creation of genomes, either from scratch or by sexual or
asexual reproduction from parents.
"""

from itertools import count
from typing import List, Dict, TYPE_CHECKING

from neat.config import ConfigParameter, DefaultClassConfig
from neat.config import Config
from neat.genome import DefaultGenome

from neuroevolution.evolution.species import MixedGenerationSpecies
from neuroevolution.evolution.species_reproduction import SpeciesReproduction
from neuroevolution.evolution.offspring_generator import OffspringGenerator

if TYPE_CHECKING: 
    from neuroevolution.evolution.stagnation import MixedGenerationStagnation

# Type aliases for better readability
Population = Dict[int, DefaultGenome]

class GenomeFactory:
    """Creates new genomes either from scratch or from parents."""
    def __init__(self, genome_type, genome_config):
        self.genome_type = genome_type
        self.genome_config = genome_config

    def create_genome(self, key):
        """Create a new genome with the given key."""
        genome = self.genome_type(key)
        genome.configure_new(self.genome_config)
        return genome


class MixedGenerationReproduction(DefaultClassConfig):
    """
    Implements the default NEAT-python reproduction scheme:
    explicit fitness sharing with fixed-time species stagnation.
    """
    def __init__(self, config: Config, stagnation: "MixedGenerationStagnation"):
        # pylint: disable=super-init-not-called
        self.config = config.reproduction_config
        self.genome_factory = GenomeFactory(DefaultGenome, config.genome_config)
        self.offspring_generator = OffspringGenerator(config)
        self.stagnation = stagnation
        self.genome_indexer = count(1)
        self.ancestors = {}
    
    @classmethod
    def parse_config(cls, param_dict):
        """Parse the configuration parameters."""
        return DefaultClassConfig(
            param_dict,
            [
                ConfigParameter("elitism", int, 0),
                ConfigParameter("survival_threshold", float, 0.2),
                ConfigParameter("min_species_size", int, 2),
            ],
        )

    def create_new_genomes(self, num_genomes: int) -> Population:
        """Create a number of new genomes from scratch."""
        new_genomes = {}
        for _ in range(num_genomes):
            key = next(self.genome_indexer)
            g = self.genome_factory.create_genome(key)
            new_genomes[key] = g
            self.ancestors[key] = tuple()
        return new_genomes

    def reproduce_evaluated(
        self,
        active_species: List[MixedGenerationSpecies],
        selected_genome_ids: List[int],
    ) -> Population:
        """
        Handles the reproduction of genomes from a selected subset of species,
        involving both creation of new genomes and reproduction from existing genomes.
        """
        species_reproduction = SpeciesReproduction(active_species, selected_genome_ids, self.get_minimum_species_size(), self.config)
        new_population = species_reproduction.reproduce()
        return new_population
        
    def get_minimum_species_size(self) -> int:
        """Get the minimum species size."""
        return max(
            self.config.min_species_size, self.config.elitism
        )

# main function for module
if __name__ == "__main__":
    pass
