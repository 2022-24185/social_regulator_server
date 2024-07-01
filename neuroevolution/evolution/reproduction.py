"""
Handles creation of genomes, either from scratch or by sexual or
asexual reproduction from parents.
"""

from itertools import count
from math import ceil
from typing import List, Dict, TYPE_CHECKING, Tuple

from neat.config import ConfigParameter, DefaultClassConfig
from neat.config import Config
from neat.genome import DefaultGenome

from neuroevolution.evolution.species import MixedGenerationSpecies
from neuroevolution.evolution.species_reproduction import SpeciesReproduction
from neuroevolution.evolution.offspring_generator import OffspringGenerator
from neuroevolution.evolution.elites import Elites

if TYPE_CHECKING: 
    from neuroevolution.evolution.stagnation import MixedGenerationStagnation

# Type aliases for better readability
Population = Dict[int, DefaultGenome]
Member = Tuple[int, DefaultGenome]
Members = List[Member]


class MixedGenerationReproduction(DefaultClassConfig):
    """
    Implements the default NEAT-python reproduction scheme:
    explicit fitness sharing with fixed-time species stagnation.
    """
    def __init__(self, config: Config, stagnation: "MixedGenerationStagnation"):
        # pylint: disable=super-init-not-called
        self.config = config
        self.reproduction_config = config.reproduction_config
        self.offspring_generator = self.create_offspring_generator(DefaultGenome, config.genome_config, config.reproduction_config)
        self.elites = Elites(self.reproduction_config.elitism)
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
    
    def create_offspring_generator(self, genome_type, genome_config, reprod_config) -> OffspringGenerator:
        """
        Creates a new offspring generator.
        """
        return OffspringGenerator(genome_type, genome_config, reprod_config)

    def create_new_genomes(self, num_genomes: int) -> Population:
        """
        Create a number of new genomes from scratch.
        
        :param num_genomes: The number of genomes to create.
        :return: A dictionary mapping genome key to genome.
        """
        new_genomes = self.offspring_generator.create_without_parents(num_genomes)
        for genome in new_genomes.values():
            self.ancestors[genome.key] = tuple()

        return new_genomes
    
    def create_offspring_for_species(self, species: MixedGenerationSpecies, sorted_parents: Members, dying_parents_count: int) -> Dict[int, "DefaultGenome"]:
        """
        Creates offspring for a given species.
        """
        if dying_parents_count > 0:
            reproduction_cutoff = max(int(ceil(self.reproduction_config.survival_threshold * len(sorted_parents))), self.reproduction_config.min_species_size)
            offspring = self.offspring_generator.create_offspring(sorted_parents, dying_parents_count, reproduction_cutoff)
            return offspring
        else:
            print("No offspring created for species %s", species.key)
            return {}

    def reproduce_evaluated(
        self,
        active_species: List[MixedGenerationSpecies],
        evaluated_genome_ids: List[int],
    ) -> Population:
        """
        Reproduce the given genomes into a new generation.

        :param active_species: The currently active species.
        :param selected_genome_ids: The genome ids of the genomes to be reproducted.
        :return: The new population.
        """
        species_reprod = SpeciesReproduction(active_species, evaluated_genome_ids, self.get_minimum_species_size(), self.config)
        offspring_count = species_reprod.compute_offspring_counts()
        new_population = {}
        for i, species in enumerate(active_species):
            sorted_parents = species.get_sorted_by_fitness(evaluated_genome_ids)
            elites = self.elites.preserve(sorted_parents, offspring_count[i])
            new_population.update(elites)
            num_dying = species_reprod.process_dying_parents(species, sorted_parents, elites)
            offspring = self.create_offspring_for_species(species, sorted_parents, num_dying)
            new_population.update(offspring)
        return new_population
        
    def get_minimum_species_size(self) -> int:
        """Get the minimum species size."""
        return max(
            self.reproduction_config.min_species_size, self.reproduction_config.elitism
        )

# main function for module
if __name__ == "__main__":
    pass
