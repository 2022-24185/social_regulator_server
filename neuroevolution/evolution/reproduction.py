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
from neat.math_util import mean

from neuroevolution.evolution.species import MixedGenerationSpecies
from neuroevolution.evolution.offspring_generator import OffspringGenerator
from neuroevolution.evolution.population_manager import PopulationManager
from neuroevolution.evolution.elites import Elites
from neuroevolution.evolution.species_metrics import SpeciesMetrics

# Type aliases for better readability
Population = Dict[int, DefaultGenome]
Member = Tuple[int, DefaultGenome]
Members = List[Member]


class MixedGenerationReproduction(DefaultClassConfig):
    """
    Implements the default NEAT-python reproduction scheme:
    explicit fitness sharing with fixed-time species stagnation.
    """
    def __init__(self, config: Config, population_manager: PopulationManager):
        # pylint: disable=super-init-not-called
        self.config = config
        self.reproduction_config = config.reproduction_config
        self.manager = population_manager
        self.offspring_generator = self.create_offspring_generator(DefaultGenome, config.genome_config, config.reproduction_config)
        self.elites = Elites(self.reproduction_config.elitism)
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
        self.manager.genomes.add_genomes(new_genomes)
        for genome in new_genomes.values():
            self.ancestors[genome.key] = tuple()
    
    def create_offspring_for_species(self, species: MixedGenerationSpecies) -> Dict[int, "DefaultGenome"]:
        """
        Creates offspring for a given species.
        """
        dying_count = len(self.manager.genomes.get_dying_genomes_for_species(species.key))
        sorted_parents = self.manager.genomes.get_genomes_sorted_by_fitness(species.key)
        print(f"💀 Species {species.key} dying count: {dying_count}")
        if dying_count > 0:
            reproduction_cutoff = SpeciesMetrics.get_reproduction_cutoff(
                self.reproduction_config.survival_threshold, 
                len(sorted_parents), 
                self.reproduction_config.min_species_size
                )
            offspring = self.offspring_generator.create_offspring(sorted_parents, dying_count, reproduction_cutoff)
            print(f"🐣 Species {species.key} offspring size: {len(offspring)}")
            self.manager.genomes.add_genomes(offspring)
            for genome in offspring.values():
                self.ancestors[genome.key] = tuple()
        else:
            pass
            #print("No offspring created for species %s", species.key)
        
    def set_elites(self, species_id) -> None: 
        expected_offspring = self.manager.species.get_expected_offspring(species_id)
        sorted_parents = self.manager.genomes.get_genomes_sorted_by_fitness(species_id)
        print(f"Expected offspring: {expected_offspring}")
        print(f"Sorted parents: {[p.key for p in sorted_parents]}")
        elites = self.elites.preserve(sorted_parents, expected_offspring)
        for elite in elites: 
            self.manager.genomes.set_elite(elite.key)
        print(f"Elite count: {len(self.manager.genomes.get_elite_genomes())}")

    def reproduce(self) -> Population:
        """
        Reproduce the given genomes into a new generation.

        :param active_species: The currently active species.
        :return: The new population.
        """
        active_species = self.manager.species.get_active_species()
        if not active_species:
            raise ValueError("No species to reproduce.")
        self.compute_offspring_counts()
        for species in active_species:
            self.set_elites(species.key)
            self.create_offspring_for_species(species)
            self.process_dying_parents(species.key)

    def process_dying_parents(self, species_id) -> int:
        """
        Identifies and processes dying parents.

        :param species: The species to process.
        :param sorted_parents: The list of parents sorted by fitness.
        :param elites: The elites.
        :return: The number of dying parents.
        """
        dying = self.manager.genomes.get_dying_genomes_for_species(species_id)
        for gid in dying: 
            self.manager.genomes.remove_genome(gid)
        return len(dying)
    
    def get_adjusted_genome_fitness(self):
        """
        Adjusts the fitnesses of the offspring.
        """
        all_fitnesses = self.manager.genomes.get_evaluated_genome_fitnesses()
        active = self.manager.species.get_active_species()
        for species in active: 
            fitnesses = self.manager.genomes.get_genome_fitnesses_for_species(species.key)
            species_fitnesses = [fitness for _, fitness in fitnesses]
            adjusted_fitness = SpeciesMetrics.get_adjusted_mean_fitness(all_fitnesses, species_fitnesses)
            species.set_adjusted_fitness(adjusted_fitness)
        return mean(species.adjusted_fitness for species in active)
    
    def compute_offspring_counts(self) -> List[int]:
        """
        Compute the number of offspring for each species.
        
        :return: A list of the number of offspring for each species.
        """
        aging = len(self.manager.genomes.get_evaluated_genomes())
        active_species = self.manager.species.get_active_species()
        min_size = self.get_minimum_species_size()
        print(f"👵 Total evaluated pop: {aging}")

        total_deficit = 0
        for species in active_species:
            exp_size = SpeciesMetrics.compute_expected_size(
                self.get_adjusted_genome_fitness(),
                species.adjusted_fitness,
                aging,
                min_size,
            )
            deficit = SpeciesMetrics.compute_species_deficit(exp_size, aging)
            print(f"🧠 Species {species.key} deficit: {deficit}")
            print(f"🧠 Species {species.key} expected size: {exp_size}")
            self.manager.species.set_population_deficit(species.key, deficit)
            total_deficit += deficit

        for species in active_species:
            expected = SpeciesMetrics.compute_offspring_count(
                self.manager.species.get_deficit(species.key),
                aging / total_deficit,
                min_size,
            )
            self.manager.species.set_expected_offspring(species.key, expected)
            print(f"🧠 Species {species.key} final offspring count: {expected}")
        
    def get_minimum_species_size(self) -> int:
        """Get the minimum species size."""
        return max(
            self.reproduction_config.min_species_size, self.reproduction_config.elitism
        )

# main function for module
if __name__ == "__main__":
    pass
