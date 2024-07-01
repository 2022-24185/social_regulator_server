"""Handles reproduction logic for species"""
import logging

from typing import List, Tuple, Dict, TYPE_CHECKING
from neat.math_util import mean
from neuroevolution.evolution.species import MixedGenerationSpecies
from neuroevolution.evolution.fitness_manager import FitnessManager

if TYPE_CHECKING:
    from neat.genome import DefaultGenome

Member = Tuple[int, 'DefaultGenome']
Members = List[Member]

class SpeciesReproduction: 
    """Handles reproduction logic for species"""
    def __init__(self, active_species, selected_genome_ids, min_species_size, reproduction_config) -> None:
        self.reprod_config = reproduction_config
        self.min_species_size = min_species_size
        self.active_species: List[MixedGenerationSpecies] = active_species
        self.evaluated_genome_ids = selected_genome_ids
        self.total_adjusted_fitness: float = 0.0
        self.total_death_count = 0
        self.fitness_collector = self.create_fitness_manager()

    def create_fitness_manager(self):
        """
        Creates a new fitness manager.
        """
        return FitnessManager()
    
    def get_total_adjusted_fitness(self): 
        """
        Returns the total adjusted fitness of all species.
        """
        if not self.active_species:
            return 0.0
        return mean(species.adjusted_fitness for species in self.active_species)

    def get_total_death_counts(self): 
        """
        Returns the total number of dying members across all species.
        """
        return sum(species.dying_count for species in self.active_species)

    def get_evaluated_genome_ids(self):
        """
        Returns the list of genome IDs that have been evaluated.
        """
        return self.evaluated_genome_ids

    def process_dying_parents(self, species: MixedGenerationSpecies, sorted_parents: Members, elites: Dict[int, 'DefaultGenome']) -> int:
        """
        Identifies and processes dying parents.

        :param species: The species to process.
        :param sorted_parents: The list of parents sorted by fitness.
        :param elites: The elites.
        :return: The number of dying parents.
        """
        dying_parents = set([member[0] for member in sorted_parents]) - set(elites.keys())
        species.kill_members(dying_parents)
        return len(dying_parents)

    def adjust_fitnesses(self):
        """
        Adjusts the fitnesses of the offspring.
        """
        self.fitness_collector.adjust_fitnesses(self.active_species, self.evaluated_genome_ids)

    def normalize_spawn_counts(self, total_dying_pop, deficit_per_species) -> List[int]:
        """
        Normalize the spawn amounts so that the next generation is roughly
        the population size requested by the user.
        
        :param total_dying_pop: The total number of dying members.
        :param deficit_per_species: The deficit of each species.
        :return: A list of the number of offspring for each species.
        """
        total_deficit = sum(deficit_per_species)
        if total_deficit == 0:
            return [self.min_species_size for _ in deficit_per_species]
        norm = total_dying_pop / total_deficit
        # Calculate the final number of offspring each species should produce
        final_offspring_counts = []
        for deficit in deficit_per_species:
            normalized_deficit = int(round(deficit * norm))
            final_offspring_count = max(self.min_species_size, normalized_deficit)
            final_offspring_counts.append(final_offspring_count)

        return final_offspring_counts

    def compute_offspring_counts(self) -> List[int]:
        """
        Compute the number of offspring for each species.
        
        :return: A list of the number of offspring for each species.
        """
        self.adjust_fitnesses()
        taf = self.get_total_adjusted_fitness()
        tdp = self.get_total_death_counts()
        population_deficits = []
        for species in self.active_species: 
            exp_size = species.compute_expected_size(self.min_species_size, taf, tdp)
            population_deficits.append(species.compute_pop_deficit(exp_size))
        return self.normalize_spawn_counts(tdp, population_deficits)
    
