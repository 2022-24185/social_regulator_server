from typing import List, Tuple, Dict, TYPE_CHECKING
import logging
from neat.math_util import mean
from neuroevolution.evolution.species import MixedGenerationSpecies
from neuroevolution.evolution.fitness_manager import FitnessManager
from neuroevolution.evolution.elites import Elites
from neuroevolution.evolution.offspring_generator import OffspringGenerator

if TYPE_CHECKING:
    from neat.genome import DefaultGenome

Member = Tuple[int, 'DefaultGenome']
Members = List[Member]

class SpeciesReproduction: 
    """Handles reproduction logic for species"""
    def __init__(self, active_species, selected_genome_ids, min_species_size, config) -> None:
        self.config = config
        self.min_species_size = min_species_size
        self.active_species: List[MixedGenerationSpecies] = active_species
        self.evaluated_genome_ids = selected_genome_ids
        self.total_adjusted_fitness: float = 0.0
        self.total_death_count = 0
        self.fitness_collector = self.create_fitness_manager()
        self.offspring_generator = self.create_offspring_generator()
        self.elites = Elites(config.elitism)

    def create_fitness_manager(self):
        """
        Creates a new fitness manager.
        """
        return FitnessManager()
    
    def create_offspring_generator(self):
        """
        Creates a new offspring generator.
        """
        return OffspringGenerator(self.config)

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
    
    def create_offspring_for_species(self, species: MixedGenerationSpecies, sorted_parents: Members, dying_parents_count: int) -> Dict[int, "DefaultGenome"]:
        """
        Creates offspring for a given species.
        """
        if dying_parents_count > 0:
            return self.offspring_generator.create_offspring(sorted_parents, dying_parents_count)
        else:
            logging.info(f"No offspring created for species {species.key}")
            return {}
    
    def process_dying_parents(self, species: MixedGenerationSpecies, sorted_parents: Members, elites: Dict[int, 'DefaultGenome']) -> int:
        """
        Identifies and processes dying parents.

        Returns the number of dying parents.
        """
        dying_parents = set(sorted_parents[0]) - set(elites.keys())
        species.kill_members(dying_parents)
        return len(dying_parents)
    
    def adjust_fitnesses(self):
        """
        Adjusts the fitnesses of the offspring.
        """
        self.fitness_collector.adjust_fitnesses(self.active_species, self.evaluated_genome_ids)

    def reproduce(self) -> Dict[int, "DefaultGenome"]: 
        """
        Reproduces the genomes in the active species and returns a Dict of all offspring.
        """
        offspring_count = self.compute_offspring_counts()
        new_population = {}
        for species in self.active_species:
            sorted_parents = species.get_sorted_by_fitness(self.evaluated_genome_ids)
            elites = self.elites.preserve(sorted_parents, offspring_count)
            new_population.update(elites)
            num_dying = self.process_dying_parents(species, sorted_parents, elites)
            offspring = self.create_offspring_for_species(species, sorted_parents, num_dying)
            new_population.update(offspring)
        return new_population

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
    
