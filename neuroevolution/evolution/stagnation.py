"""Keeps track of whether species are making progress and helps remove ones that are not."""

import sys
from typing import List, Tuple, Dict

from neat.config import ConfigParameter, DefaultClassConfig, Config
from neat.math_util import stat_functions

from neuroevolution.evolution.species_set import MixedGenerationSpeciesSet
from neuroevolution.evolution.species import MixedGenerationSpecies

SpeciesData = List[Tuple[int, MixedGenerationSpecies]]
StagnationResult = List[Tuple[int, MixedGenerationSpecies, bool]]

class MixedGenerationStagnation(DefaultClassConfig):
    """
    Keeps track of whether species are making progress and helps remove ones that are not.
    """

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(
            param_dict,
            [
                ConfigParameter("species_fitness_func", str, "mean"),
                ConfigParameter("max_stagnation", int, 15),
                ConfigParameter("species_elitism", int, 0),
            ],
        )

    def __init__(self, config: Config, reporters):
        # pylint: disable=super-init-not-called
        self.stagnation_config = config
        self.species_fitness_func = self.set_fitness_func(config)
        self.reporters = reporters

    def set_fitness_func(self, config):
        """
        Returns the species fitness function based on the configuration.
        """
        func = stat_functions.get(config.species_fitness_func)
        if func is None:
            raise RuntimeError(f"Unexpected species fitness func: {config.species_fitness_func}")
        return func

    def update(self,
        species_set: MixedGenerationSpeciesSet,
        genome_ids_to_consider: List[int],
        generation: int,
    ) -> Dict[int, bool]:
        """
        Updates species fitness history information,
        checks for species that have not improved in max_stagnation generations,
        and returns a list with stagnant species marked for removal.
        """
        species_data = self._update_fitness_history_for_species(
            species_set, genome_ids_to_consider, generation)
        species_data.sort(key=lambda x: x[1].fitness)
        result = self._identify_stagnant_species(species_data, generation)
        return result
    
    def calculate_prev_fitness(self, species: MixedGenerationSpecies) -> float:
        """
        Returns the previous fitness of the species.

        :param species: The species to get the previous fitness for.
        :return: The previous fitness of the species.
        """
        return max(species.fitness_history) if species.fitness_history else -sys.float_info.max
    
    def update_species_fitness(self, species: MixedGenerationSpecies, evaluated_genome_ids):
        """
        Updates the fitness of the species.
        
        :param species: The species to update.
        :param evaluated_genome_ids: The evaluated genomes.
        """
        species.fitness = self.species_fitness_func(
            species.get_fitnesses(evaluated_genome_ids)
        )
        print(f"appending {species.fitness}")
        species.fitness_history.append(species.fitness)
        species.adjusted_fitness = None

    def _update_fitness_history_for_species(
        self,
        species_set: MixedGenerationSpeciesSet,
        evaluated_genome_ids: List[int],
        generation: int,
    ) -> SpeciesData:
        """
        Updates the fitness history for all species in the species set.
        
        :param species_set: The species set to update.
        :param evaluated_genome_ids: The evaluated genomes.
        :param generation: The current generation.
        :return: A list of tuples containing the species ID and the species instance.
        """
        species_data = []
        for sid, species in species_set.species.items():
            prev_fitness = self.calculate_prev_fitness(species)
            self.update_species_fitness(species, evaluated_genome_ids)
            print(f"prev_fitness: {prev_fitness}, species.fitness: {species.fitness}")
            if prev_fitness is None or species.fitness > prev_fitness:
                species.last_improved = generation
            species_data.append((sid, species))
        return species_data
    
    def sort_by_fitness(self, species_data: SpeciesData) -> SpeciesData:
        """
        Sorts the species by fitness in descending order.
        
        :param species_data: A list of tuples containing the species ID and the species instance.
        :return: A list of tuples containing the species ID and the species instance, sorted by fitness.
        """
        sorted_data = species_data.copy()
        sorted_data.sort(key=lambda x: x[1].fitness, reverse=True)
        return sorted_data
    
    def _is_species_stagnant(self, species: MixedGenerationSpecies, generation: int, index: int, num_non_stagnant: int) -> bool:
        """
        Determines whether a species is stagnant.
        
        :param species: The species to check.
        :param generation: The current generation.
        :param index: The index of the species in the list of all species.
        :param num_non_stagnant: The number of non-stagnant species.
        :return: True if the species is stagnant, False otherwise.
        """
        # species elitism protects n species from stagnation
        stagnated = True
        if generation - species.last_improved <= self.stagnation_config.max_stagnation: 
            stagnated = False
        if (num_non_stagnant - index) < self.stagnation_config.species_elitism:
            stagnated = False
        return stagnated

    def _identify_stagnant_species(self, species_data: SpeciesData, generation: int) -> Dict[int, bool]:
        """
        Identifies stagnant species that have not improved in max_stagnation generations.
        
        :param species_data: A list of tuples containing the species ID and the species instance.
        :param generation: The current generation.
        :return: A dictionary mapping species IDs to a boolean indicating whether the species is stagnant.
        """
        result = {}
        num_non_stagnant = len(species_data)
        sorted_data = self.sort_by_fitness(species_data) # to fascilitate species elitism
        for i, (species_id, species) in enumerate(sorted_data):
            is_stagnant = self._is_species_stagnant(species, generation, i, num_non_stagnant)
            print(f"is_stagnant: {is_stagnant}, {species_id}")
            if is_stagnant:
                num_non_stagnant -= 1
            result.update({species_id: is_stagnant})
        return result
