"""Keeps track of whether species are making progress and helps remove ones that are not."""

import sys
from typing import List, Tuple, Dict, TYPE_CHECKING

from neat.config import ConfigParameter, DefaultClassConfig, Config
from neat.math_util import stat_functions

from neuroevolution.evolution.population_manager import PopulationManager
from neuroevolution.evolution.species import MixedGenerationSpecies

if TYPE_CHECKING: 
    from neuroevolution.evolution.species_set import MixedGenerationSpeciesSet
    from neuroevolution.lab.note_taker import NoteTaker

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

    def __init__(self, config: Config, population_manager: PopulationManager, reporter: 'NoteTaker'):
        # pylint: disable=super-init-not-called
        self.stagnation_config = config
        self.species_fitness_func = self.set_fitness_func(config)
        self.manager = population_manager
        self.reporter = reporter

    def set_fitness_func(self, config):
        """
        Returns the species fitness function based on the configuration.
        """
        func = stat_functions.get(config.species_fitness_func)
        if func is None:
            raise RuntimeError(f"Unexpected species fitness func: {config.species_fitness_func}")
        return func

    def update_active_species(self) -> Dict[int, bool]:
        """
        Updates species fitness history information,
        checks for species that have not improved in max_stagnation generations,
        and returns a list with stagnant species marked for removal.
        """
        for species in self.manager.species.get_all_species_objects():
            self.update_species_fitness(species)
        
        stagnant_map = self._identify_stagnant_species()
        self.remove_stagnant(stagnant_map)
        return stagnant_map
    
    def remove_stagnant(self, is_stagnant: Dict[int, bool]) -> None:
        """
        Marks stagnant species as inactive.

        :param is_stagnant: A dictionary mapping species IDs to boolean values indicating whether the species is stagnant.
        """
        for species_id in self.manager.species.get_all_species_ids():
            if is_stagnant[species_id]:
                self.manager.species.mark_stagnant(species_id)
        self.manager.species.remove_stagnant_species()
        
    def update_species_fitness(self, species: MixedGenerationSpecies):
        """
        Updates the fitness of the species.
        
        :param species: The species to update.
        :param evaluated_genome_ids: The evaluated genomes.
        """
        evaluated_fitnesses = self.manager.genomes.get_genome_fitnesses_for_species(species.key)
        if evaluated_fitnesses:
            fitness = self.species_fitness_func(evaluated_fitnesses)
        else: 
            fitness = -sys.float_info.max
        self.manager.species.update_species_fitness(species, fitness)
        species.fitness_history.append(fitness)
        species.adjusted_fitness = None

    def identify_elite_species_ids(self, fitnesses: List[Tuple[int, float]]) -> List[int]:
        """
        Returns the elite species.
        """
        elite_fitnesses = fitnesses[:self.stagnation_config.species_elitism]
        return [species_id for species_id, _ in elite_fitnesses]

    def _identify_stagnant_species(self) -> Dict[int, bool]:
        """
        Identifies stagnant species that have not improved in max_stagnation generations.
        But preserves elite species based on top fitness.
        :param fitnesses: A list of tuples containing the species ID and the species fitness.
        :return: A dictionary mapping species IDs to a boolean indicating whether the species is stagnant.
        """
        fitnesses = self.manager.species.get_species_ids_fitness_sorted()
        result = {}
        elite_ids = self.identify_elite_species_ids(fitnesses)
        for species_id, _ in fitnesses:
            if species_id in elite_ids:
                result[species_id] = False
            else: 
                is_stagnant = self.manager.species.is_stagnant(species_id, self.stagnation_config.max_stagnation)
                result[species_id] = is_stagnant
        return result
