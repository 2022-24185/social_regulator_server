"""Keeps track of whether species are making progress and helps remove ones that are not."""

import sys, logging
from typing import List, Tuple, Dict

from neat.config import ConfigParameter, DefaultClassConfig
from neat.math_util import stat_functions

from neuroevolution.evolution.mixed_generation_species import (
    MixedGenerationSpeciesSet,
    MixedGenerationSpecies,
)

SpeciesData = List[Tuple[int, MixedGenerationSpecies]]
StagnationResult = List[Tuple[int, MixedGenerationSpecies, bool]]

class TournamentStagnation(DefaultClassConfig):
    """Keeps track of whether species are making progress and helps remove ones that are not."""

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

    def __init__(self, config, reporters):
        # pylint: disable=super-init-not-called
        self.stagnation_config = config

        self.species_fitness_func = stat_functions.get(config.species_fitness_func)
        if self.species_fitness_func is None:
            raise RuntimeError(
                "Unexpected species fitness func: {0!r}".format(
                    config.species_fitness_func
                )
            )

        self.reporters = reporters

    def update(self,
        species_set: MixedGenerationSpeciesSet,
        genome_ids_to_consider: List[int],
        generation: int,
    ) -> StagnationResult:
        """
        Updates species fitness history information,
        checks for species that have not improved in max_stagnation generations,
        and returns a list with stagnant species marked for removal.
        """
        logging.info(f"Updating species fitness history for generation {generation} and {species_set.species}.")
        species_data = self._update_fitness_history_for_species(
            species_set, genome_ids_to_consider, generation)
        logging.info(f"Species data: {species_data}")
        species_data.sort(key=lambda x: x[1].fitness)
        result = self._identify_stagnant_species(species_data, generation)
        return result

    def _update_fitness_history_for_species(
        self,
        species_set: MixedGenerationSpeciesSet,
        genome_ids_to_consider: List[int],
        generation: int,
    ) -> SpeciesData:
        species_data = []
        for sid, species in species_set.species.items():
            prev_fitness = (
                max(species.fitness_history) if species.fitness_history else -sys.float_info.max
            )
            species.fitness = self.species_fitness_func(
                species.get_subset_of_fitnesses(genome_ids_to_consider)
            )
            species.fitness_history.append(species.fitness)
            species.adjusted_fitness = None
            if prev_fitness is None or species.fitness > prev_fitness:
                species.last_improved = generation
            species_data.append((sid, species))
        return species_data

    def _identify_stagnant_species(self, species_data: SpeciesData, generation: int) -> StagnationResult:
        result = []
        num_non_stagnant = len(species_data)
        logging.info(f"Identifying stagnant species for generation {generation}.")
        for i, (species_id, species) in enumerate(species_data):
            is_stagnant = False
            if num_non_stagnant > self.stagnation_config.species_elitism:
                is_stagnant = (
                    generation - species.last_improved
                    >= self.stagnation_config.max_stagnation
                )
                logging.info(f"Species {species_id} is stagnant: {is_stagnant} because generation - species.last_improved >= self.stagnation_config.max_stagnation.")
            if (len(species_data) - i) <= self.stagnation_config.species_elitism:
                is_stagnant = False
                logging.info(f"Species {species_id} is not stagnant because len(species_data) - i <= self.stagnation_config.species_elitism.")
            if is_stagnant:
                num_non_stagnant -= 1
            result.append((species_id, species, is_stagnant))
        return result
