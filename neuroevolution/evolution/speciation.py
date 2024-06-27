"""Divides the population into species based on genomic distances."""
import logging
import random
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple, Set

from neat.config import Config, ConfigParameter, DefaultClassConfig

from neuroevolution.evolution.genome_distance_cache import GenomeDistanceCache
from neuroevolution.evolution.shared.speciation_utils import \
    find_closest_element
from neuroevolution.evolution.species_set import MixedGenerationSpeciesSet

if TYPE_CHECKING:
    from neat.genome import DefaultGenome

Member = Tuple[int, "DefaultGenome"]
Members = List[Member]


class Speciation(DefaultClassConfig):
    """
    Handles the speciation process within a genetic algorithm, dividing a 
    population of genomes into species based on genetic distances. This allows 
    for more diverse evolutionary paths and helps preserve innovations.
    """

    @classmethod
    def parse_config(cls, param_dict):
        """Parse the configuration parameters."""
        print(f"PARSIIIIIING, {param_dict}")
        return DefaultClassConfig(
            param_dict,
            [
                ConfigParameter("compatibility_threshold", float)
            ],
        )

    def __init__(self, config: Config) -> None:
        # pylint: disable=super-init-not-called
        """
        Initializes the Speciation instance.

        :param config: The configuration parameters for the speciation process.
        """
        self.compatibility_threshold = config.species_set_config.compatibility_threshold
        self.distance_cache = self.create_distance_cache(config)
        self.species_set = self.create_species_set()


    def create_distance_cache(self, config) -> GenomeDistanceCache:
        """
        Creates a new GenomeDistanceCache instance.
        """
        return GenomeDistanceCache(config)
    
    def create_species_set(self) -> MixedGenerationSpeciesSet:
        """
        Creates a new species set instance.
        """
        return MixedGenerationSpeciesSet()

    def speciate(
        self, population: Dict[int, "DefaultGenome"], current_generation: int
    ) -> MixedGenerationSpeciesSet:
        """
        Organizes genomes into species based on their genetic distances.

        :param population: A dictionary mapping genome IDs to genomes.
        :param current_generation: The current generation number.
        """
        self.set_new_representatives(population)
        self.partition_population(population, current_generation)
        return self.species_set

    def update_stagnant_species(self, is_stagnant: Dict[int, bool]) -> None:
        """
        Marks stagnant species as inactive.

        :param is_stagnant: A dictionary mapping species IDs to boolean values indicating whether the species is stagnant.
        """
        for species_id in self.species_set.get_all_species_ids():
            if is_stagnant[species_id]:
                self.species_set.mark_stagnant(species_id)

    def extract_new_representative(self, unspeciated: Set[int], distance_fn: Callable, rep_id: int = None) -> Tuple[int, List[int]]:
        """
        Extracts a new representative from the unspeciated genomes.
        
        :param unspeciated: A list of unspeciated genomes.
        :param distance_fn: A function to calculate the distance between two genomes.
        :param rep_id: The ID of the current representative.
        :return: A tuple containing the new representative and the updated list of unspeciated genomes.
        """
        if not rep_id:
            new_rep = random.choice(list(unspeciated))
        else:
            new_rep = find_closest_element(rep_id, unspeciated, distance_fn)
        unspeciated.remove(new_rep)
        return new_rep, unspeciated

    def set_new_representatives(self, population: Dict[int, "DefaultGenome"]) -> None:
        """
        Selects new representatives for each species from the unspeciated genomes and 
        moves them to their respective species.

        :param population: A dictionary mapping genome IDs to genomes.
        """
        unspeciated = self.species_set.get_unspeciated(population)
        if not unspeciated:
            print("No unspeciated genomes found.")
            return

        def distance_fn(id_a, id_b):
            return self.distance_cache(population[id_a], population[id_b])
        
        for species_id, species_instance in self.species_set.get_all_species():
            try:
                rep_id = species_instance.get_representative_id()
                new_rep, unspeciated = self.extract_new_representative(
                    unspeciated, distance_fn, rep_id
                )
                self.species_set.update_species_representative(species_id, new_rep, population[new_rep])
            except Exception as e:
                logging.error("Error while setting new representatives for species %s: %s", species_id, e)
                continue


    def partition_population(self, population: Dict[int, "DefaultGenome"], generation: int) -> None:
        """
        Partitions the unspeciated genomes into existing or new species based on 
        genetic distances.

        :param population: A dictionary mapping genome IDs to genomes.
        :param generation: The current generation number.
        """
        def is_compatible(ga, gb):
            return self.distance_cache(ga, gb) < self.compatibility_threshold
        
        for gid in self.species_set.get_unspeciated(population):
            candidates = self.species_set.get_compatible_genomes(
                self.species_set.get_all_species_ids(), gid, population, is_compatible)
            try:
                if candidates:
                    _, best_group_id = min(candidates, key=lambda x: x[0])
                    self.species_set.add_member(best_group_id, (gid, population[gid]))
                else:
                    new_species_id = self.species_set.create_new_species(generation)
                    self.species_set.add_member(new_species_id, (gid, population[gid]))
            except Exception as e:
                logging.error("Error partitioning genome %s: %s", gid, e)
