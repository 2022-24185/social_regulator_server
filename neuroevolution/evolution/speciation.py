"""Divides the population into species based on genomic distances."""
import logging
import random
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple, Set

from neat.config import Config, ConfigParameter, DefaultClassConfig

from neuroevolution.evolution.genome_distance_cache import GenomeDistanceCache
from neuroevolution.shared.speciation_utils import find_closest_element
from neuroevolution.evolution.species_set import MixedGenerationSpeciesSet
from neuroevolution.evolution.population_manager import PopulationManager

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
        return DefaultClassConfig(
            param_dict,
            [
                ConfigParameter("compatibility_threshold", float)
            ],
        )

    def __init__(self, config: Config, population_manager: PopulationManager) -> None:
        # pylint: disable=super-init-not-called
        """
        Initializes the Speciation instance.

        :param config: The configuration parameters for the speciation process.
        """
        self.compatibility_threshold = config.species_set_config.compatibility_threshold
        self.manager = population_manager
        self.distance_cache = self.create_distance_cache(config.genome_config)

    def create_distance_cache(self, config) -> GenomeDistanceCache:
        """
        Creates a new GenomeDistanceCache instance.
        """
        return GenomeDistanceCache(config)
    
    def speciate(self) -> MixedGenerationSpeciesSet:
        """
        Organizes genomes into species based on their genetic distances.

        :param population: A dictionary mapping genome IDs to genomes.
        :param current_generation: The current generation number.
        """
        self.set_new_representatives()
        #print(f"pop_legth: {len(self.manager.genomes.get_all_genomes())}")
        #print(f"Representatives: {self.manager.genomes.representative_genomes}")
        self.partition_population()
        print(f"Species: {self.manager.species.get_all_species_ids()}")

    def extract_new_representative(self, rep: 'DefaultGenome') -> 'DefaultGenome':
        """
        Extracts a new representative from the unspeciated genomes.
        
        :param distance_fn: A function to calculate the distance between two genomes.
        :param rep_id: The ID of the current representative.
        :return: The new representative.
        """
        def distance_fn(genome_a, genome_b):
            return self.distance_cache(genome_a, genome_b)
        
        unspeciated = self.manager.genomes.get_unspeciated_genomes()
        if not unspeciated: 
            return None
        if not rep:
            new_rep = random.choice(list(unspeciated))
        else:
            new_rep = find_closest_element(rep, unspeciated, distance_fn)
        return new_rep

    def set_new_representatives(self) -> None:
        """
        Selects new representatives for each species from the unspeciated genomes and 
        moves them to their respective species.

        :param population: A dictionary mapping genome IDs to genomes.
        """
        for species_id, _ in self.manager.species.get_all_species():
            try:
                rep_genome = self.manager.genomes.get_representative(species_id)
                new_rep = self.extract_new_representative(rep_genome)
                if new_rep is None:
                    continue
                self.manager.genomes.assign_genome_to_species(new_rep.key, species_id)
                self.manager.genomes.set_representative(species_id, new_rep.key)
            except Exception as e:
                logging.error("Error while setting new representatives for species %s: %s", species_id, e)
                raise e
            
    def get_compatible_genomes(self, genome: 'DefaultGenome') -> List[Tuple[float, int]]:
        """
        Returns a list of (compatibility, group_id) tuples for each group in groups.
        
        :param genome_id: The ID of the genome to compare.
        :return: A list of (compatibility, group_id) tuples.
        """
        def how_compatible(ga, gb) -> float:
            if self.compatibility_threshold - self.distance_cache(ga, gb) < 0:
                return 0.0
            else:
                return self.distance_cache(ga, gb)
            
        comp_list = []
        for species in self.manager.species.get_all_species_objects():
            #print(f"species: {species.key}")
            rep = self.manager.genomes.get_representative(species.key)
            #print(f"rep: {rep.key}, genome: {genome.key}")
            compatibility = how_compatible(rep, genome)
            if compatibility:
                comp_list.append((compatibility, species.key))
        return comp_list

    def partition_population(self) -> None:
        """
        Partitions the unspeciated genomes into existing or new species based on 
        genetic distances.

        :param population: A dictionary mapping genome IDs to genomes.
        :param generation: The current generation number.
        """
        for genome in self.manager.genomes.get_unspeciated_genomes():
            compatibilities = self.get_compatible_genomes(genome)
            try:
                if compatibilities:
                    _, best_species_id = min(compatibilities, key=lambda x: x[0])
                    self.manager.genomes.assign_genome_to_species(genome.key, best_species_id)
                else:
                    new_species_id = self.manager.species.create_new_species()
                    self.manager.genomes.assign_genome_to_species(genome.key, new_species_id)
                    self.manager.genomes.set_representative(new_species_id, genome.key)
            except Exception as e:
                logging.error("Error partitioning genome %s: %s", genome.key, e)
                raise e
        #print(f"species members are {[(species_id, species_instance.members.keys()) for species_id, species_instance in self.species_set.get_all_species()]}")
