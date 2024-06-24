"""Divides the population into species based on genomic distances."""

from dataclasses import dataclass
from itertools import count
import logging
from neat.math_util import mean, stdev
from neat.six_util import iteritems, iterkeys, itervalues
from neat.config import ConfigParameter, DefaultClassConfig
from neat.genome import DefaultGenome
from typing import Dict, List, Set, Tuple, Optional, TYPE_CHECKING
from neat.config import Config

from neuroevolution.evolution.shared.speciation_utils import find_closest_element, get_compatible_elements, partition_elements

if TYPE_CHECKING:
    from neat.genome import DefaultGenome
    from neat.species import SpeciesSet, GenomeDistanceCache


class MixedGenerationSpecies:
    def __init__(self, key, generation):
        self.key = key
        self.created = generation
        self.last_improved = generation
        self.representative : DefaultGenome  = None
        self.members: Dict[int, DefaultGenome] = {}
        self.fitness = None
        self.adjusted_fitness = None
        self.fitness_history = []

    def update(self, representative, members):
        """Updates the species with the given representative and members."""
        self.representative = representative
        self.members = members

    def get_fitnesses(self):
        """Returns a list of the fitnesses of the members as long as their id is in genome_ids_to_consider."""
        return [m.fitness for m in itervalues(self.members)]

    def get_subset_of_fitnesses(self, genome_ids_to_consider: List[int]):
        """Returns a list of the fitnesses of the members as long as their id is in genome_ids_to_consider."""
        return [
            genome.fitness
            for genome_id, genome in self.members.items()
            if genome_id in genome_ids_to_consider
        ]

    def remove_members(self, dead_genomes: Set[int]):
        """Removes members that are no longer alive."""
        for genome_id in dead_genomes:
            if genome_id in self.members:
                del self.members[genome_id]


class GenomeDistanceCache(object):
    def __init__(self, config):
        self.distances = {}
        self.config = config
        self.hits = 0
        self.misses = 0

    def __call__(self, genome0: DefaultGenome, genome1: DefaultGenome):
        g0 = genome0.key
        g1 = genome1.key
        d = self.distances.get((g0, g1))
        if d is None:
            # Distance is not already computed.
            d = genome0.distance(genome1, self.config)
            self.distances[g0, g1] = d
            self.distances[g1, g0] = d
            self.misses += 1
        else:
            self.hits += 1

        return d
    

def species_factory(species_id, generation):
    return MixedGenerationSpecies(species_id, generation)

def distance_cache_factory(config):
    return GenomeDistanceCache(config)


class MixedGenerationSpeciesSet(DefaultClassConfig):
    """Encapsulates the default speciation scheme."""

    def __init__(self, config, reporters, species_factory, distance_cache_factory):
        # pylint: disable=super-init-not-called
        self.species_set_config: Config = config
        self.reporters = reporters
        self.indexer = count(1)
        self.species_factory = species_factory
        self.distance_cache_factory = distance_cache_factory
        self.species: Dict[int, MixedGenerationSpecies] = {}
        self.genome_to_species = {}
        self.distance_cache: GenomeDistanceCache = None
        self.compatibility_threshold = None
        self.population = None
        self.unspeciated = None

    @classmethod
    def parse_config(cls, param_dict):
        """Read the configuration parameters from the dictionary."""
        return DefaultClassConfig(
            param_dict, [ConfigParameter("compatibility_threshold", float)]
        )

    def set_new_population(self, population: Dict[int, DefaultGenome], reset_species: bool = True):
        """Initialize the species set with the given population. Resets species by default."""
        self.population = population
        self.unspeciated = set(population.keys())
        if reset_species:
            self.species.clear()
            self.genome_to_species.clear()

    def remove_dead_genomes(self, dead_genomes: Set[int]):
        """Remove dead genomes from the population and species."""
        for genome_id in dead_genomes:
            if genome_id in self.population:
                del self.population[genome_id]
            if genome_id in self.genome_to_species:
                species_id = self.genome_to_species[genome_id]
                self.species[species_id].remove_members(dead_genomes)
                del self.genome_to_species[genome_id]

    def batch_update_population(self, new_population: Dict[int, DefaultGenome]):
        """Update a subset of the population while keeping the rest intact."""
        for genome_id, genome in new_population.items():
            self.population[genome_id] = genome
        self.unspeciated.update(new_population.keys())

    def initialize_parameters(self, config: Config):
        """Initialize the parameters for the species set."""
        self.distance_cache = self.distance_cache_factory(config.genome_config)
        self.compatibility_threshold = self.species_set_config.compatibility_threshold

    def speciate(self, generation: int, config: Config):
        """Organize genomes into species based on genetic similarity."""
        new_representatives, new_members = self.find_new_representatives(self.species, self.unspeciated, self.population, self.distance_cache)
        new_groups, new_members = self.partition_population(self.unspeciated, self.population, new_representatives, new_members, self.distance_cache, self.compatibility_threshold)
        self.update_species_collection(new_representatives, new_members, generation, self.species_factory)
        self.log_species_statistics(self.distance_cache, self.reporters)

    # Pure Functions
    def find_closest_unspeciated_genome(self, representative: int, unspeciated: Set[int], population: Dict[int, DefaultGenome], distance_cache: GenomeDistanceCache) -> int:
        return find_closest_element(representative, unspeciated, lambda a, b: distance_cache(population[a], population[b]))

    def find_new_representatives(self, species: Dict[int, MixedGenerationSpecies], unspeciated: Set[int], population: Dict[int, DefaultGenome], distance_cache: GenomeDistanceCache) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        new_representatives, new_members = {}, {}
        for species_id, species in species.items():
            if not unspeciated:
                break
            closest_genome_id = self.find_closest_unspeciated_genome(species.representative.key, unspeciated, population, distance_cache)
            new_representatives[species_id] = closest_genome_id
            new_members[species_id] = [closest_genome_id]
            unspeciated.remove(closest_genome_id)
        return new_representatives, new_members

    def partition_population(self, unspeciated: Set[int], population: Dict[int, DefaultGenome], new_representatives: Dict[int, int], new_members: Dict[int, List[int]], distance_cache: GenomeDistanceCache, compatibility_threshold: float) -> Tuple[Set[int], Dict[int, List[int]]]:
        new_groups, new_memberships = partition_elements(unspeciated, population, new_representatives, new_members, lambda a, b: distance_cache(a, b) < compatibility_threshold)
        return new_groups, new_memberships

    # Mutating Methods
    def update_species_collection(self, new_representatives: Dict[int, int], new_members: Dict[int, List[int]], generation: int, species_factory):
        for species_id, rep_id in new_representatives.items():
            if not new_members.get(species_id):
                continue
            self.update_species(species_id, rep_id, new_members[species_id], generation, species_factory)

    def update_species(self, species_id: int, rep_id: int, member_ids: List[int], generation: int, species_factory):
        self.unspeciated.difference_update(member_ids)
        species = self.species.get(species_id, species_factory(species_id, generation))
        members = {genome_id: self.population[genome_id] for genome_id in member_ids}
        species.update(self.population[rep_id], members)
        self.species[species_id] = species
        for genome_id in member_ids:
            self.genome_to_species[genome_id] = species_id

    def log_species_statistics(self, distance_cache: GenomeDistanceCache, reporters):
        distances = list(distance_cache.distances.values())
        if distances:
            reporters.info(
                f"Mean genetic distance {mean(distances):.3f}, "
                f"standard deviation {stdev(distances):.3f}"
            )

    def get_species_id(self, individual_id):
        """Returns the species id of the individual with the given id."""
        return self.genome_to_species[individual_id]

    def get_species(self, individual_id):
        """Returns the species of the individual with the given id."""
        sid = self.genome_to_species[individual_id]
        return self.species[sid]
