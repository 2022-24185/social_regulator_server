"""Divides the population into species based on genomic distances."""

from itertools import count
import logging
from neat.math_util import mean, stdev
from neat.six_util import iteritems, iterkeys, itervalues
from neat.config import ConfigParameter, DefaultClassConfig
from neat.genome import DefaultGenome
from typing import Dict, List, Set, Tuple, Optional, TYPE_CHECKING
from neat.config import Config

if TYPE_CHECKING:
    from neat.genome import DefaultGenome
    from neat.species import SpeciesSet, GenomeDistanceCache


class TournamentSpecies:
    def __init__(self, key, generation):
        self.key = key
        self.created = generation
        self.last_improved = generation
        self.representative = None
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


class GenomeDistanceCache(object):
    def __init__(self, config):
        self.distances = {}
        self.config = config
        self.hits = 0
        self.misses = 0

    def __call__(self, genome0, genome1):
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
    return TournamentSpecies(species_id, generation)

def distance_cache_factory(config):
    return GenomeDistanceCache(config)


class TournamentSpeciesSet(DefaultClassConfig):
    """Encapsulates the default speciation scheme."""

    def __init__(self, config, reporters, species_factory, distance_cache_factory):
        # pylint: disable=super-init-not-called
        self.species_set_config : Config = config
        self.reporters = reporters
        self.indexer = count(1)
        self.species_factory = species_factory
        self.distance_cache_factory = distance_cache_factory
        self.species: Dict[int, TournamentSpecies] = {}
        self.genome_to_species = {}
        self.distance_cache : GenomeDistanceCache = None
        self.compatibility_threshold = None
        self.population = None
        self.unspeciated = None

    @classmethod
    def parse_config(cls, param_dict):
        """Read the configuration parameters from the dictionary."""
        return DefaultClassConfig(
            param_dict, [ConfigParameter("compatibility_threshold", float)]
        )
    
    def set_new_population(self, population: Dict[int, DefaultGenome]):
        """Initialize the species set with the given population."""
        self.population = population
        self.unspeciated = set(population.keys())

    def initialize_parameters(self, config: Config):
        """Initialize the parameters for the species set."""
        self.distance_cache = self.distance_cache_factory(config.genome_config)
        self.compatibility_threshold = self.species_set_config.compatibility_threshold

    def speciate(self, generation: int, config: Config):
        """ Organize genomes into species based on genetic similarity. """
        logging.info(f"Speciating generation {generation}")
        new_representatives, new_members = self._find_new_representatives()
        self._partition_population(new_representatives, new_members)
        self._update_species_collection(new_representatives, new_members, generation)
        self._log_species_statistics()

    def _find_closest_unspeciated_genome(self, representative: int) -> int:
        """ Find the unspeciated genome that is closest to the representative for the species. """
        candidates = [(self.distance_cache(representative, self.population[gid]), gid) for gid in self.unspeciated]
        _, closest_genome_id = min(candidates, key=lambda x: x[0])

        return closest_genome_id

    def _find_new_representatives(self) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        """ Determine new representatives based on minimum genetic distance to unspeciated genomes. """
        logging.info("Finding new representatives")
        new_representatives = {}
        new_members = {}
        for species_id, species in self.species.items():
            if not self.unspeciated:
                break
            closest_genome_id = self._find_closest_unspeciated_genome(representative=species.representative)
            new_representatives[species_id] = closest_genome_id
            new_members[species_id] = [closest_genome_id]
            self.unspeciated.remove(closest_genome_id)

        return new_representatives, new_members
    
    def _partition_population(self, new_representatives: Dict[int, int], 
                              new_members: Dict[int, List[int]]) -> None:
        """ Group remaining genomes into existing or new species based on compatibility. """
        logging.info(f"Partitioning population into species. Unspeciated: {len(self.unspeciated)}")

        while self.unspeciated:
            genome_id = self.unspeciated.pop()
            genome = self.population[genome_id]
            candidates = [(self.distance_cache(self.population[representative_id], genome), species_id)
                          for species_id, representative_id in new_representatives.items()
                          if self.distance_cache(self.population[representative_id], genome) < self.compatibility_threshold]
            if candidates:
                logging.debug(f"Adding genome {genome_id} to existing species")
                _, best_species_id = min(candidates, key=lambda x: x[0])
                new_members[best_species_id].append(genome_id)
            else:
                logging.debug(f"Creating new species for genome {genome_id}")
                species_id = next(self.indexer)
                new_representatives[species_id] = genome_id
                new_members[species_id] = [genome_id]

    def _update_species_collection(self, new_representatives: Dict[int, int], new_members: Dict[int, List[int]], generation: int):
        """ Update species data with new members and representatives for the new generation. """
        for species_id, rep_id in new_representatives.items():
            if not new_members.get(species_id):
                continue
            species = self.species.get(species_id, self.species_factory(species_id, generation))
            member_dict = {genome_id: self.population[genome_id] for genome_id in new_members[species_id]}
            species.update(self.population[rep_id], member_dict)
            self.species[species_id] = species
            for genome_id in new_members[species_id]:
                self.genome_to_species[genome_id] = species_id
    
    def _log_species_statistics(self):
        """ Log mean and standard deviation of genetic distances. """
        distances = list(self.distance_cache.distances.values())
        if distances:
            self.reporters.info(
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
