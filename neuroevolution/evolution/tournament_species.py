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


class TournamentSpeciesSet(DefaultClassConfig):
    """Encapsulates the default speciation scheme."""

    def __init__(self, config, reporters):
        # pylint: disable=super-init-not-called
        self.species_set_config : Config = config
        self.reporters = reporters
        self.indexer = count(1)
        self.species: Dict[int, TournamentSpecies] = {}
        self.genome_to_species = {}
        self.distances = None
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

    def speciate(self, generation: int, config: Config):
        """ Organize genomes into species based on genetic similarity. """
        logging.info(f"Speciating generation {generation}")
        self.distances = GenomeDistanceCache(config.genome_config)
        compatibility_threshold = self.species_set_config.compatibility_threshold
        new_representatives, new_members = self._find_new_representatives()
        self._partition_population(new_representatives, new_members, compatibility_threshold)
        self._update_species_collection(new_representatives, new_members, generation)
        self._log_species_statistics()
    
    def _find_new_representatives(self) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        """ Determine new representatives based on minimum genetic distance to unspeciated genomes. """
        logging.info("Finding new representatives")
        new_representatives = {}
        new_members = {}
        for species_id, species in self.species.items():
            if not self.unspeciated:
                break
            candidates = [(self.distances(species.representative, self.population[gid]), gid)
                          for gid in self.unspeciated]
            _, new_rep_genome_id = min(candidates, key=lambda x: x[0])
            new_representatives[species_id] = new_rep_genome_id
            new_members[species_id] = [new_rep_genome_id]
            self.unspeciated.remove(new_rep_genome_id)
        return new_representatives, new_members
    
    def _partition_population(self, new_representatives: Dict[int, int], 
                              new_members: Dict[int, List[int]],
                              compatibility_threshold: float) -> None:
        """ Group remaining genomes into existing or new species based on compatibility. """
        logging.info(f"Partitioning population into species. Unspeciated: {len(self.unspeciated)}")

        while self.unspeciated:
            genome_id = self.unspeciated.pop()
            genome = self.population[genome_id]
            candidates = [(self.distances(self.population[representative_id], genome), species_id)
                          for species_id, representative_id in new_representatives.items()
                          if self.distances(self.population[representative_id], genome) < compatibility_threshold]
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
            species = self.species.get(species_id, TournamentSpecies(species_id, generation))
            member_dict = {genome_id: self.population[genome_id] for genome_id in new_members[species_id]}
            species.update(self.population[rep_id], member_dict)
            self.species[species_id] = species
            for genome_id in new_members[species_id]:
                self.genome_to_species[genome_id] = species_id
    
    def _log_species_statistics(self):
        """ Log mean and standard deviation of genetic distances. """
        distances = list(self.distances.distances.values())
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
    
    # def speciate(self, config, population, generation):
    #     """
    #     Place genomes into species by genetic similarity.

    #     Note that this method assumes the current representatives of the species are from the old
    #     generation, and that after speciation has been performed, the old representatives should be
    #     dropped and replaced with representatives from the new generation.  If you violate this
    #     assumption, you should make sure other necessary parts of the code are updated to reflect
    #     the new behavior.
    #     """
    #     assert isinstance(population, dict)

    #     compatibility_threshold = self.species_set_config.compatibility_threshold

    #     # Find the best representatives for each existing species.
    #     unspeciated = set(iterkeys(population))
    #     distances = GenomeDistanceCache(config.genome_config)
    #     new_representatives = {}
    #     new_members = {}
    #     for sid, s in iteritems(self.species):
    #         candidates = []
    #         for gid in unspeciated:
    #             g = population[gid]
    #             d = distances(s.representative, g)
    #             candidates.append((d, g))

    #         # The new representative is the genome closest to the current representative.
    #         ignored_rdist, new_rep = min(candidates, key=lambda x: x[0])
    #         new_rid = new_rep.key
    #         new_representatives[sid] = new_rid
    #         new_members[sid] = [new_rid]
    #         unspeciated.remove(new_rid)

    #     # Partition population into species based on genetic sximilarity.
    #     while unspeciated:
    #         gid = unspeciated.pop()
    #         g = population[gid]

    #         # Find the species with the most similar representative.
    #         candidates = []
    #         for sid, rid in iteritems(new_representatives):
    #             rep = population[rid]
    #             d = distances(rep, g)
    #             if d < compatibility_threshold:
    #                 candidates.append((d, sid))

    #         if candidates:
    #             ignored_sdist, sid = min(candidates, key=lambda x: x[0])
    #             new_members[sid].append(gid)
    #         else:
    #             # No species is similar enough, create a new species, using
    #             # this genome as its representative.
    #             sid = next(self.indexer)
    #             new_representatives[sid] = gid
    #             new_members[sid] = [gid]

    #     # Update species collection based on new speciation.
    #     self.genome_to_species = {}
    #     for sid, rid in iteritems(new_representatives):
    #         s = self.species.get(sid)
    #         if s is None:
    #             s = TournamentSpecies(sid, generation)
    #             self.species[sid] = s

    #         members = new_members[sid]
    #         for gid in members:
    #             self.genome_to_species[gid] = sid

    #         member_dict = dict((gid, population[gid]) for gid in members)
    #         s.update(population[rid], member_dict)

    #     gdmean = mean(itervalues(distances.distances))
    #     gdstdev = stdev(itervalues(distances.distances))
    #     self.reporters.info(
    #         "Mean genetic distance {0:.3f}, standard deviation {1:.3f}".format(
    #             gdmean, gdstdev
    #         )
    #     )
