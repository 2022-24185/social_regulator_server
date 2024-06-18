"""
Handles creation of genomes, either from scratch or by sexual or
asexual reproduction from parents.
"""

from typing import List, Tuple, Dict, Callable

import math
import random
from itertools import count
import logging

from neat.config import ConfigParameter, DefaultClassConfig
from neat.math_util import mean
from neat.six_util import iteritems
from neat.config import Config
from neat.genome import DefaultGenome, DefaultGenomeConfig
from neat.reporting import ReporterSet

from neuroevolution.evolution.tournament_species import (
    TournamentSpeciesSet,
    TournamentSpecies,
)
from neuroevolution.evolution.tournament_stagnation import TournamentStagnation
from neuroevolution.evolution.species_utils import compute_species_adjusted_fitness, compute_offspring_counts


# Type aliases for better readability
FitnessFunction = Callable[[List[Tuple[int, DefaultGenome]], Config], None]
FitnessSummarizer = Callable[[List[float]], float]
Population = Dict[int, DefaultGenome]
State = Tuple[Population, TournamentSpeciesSet, int]
StagnationResult = List[Tuple[int, TournamentSpecies, bool]]
Member = Tuple[int, DefaultGenome]
Members = List[Member]


class TournamentReproduction(DefaultClassConfig):
    """
    Implements the default NEAT-python reproduction scheme:
    explicit fitness sharing with fixed-time species stagnation.
    """

    @classmethod
    def parse_config(cls, param_dict):
        """Parse the configuration parameters."""
        return DefaultClassConfig(
            param_dict,
            [
                ConfigParameter("elitism", int, 0),
                ConfigParameter("survival_threshold", float, 0.2),
                ConfigParameter("min_species_size", int, 2),
            ],
        )

    def __init__(
        self, config: Config, reporters: ReporterSet, stagnation: TournamentStagnation
    ):
        # pylint: disable=super-init-not-called
        self.reproduction_config = config
        self.reporters = reporters
        self.genome_indexer = count(1)
        self.stagnation = stagnation
        self.ancestors = {}

    def create_new_genomes(
        self,
        genome_type: DefaultGenome,
        genome_config: DefaultGenomeConfig,
        num_genomes: int,
    ):
        """Create new genomes from scratch."""
        new_genomes: Dict[int, DefaultGenome] = {}
        for _ in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config)
            new_genomes[key] = g
            self.ancestors[key] = tuple()

        return new_genomes

    def update_stagnation_and_collect_fitnesses(self, species: TournamentSpeciesSet,
        genome_ids_to_consider: List[int], generation: int,) -> Tuple[List[float], List[TournamentSpecies]]:
        
        """Update species and collect the fitnesses of all members."""
        all_existing_fitnesses = []
        remaining_species = []
        for stag_species_id, stag_species, is_stagnant in self.stagnation.update(
            species, genome_ids_to_consider, generation
        ):
            if is_stagnant:
                self.reporters.species_stagnant(stag_species_id, stag_species)
            else:
                # Only consider selected genomes when collecting fitnesses
                selected_members = {
                    gid: genome
                    for gid, genome in stag_species.members.items()
                    if gid in genome_ids_to_consider
                }
                all_existing_fitnesses.extend(
                    m.fitness for m in selected_members.values()
                )
                remaining_species.append(stag_species)
        return all_existing_fitnesses, remaining_species


    def create_child(self, old_members: Members, config: Config) -> Member:
        """Select parents and create a child by crossover and mutation."""
        parent1_id, parent1 = random.choice(old_members)
        parent2_id, parent2 = random.choice(old_members)
        child_id = next(self.genome_indexer)
        child = config.genome_type(child_id)
        child.configure_crossover(parent1, parent2, config.genome_config)
        child.mutate(config.genome_config)
        self.ancestors[child_id] = (parent1_id, parent2_id)
        return child_id, child

    def create_offspring(
        self,
        old_members: Members,
        new_population: Population,
        spawn: int,
        config: Config,
    ) -> None:
        """Create new offspring by crossover and mutation."""
        logging.info(f"Creating offspring for {spawn} members")
        reproduction_cutoff = int(
            math.ceil(self.reproduction_config.survival_threshold * len(old_members))
        )
        reproduction_cutoff = max(reproduction_cutoff, 2)
        logging.info(f"Reproduction cutoff: {reproduction_cutoff}")
        old_members = old_members[:reproduction_cutoff]
        logging.info(f"Total members after reproduction cutoff: {len(old_members)}")
        while spawn > 0:
            spawn -= 1
            child_id, child = self.create_child(old_members, config)
            new_population[child_id] = child
        logging.info(f"Total offspring created: {len(new_population)}")

    def preserve_elite_genomes(self, old_members: Members, new_population: Population, spawn: int) -> int:
        """Handle elitism by preserving the best members of the old population."""
        logging.info(f"Preserving elite genomes")
        if self.reproduction_config.elitism > 0:
            for i, m in old_members[: self.reproduction_config.elitism]:
                new_population[i] = m
                spawn -= 1
        return spawn

    def prepare_old_members_for_reproduction(self, species: TournamentSpecies, genomes_to_consider: List[int]) -> Members:
        """Prepare old members for reproduction."""
        logging.info(f"Preparing old members for species {species.key}")
        old_members = [(gid, genome) for gid, genome in iteritems(species.members) if gid in genomes_to_consider]
        species.members = {}
        old_members.sort(reverse=True, key=lambda x: x[1].fitness)
        return old_members

    def create_new_population(
        self,
        config: Config,
        offspring_per_species: List[int],
        species_set: TournamentSpeciesSet,
        non_stagnated_species: List[TournamentSpecies],
        genome_ids_to_consider: List[int],
    ) -> Population:
        """Create a new population from the old species.
        This includes elitism, selection of parents, and creation of offspring.
        """
        logging.info(f"Creating new population from {len(non_stagnated_species)} non-stagnated species.")
        new_population = {}
        logging.info("Resetting species sets for re-classification")
        species_set.species = {}
        for n_offspring, current_species in zip(
            offspring_per_species, non_stagnated_species
        ):
            n_offspring = max(n_offspring, self.reproduction_config.elitism)
            logging.info(f"Creating {n_offspring} offspring for species {current_species.key}")
            assert n_offspring > 0
            old_members = self.prepare_old_members_for_reproduction(current_species, genome_ids_to_consider)
            n_offspring = self.preserve_elite_genomes(
                old_members, new_population, n_offspring
            )
            logging.info(f"Total offspring after preserving elite genomes: {n_offspring}")
            if n_offspring > 0:
                logging.info(f"Creating offspring for species {current_species.key}")
                self.create_offspring(old_members, new_population, n_offspring, config)
            else: 
                logging.info(f"No offspring created for species {current_species.key}")
        logging.info(f"Total new population created: {len(new_population)}")
        return new_population

    def reproduce_selected(
        self,
        config: Config,
        species_set: TournamentSpeciesSet,
        generation: int,
        selected_genome_ids: List[int],
    ) -> Population:
        """
        Handles creation of genomes from a selected subset of species, either
        from scratch or by sexual or asexual reproduction from parents.
        """
        total_dying_pop = len(selected_genome_ids)
        all_fitnesses, non_stagnant_species = self.update_stagnation_and_collect_fitnesses(
            species_set, selected_genome_ids, generation)
        logging.info(f"Total dying population: {total_dying_pop}")
        if not non_stagnant_species:
            logging.info("All species are stagnant - no new population created.")
            self.reporters.info("All species are stagnant - no new population created.")
            species_set.species = {}
            return {}
        logging.info(f"Total non-stagnant species: {len(non_stagnant_species)}")
        adjusted_fitnesses = self.compute_adjusted_fitnesses(
            non_stagnant_species, all_fitnesses, selected_genome_ids)
        logging.info(f"Total adjusted fitnesses: {len(adjusted_fitnesses)}")
        self.report_average_adjusted_fitness(adjusted_fitnesses)

        dying_per_species = self.compute_dying_per_species(non_stagnant_species, selected_genome_ids)
        min_species_size = self.get_min_species_size()
        offspring_per_species = compute_offspring_counts(
            adjusted_fitnesses, dying_per_species, total_dying_pop, min_species_size)
        logging.info(f"Total offspring per species: {offspring_per_species}")

        new_population = self.create_new_population(
            config, offspring_per_species, species_set, non_stagnant_species, selected_genome_ids)
        logging.info(f"Returning population: {len(new_population)}")
        return new_population
    
    def compute_adjusted_fitnesses(
        self,
        non_stagnant_species: List[TournamentSpecies],
        all_fitnesses: List[float],
        selected_genome_ids: List[int],
    ) -> List[float]:
        """Compute adjusted fitnesses for non-stagnant species."""
        return compute_species_adjusted_fitness(
            non_stagnant_species, all_fitnesses, selected_genome_ids
        )

    def report_average_adjusted_fitness(self, adjusted_fitnesses: List[float]) -> None:
        """Report the average adjusted fitness."""
        avg_adjusted_fitness = mean(adjusted_fitnesses)
        self.reporters.info(f"Average adjusted fitness: {avg_adjusted_fitness:.3f}")

    def compute_dying_per_species(
        self,
        non_stagnant_species: List[TournamentSpecies],
        selected_genome_ids: List[int],
    ) -> List[int]:
        """Compute the number of dying members per species."""
        return [
            len(set(species.members.keys()).intersection(selected_genome_ids))
            for species in non_stagnant_species
        ]

    def get_min_species_size(self) -> int:
        """Get the minimum species size."""
        return max(
            self.reproduction_config.min_species_size, self.reproduction_config.elitism
        )

# main function for module
if __name__ == "__main__":
    pass
