"""Divides the population into species based on genomic distances."""

import math
import random
from itertools import count
from typing import Dict, List, Tuple

from neat.genome import DefaultGenome

Member = Tuple[int, DefaultGenome]
Members = List[Member]

class OffspringGenerator:
    def __init__(self, config):
        """
        Initializes the offspring generator.
        
        :param config: The NEAT configuration.
        :type config: neat.Config
        """
        self.config = config
        self.genome_indexer = count(1)

    def mate_parents(self, parent1: DefaultGenome, parent2: DefaultGenome) -> Member:
        """
        Produces a single offspring from two parents.

        :param parent1: The first parent.
        :param parent2: The second parent.

        :return: The child as a Member
        """
        child_id = next(self.genome_indexer)
        child: DefaultGenome = self.config.genome_type(child_id)
        child.configure_crossover(parent1, parent2, self.config.genome_config)
        child.mutate(self.config.genome_config)
        return child_id, child

    def create_offspring(self, parent_pool: Members, spawn: int) -> Dict[int, DefaultGenome]:
        """
        Returns a list of offspring created from the members
        
        :param parent_pool: The members to produce offspring from.
        :param spawn: The number of offspring to produce.

        :return: A list of offspring.
        """
        if len(parent_pool) < 2:
            raise ValueError("Insufficient parents to generate offspring.")
        reproduction_cutoff = max(int(math.ceil(self.config.survival_threshold * len(parent_pool))), self.config.min_species_size)
        selected_members = parent_pool[:reproduction_cutoff]
        new_population = {}
        for _ in range(spawn):
            parent1, parent2 = random.sample(selected_members, 2)
            child_id, child = self.mate_parents(parent1[1], parent2[1])
            new_population[child_id] = child

        return new_population