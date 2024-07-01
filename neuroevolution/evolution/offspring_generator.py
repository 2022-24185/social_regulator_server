"""Divides the population into species based on genomic distances."""
import random
from itertools import count
from typing import Dict, List, Tuple

from neat.genome import DefaultGenome

Member = Tuple[int, DefaultGenome]
Members = List[Member]

class GenomeFactory:
    """Creates new genomes either from scratch or from parents."""
    def __init__(self, genome_type, genome_config):
        self.genome_type = genome_type
        self.genome_config = genome_config

    def create_genome(self, key):
        """Create a new genome with the given key."""
        genome = self.genome_type(key)
        genome.configure_new(self.genome_config)
        return genome

class OffspringGenerator:
    def __init__(self, genome_type, genome_config, reprod_config):
        """
        Initializes the offspring generator.
        
        :param genome_type: The type of genome to produce.
        :param genome_config: The genome configuration.
        :param reprod_config: The reproduction configuration.
        :param index: The starting index for the genome id indexer.
        """
        self.reprod_config = reprod_config
        self.genome_indexer = count(1)
        self.genome_factory = GenomeFactory(genome_type, genome_config)

    def mate_parents(self, parent1: DefaultGenome, parent2: DefaultGenome) -> Member:
        """
        Produces a single offspring from two parents.

        :param parent1: The first parent.
        :param parent2: The second parent.

        :return: The child as a Member
        """
        child_id = next(self.genome_indexer)
        child: DefaultGenome = self.genome_factory.genome_type(child_id)
        child.configure_crossover(parent1, parent2, self.genome_factory.genome_config)
        child.mutate(self.genome_factory.genome_config)
        return child_id, child

    def create_offspring(self, parent_pool: Members, spawn: int, parent_cutoff: int) -> Dict[int, DefaultGenome]:
        """
        Returns a list of offspring created from the members
        
        :param parent_pool: The members to produce offspring from.
        :param spawn: The number of offspring to produce.

        :return: A list of offspring.
        """
        if len(parent_pool) < 2:
            raise ValueError("Insufficient parents to generate offspring.")
        top_parents = parent_pool[:parent_cutoff]
        new_population = {}
        for _ in range(spawn):
            parent1, parent2 = random.sample(top_parents, 2)
            child_id, child = self.mate_parents(parent1[1], parent2[1])
            new_population[child_id] = child

        return new_population
    
    def create_without_parents(self, num_genomes: int) -> Dict[int, DefaultGenome]:
        """
        Create a number of new genomes from scratch.
        
        :param num_genomes: The number of genomes to create.
        :return: A dictionary mapping genome key to genome.
        """
        new_genomes = {}
        for _ in range(num_genomes):
            key = next(self.genome_indexer)
            g = self.genome_factory.create_genome(key)
            new_genomes[key] = g
        return new_genomes