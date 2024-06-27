"""Contains the MixedGenerationSpecies class, which holds information about a species and its members. """

from typing import TYPE_CHECKING, List, Optional, Set, Tuple

from neat.six_util import iteritems

if TYPE_CHECKING:
    from neat.genome import DefaultGenome

Member = Tuple[int, 'DefaultGenome']
Members = List[Member]


class MixedGenerationSpecies:
    """Holds information about a species and its members."""
    def __init__(self, key, generation):
        """
        Initializes the species with the given key and generation.
        
        :param key: The unique ID of the species.
        :param generation: The generation number.
        """
        self.key = key
        self.active = True
        self.created = generation
        self.last_improved = generation
        self.representative : 'DefaultGenome'  = None
        self.members: Members = {}
        self.fitness = None
        self.adjusted_fitness = None
        self.dying_count = 0
        self.expected_offspring = 0
        self.fitness_history = []

    def add_member(self, member: Member):
        """
        Adds a member to the species.
        
        :param member: A tuple containing the genome ID and the genome instance to be added to the species.
        """
        self.members[member[0]] = member[1]

    def set_representative(self, representative: Member):
        """
        Sets the representative genome for the species.
        
        :param representative: The genome instance to be set as the representative of the species.
        """
        self.representative = representative

    def get_representative_id(self) -> int:
        """
        Returns the ID of the representative genome.
        """
        return self.representative[0] if self.representative else None

    def get_fitnesses(self, genome_ids_to_consider: Optional[List[float]] = None):
        """
        Returns a list of the fitnesses of the members, optionally filtered by specific genome IDs.
        
        :param genome_ids_to_consider: A list of genome IDs to consider, or None to consider all members.
        :return: A list of fitness values.
        """
        if genome_ids_to_consider is None:
            return [genome.fitness for _, genome in self.members.items()]
        else:
            return [genome.fitness for genome_id, genome in self.members.items() if genome_id in genome_ids_to_consider]

    def get_sorted_by_fitness(self, selected_genome_indices: List[int]) -> Members:
        """Sorts and pops the members with the given indices from the species."""
        old_members = [(gid, genome) for gid, genome in iteritems(self.members) if gid in selected_genome_indices]
        old_members.sort(reverse=True, key=lambda x: x[1].fitness)

        return old_members
    
    def set_adjusted_fitness(self, adjusted_fitness):
        """Sets the adjusted fitness of the species."""
        self.adjusted_fitness = adjusted_fitness

    def mark_stagnant(self):
        """Marks the species as stagnant."""
        self.active = False

    def kill_members(self, dead_genomes: Set[int]):
        """Removes members that are no longer alive."""
        self.members = {gid: genome for gid, genome in self.members.items() if gid not in dead_genomes}
        self.dying_count = len(dead_genomes)

    def compute_expected_size(self, min_species_size, total_adjusted_fitness, total_dying_pop):
        """Computes the expected number of genomes in a species."""
        if total_adjusted_fitness > 0:
            expected_species_size = max(
                min_species_size,
                self.adjusted_fitness / total_adjusted_fitness * total_dying_pop,
            )
        else:
            expected_species_size = min_species_size
        return expected_species_size
    
    def compute_pop_deficit(self, expected_species_size):
        """
        Computes the population deficit for the species.
        
        :param expected_species_size: The expected size of the species.
        :return: The population deficit.
        """
        size_diff = (expected_species_size - self.dying_count) * 0.5
        rounded_diff = int(round(size_diff))
        population_deficit = self.dying_count
        if abs(rounded_diff) > 0:
            population_deficit += rounded_diff
        elif size_diff > 0:
            population_deficit += 1
        elif size_diff < 0:
            population_deficit -= 1

        return population_deficit
    
    def is_member(self, genome_id):
        """Returns True if the given genome ID is a member of the species."""
        return genome_id in self.members
    
    def is_active(self):
        """Returns True if the species is still active."""
        return self.active
