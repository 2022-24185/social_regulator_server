
from typing import List
from neat.math_util import mean

from neuroevolution.evolution.species import MixedGenerationSpecies


class FitnessManager:
    def __init__(self):
        self.all_new_fitnesses = []

    def collect_new_fitnesses(self, active_species: List[MixedGenerationSpecies], evaluated_genome_ids: List[int]):
        """
        Collects the fitnesses of all genomes in the active species.
        
        :param active_species: A list of active species.
        :param evaluated_genome_ids: A list of genome IDs that have been evaluated.
        """
        self.all_new_fitnesses = []
        for species in active_species: 
            active_genomes = {gid: genome for gid, genome in species.members.items() if gid in evaluated_genome_ids}
            new_fitnesses = [genome.fitness for genome in active_genomes.values()]
            self.all_new_fitnesses.extend(new_fitnesses)
    
    def adjust_fitnesses(self, active_species: List[MixedGenerationSpecies], evaluated_genome_ids: List[int]):
        """
        Adjusts the fitness of each species based on the fitness of its members.
        
        :param active_species: A list of active species.
        :param evaluated_genome_ids: A list of genome IDs that have been evaluated.
        :return: A list of adjusted fitness values for each species.
        """
        self.collect_new_fitnesses(active_species, evaluated_genome_ids)
        if not self.all_new_fitnesses:
            return []
        min_fitness = min(self.all_new_fitnesses)
        max_fitness = max(self.all_new_fitnesses)
        new_fitness_range = max(1.0, max_fitness - min_fitness) if max_fitness != min_fitness else 1.0
        adjusted_fitnesses = []
        for species in active_species:
            mean_species_fitness = mean(species.get_fitnesses(evaluated_genome_ids))
            af = (mean_species_fitness - min_fitness) / new_fitness_range
            species.set_adjusted_fitness(af)
            adjusted_fitnesses.append(af)
        return adjusted_fitnesses