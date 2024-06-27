
from neat.math_util import mean
from typing import List

from neuroevolution.evolution.species import MixedGenerationSpecies


class FitnessManager:
    def __init__(self):
        self.all_new_fitnesses = []

    def collect_new_fitnesses(self, active_species: List[MixedGenerationSpecies], evaluated_genome_ids: List[int]):
        self.all_new_fitnesses = []
        print("in new fitnesses")
        for species in active_species: 
            active_genomes = {gid: genome for gid, genome in species.members.items() if gid in evaluated_genome_ids}
            print(f"active genomes {active_genomes}")
            new_fitnesses = [genome.fitness for genome in active_genomes.values()]
            print(f"new fitnesses {new_fitnesses}")
            self.all_new_fitnesses.extend(new_fitnesses)
    
    def adjust_fitnesses(self, active_species: List[MixedGenerationSpecies], evaluated_genome_ids: List[int]):
        self.collect_new_fitnesses(active_species, evaluated_genome_ids)
        if not self.all_new_fitnesses:
            return []
        min_fitness = min(self.all_new_fitnesses)
        max_fitness = max(self.all_new_fitnesses)
        new_fitness_range = max(1.0, max_fitness - min_fitness) if max_fitness != min_fitness else 1.0
        adjusted_fitnesses = []
        for species in active_species:
            print("HELLOO")
            mean_species_fitness = mean(species.get_fitnesses(evaluated_genome_ids))
            print(f"mean_species_fitness: {mean_species_fitness}")
            af = (mean_species_fitness - min_fitness) / new_fitness_range
            species.set_adjusted_fitness(af)
            adjusted_fitnesses.append(af)
        return adjusted_fitnesses