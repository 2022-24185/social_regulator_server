from typing import List
from neat.math_util import mean
from neuroevolution.evolution.mixed_generation_species import MixedGenerationSpecies

@staticmethod
def compute_species_spawn(
    adjusted_fitness: float,
    dying_in_species: int,
    total_adjusted_fitness: float,
    total_dying_pop: int,
    min_species_size: int,
) -> int:
    """Compute the spawn for an individual species."""
    if total_adjusted_fitness > 0:
        expected_species_size = max(
            min_species_size,
            adjusted_fitness / total_adjusted_fitness * total_dying_pop,
        )
    else:
        expected_species_size = min_species_size

    size_diff = (expected_species_size - dying_in_species) * 0.5
    rounded_diff = int(round(size_diff))
    new_species_size = dying_in_species
    if abs(rounded_diff) > 0:
        new_species_size += rounded_diff
    elif size_diff > 0:
        new_species_size += 1
    elif size_diff < 0:
        new_species_size -= 1

    return new_species_size

@staticmethod
def compute_species_adjusted_fitness(
    remaining_species: List[MixedGenerationSpecies],
    all_fitnesses: List[float],
    genome_ids_to_consider: List[int],
) -> List[float]:
    """Compute the adjusted fitness for each species."""
    min_fitness = min(all_fitnesses)
    max_fitness = max(all_fitnesses)
    fitness_range = max(1.0, max_fitness - min_fitness)
    for species in remaining_species:
        mean_species_fitness = mean(
            [
                m.fitness
                for gid, m in species.members.items()
                if gid in genome_ids_to_consider
            ]
        )
        af = (mean_species_fitness - min_fitness) / fitness_range
        species.adjusted_fitness = af
    return [s.adjusted_fitness for s in remaining_species]

@staticmethod
def normalize_spawn_counts(
    offspring_per_species: List[int], total_dying_pop: int, min_species_size: int
) -> List[int]:
    """Normalize the spawn amounts so that the next generation is roughly
    the population size requested by the user."""
    total_offspring = sum(offspring_per_species)
    norm = total_dying_pop / total_offspring
    # Calculate the final number of offspring each species should produce
    final_offspring_counts = []
    for offspring_count in offspring_per_species:
        normalized_offspring_count = int(round(offspring_count * norm))
        final_offspring_count = max(min_species_size, normalized_offspring_count)
        final_offspring_counts.append(final_offspring_count)

    return final_offspring_counts

@staticmethod
def compute_offspring_counts(
    adjusted_fitness: List[float],
    dying_per_species: List[int],
    total_dying_pop: int,
    min_species_size: int,
) -> List[int]:
    """Compute the proper number of offspring per species 
    (proportional to the average fitness of the species)."""
    total_af = sum(adjusted_fitness)

    spawn_amounts = [
        compute_species_spawn(
            af, ps, total_af, total_dying_pop, min_species_size
        )
        for af, ps in zip(adjusted_fitness, dying_per_species)
    ]

    spawn_amounts = normalize_spawn_counts(
        spawn_amounts, total_dying_pop, min_species_size
    )

    return spawn_amounts
