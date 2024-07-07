"""Divides the population into species based on genomic distances."""
import sys
from typing import Dict, List, Set, Tuple, Callable, TYPE_CHECKING
from neuroevolution.evolution.species import MixedGenerationSpecies
from neuroevolution.evolution.genome_manager import GenomeManager

if TYPE_CHECKING:
    from neat.genome import DefaultGenome

Member = Tuple[int, 'DefaultGenome']
Members = List[Member]
    
def species_factory(species_id, generation):
    """Creates a new species instance with the given ID and generation."""
    return MixedGenerationSpecies(species_id, generation)

class MixedGenerationSpeciesSet():
    """Holds information about the current species and their members."""

    def __init__(self):
        """
        Initializes the species set with an empty dictionary of species.
        """
        # pylint: disable=super-init-not-called
        self.generation: int = 0
        self.species: Dict[int, 'MixedGenerationSpecies'] = {}
        self.active: Set[int] = set()
        self.current_fitnesses: Dict[int, float] = {}
        self.best_fitnesses: Dict[int, float] = {}
        self.last_improved: Dict[int, int] = {}
        self.population_deficits: Dict[int, int] = {}
        self.expected_offspring: Dict[int, int] = {}

    def set_generation(self, generation: int) -> None:
        """
        Sets the current generation number.

        :param generation: The generation number.
        """
        self.generation = generation

    def reset(self) -> None:
        """
        Clears all species from the set.
        """
        self.species.clear()
        self.set_generation(0)
        self.create_new_species()

    def create_new_species(self) -> int:
        """
        Adds a new species to the set.

        :param generation: The generation in which the species was created.
        :return: The ID of the newly created species.
        """
        new_species_id = (max(self.get_all_species_ids(), default=0) + 1)
        species_instance = species_factory(new_species_id, self.get_generation())
        self.add_species(new_species_id, species_instance)
        return new_species_id
    
    def add_species(self, species_id: int, species: 'MixedGenerationSpecies'):
        """
        Adds a species to the species set.

        :param species_id: The ID of the species.
        :param species: The species instance.
        """
        self.species[species_id] = species
        self.active.add(species_id)
        self.current_fitnesses[species_id] = -1.0
        self.best_fitnesses[species_id] = -1.0
        self.last_improved[species_id] = self.generation

    def remove_species(self, species_id: int):
        """
        Removes a species from the species set.

        :param species_id: The ID of the species.
        """
        if species_id in self.species:
            del self.species[species_id]
            self.active.discard(species_id)
            del self.current_fitnesses[species_id]
            del self.best_fitnesses[species_id]
            del self.last_improved[species_id]
    
    def update_species_fitness(self, species_id: int, fitness: float):
        """
        Updates the fitness of a species.

        :param species_id: The ID of the species.
        :param fitness: The new fitness value.
        """
        if species_id in self.species:
            self.current_fitnesses[species_id] = fitness
            if fitness > self.best_fitnesses[species_id]:
                self.best_fitnesses[species_id] = fitness
                self.last_improved[species_id] = self.generation

    def set_population_deficit(self, species_id: int, deficit: int):
        """
        Sets the population deficit of a species.
            
        :param species_id: The ID of the species.
        :param deficit: The population deficit.
        """
        self.population_deficits[species_id] = deficit

    def set_expected_offspring(self, species_id: int, expected_offspring: int):
        """
        Sets the expected number of offspring of a species.

        :param species_id: The ID of the species.
        :param expected_offspring: The expected number of offspring.
        """
        self.expected_offspring[species_id] = expected_offspring

    def get_species_current_fitness(self, species_id: int) -> float:
        """
        Retrieves the fitness of a species.

        :param species_id: The ID of the species.
        :return: The fitness value of the species.
        """
        return self.current_fitnesses.get(species_id)
    
    def get_species_best_fitness(self, species_id: int) -> float:
        """
        Retrieves the best fitness of a species.

        :param species_id: The ID of the species.
        :return: The best fitness value of the species.
        """
        return self.best_fitnesses.get(species_id, -1.0)
        
    def get_generation(self) -> int: 
        """
        Returns the current generation number.
        """
        return self.generation

    def get_species(self, species_id: int) -> 'MixedGenerationSpecies':
        """
        Retrieves a species by ID.

        :param species_id: The ID of the species.
        :return: The species instance.
        """
        return self.species.get(species_id)
    
    def get_all_species_ids(self) -> List[int]:
        """
        Returns the IDs of all species currently in the set.
        :return: A list of integers representing the IDs of all species.
        """
        return list(self.species.keys())
    
    def get_all_species_objects(self) -> List['MixedGenerationSpecies']:
        """
        Returns all species currently in the set.
        :return: A list of species instances.
        """
        return list(self.species.values())
    
    def get_all_species(self) -> List[Tuple[int, 'MixedGenerationSpecies']]:
        """
        Returns all species currently in the set.
        :return: A list of species instances.
        """
        return list(self.species.items())

    def get_species_fitness_history(self, species_id: int) -> List[float]:
        """
        Gets the fitness history of a species.

        :param species_id: The ID of the species.
        :return: The fitness history of the species.
        """
        species = self.get_species(species_id)
        if species:
            return species.fitness_history
        return []
    
    def get_previous_best_fitness(self, species_id: int) -> float:
        """
        Gets the previous best fitness of a species.
        
        :param species_id: The ID of the species.
        :return: The previous best fitness of the species.
        """
        species = self.get_species(species_id)
        if species:
            return max(species.fitness_history) if species.fitness_history else -sys.float_info.max
        return 0.0
    
    def get_species_ids_fitness_sorted(self) -> List[Tuple[int, float]]:
        """
        Returns all species_ids currently in the set, sorted top fitness first.
        :return: A list of species instances.
        """
        return sorted(self.current_fitnesses.items(), key=lambda x: x[1], reverse=True)
    
    def get_deficit(self, species_id) -> int:
        """
        Returns the population deficit of a species.
        """
        return self.population_deficits.get(species_id)
    
    def get_expected_offspring(self, species_id) -> int:
        """
        Returns the expected number of offspring of a species.
        """
        return self.expected_offspring.get(species_id)
    
    def get_total_pop_deficit(self) -> int:
        """
        Returns the total population deficit of all species.
        """
        return sum(self.population_deficits.values())
    
    def is_stagnant(self, species_id, limit: int) -> bool:
        """
        Returns whether a species is stagnant.
        """
        return self.generation - self.last_improved[species_id] > limit
    
    def get_active_species(self) -> List['MixedGenerationSpecies']:
        """
        Returns the IDs of all active species currently in the set.
        """
        return [self.get_species(sid) for sid in self.active]

    def mark_stagnant(self, species_id: int) -> None:
        """
        Marks a species as stagnant.
        """
        self.active.discard(species_id)

    def remove_stagnant_species(self) -> List[int]:
        """
        Removes all stagnant species from the set.
        """
        to_remove = set(self.get_all_species_objects()) - set(self.get_active_species())
        print(f"total: {len(self.get_all_species_objects())}, active: {len(self.get_active_species())}, to_remove: {len(to_remove)}")
        for sid in to_remove:
            self.remove_species(sid)
        return to_remove
    
    

    
