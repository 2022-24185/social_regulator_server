""" This module contains the GenomeManager class, which is responsible for managing the states of the genomes in the population. """
from typing import Dict, Set, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from neat.genome import DefaultGenome
    from pydantic import BaseModel

class GenomeManager:
    def __init__(self):
        self.genomes: Dict[int, 'DefaultGenome'] = {}
        self.free_genomes: Set[int] = set()
        self.evaluated_genomes: Set[int] = set()
        self.elite_genomes: Set[int] = set()
        self.representative_genomes: Dict[int, int] = {}
        self.unspeciated_genomes: Set[int] = set()
        self.genome_species_map: Dict[int, int] = {}
        self.alive_genomes_count: int = 0

    def reset(self) -> None:
        """
        Resets the GenomeManager to its initial state.
        """
        self.genomes.clear()
        self.free_genomes.clear()
        self.evaluated_genomes.clear()
        self.elite_genomes.clear()
        self.representative_genomes.clear()
        self.unspeciated_genomes.clear()
        self.genome_species_map.clear()
        self.alive_genomes_count = 0

    def add_genome(self, genome_id: int, genome: 'DefaultGenome') -> None:
        """
        Add a genome to the population.
        
        :param genome_id: The ID of the genome to add.
        :param genome: The genome instance to add.
        """
        self.genomes[genome_id] = genome
        self.unspeciated_genomes.add(genome_id)
        self.free_genomes.add(genome_id)
        self.alive_genomes_count += 1

    def add_genomes(self, genomes: 'Dict[int, DefaultGenome]') -> None:
        """
        Add multiple genomes to the population.

        :param genomes: A dictionary of genomes to add.
        """
        for genome_id, genome in genomes.items():
            self.add_genome(genome_id, genome)

    def remove_genome(self, genome_id: int) -> None:
        """
        Remove a genome from the population.

        :param genome_id: The ID of the genome to remove.
        """
        if genome_id in self.genomes:
            del self.genomes[genome_id]
            self.evaluated_genomes.discard(genome_id)
            self.elite_genomes.discard(genome_id)
            self.unspeciated_genomes.discard(genome_id)
            self.free_genomes.discard(genome_id)
            if genome_id in self.genome_species_map:
                species_id = self.genome_species_map[genome_id]
                if species_id in self.representative_genomes and self.representative_genomes[species_id] == genome_id:
                    del self.representative_genomes[species_id]
                del self.genome_species_map[genome_id]
            self.alive_genomes_count -= 1

    def remove_genomes_in_species(self, species_id: int) -> None: 
        """
        Remove a species from the population.
            
        :param species_id: The ID of the species to remove.
        """
        genomes = self.get_genomes_in_species(species_id)
        for genome_id in genomes:
            self.remove_genome(genome_id)

    def clear_evaluated(self) -> None:
        """
        Clear all evaluated genomes.
        """
        self.evaluated_genomes.clear()
    
    def clear_elites(self) -> None:
        """
        Clear all elite genomes.
        """
        self.elite_genomes.clear()

    def set_unavailable(self, genome_id: int) -> None:
        """
        Remove a genome id from the list of free genomes

        :param genome_id: The ID of the genome to remove.
        """
        self.free_genomes.discard(genome_id)

    def set_evaluated(self, genome_id: int) -> None:
        """
        Mark a genome as evaluated.

        :param genome_id: The ID of the genome to mark as evaluated.
        """
        self.evaluated_genomes.add(genome_id)

    def set_elite(self, genome_id: int) -> None:
        """
        Mark a genome as elite.

        :param genome_id: The ID of the genome to mark as elite.
        """
        self.elite_genomes.add(genome_id)
        self.free_genomes.add(genome_id)

    def set_representative(self, species_id: int, genome_id: int) -> None:
        """
        Set a genome as the representative of a species.

        :param species_id: The ID of the species to set the representative genome of.
        :param genome_id: The ID of the genome to set as the representative.
        """
        self.representative_genomes[species_id] = genome_id

    def update_genome_data(self, genome_id: int, data: 'BaseModel'):
        """
        Update specific data for a given genome.
        
        :param genome_id: The ID of the genome to update.
        :param data: The data to update for the genome.
        """
        genome = self.get_genome(genome_id)
        if genome:
            genome.data = data
            return genome
        else:
            raise ValueError(f"Genome ID {genome_id} not found in the population.")

    def assign_genome_to_species(self, genome_id: int, species_id: int) -> None:
        """
        Assign a genome to a species.

        :param genome_id: The ID of the genome to assign to a species.
        :param species_id: The ID of the species to assign the genome to.
        """
        self.genome_species_map[genome_id] = species_id
        self.unspeciated_genomes.discard(genome_id)

    def get_genome(self, genome_id: int) -> Optional['DefaultGenome']:
        """
        Get a genome by its ID.

        :param genome_id: The ID of the genome to get.
        :return: The genome with the given ID, or None if it does not exist.
        """
        return self.genomes.get(genome_id)

    def get_all_genomes(self) -> List['DefaultGenome']:
        """
        Get all genomes in the population.

        :return: A list of all genomes in the population.
        """
        return list(self.genomes.values())

    def get_evaluated_genomes(self) -> List['DefaultGenome']:
        """
        Get all evaluated genomes in the population.

        :return: A list of all evaluated genomes in the population.
        """
        # print if there are any gids in evaluated genomes that are not in self.genomes
        for gid in self.evaluated_genomes:
            if gid not in self.genomes:
                print("gid in evaluated but not in genomes: ", gid)

        return [self.genomes[gid] for gid in self.evaluated_genomes]
    
    def get_dying_genomes_for_species(self, species_id: int) -> List['DefaultGenome']:
        """
        Get all dying genomes in the population.
        """
        return set(self.get_evaluated_genomes_in_species(species_id)) - set(self.elite_genomes)
    
    def get_evaluated_genome_fitnesses(self) -> List[float]:
        """
        Get the fitnesses of all evaluated genomes in the population.

        :return: A list of all evaluated genomes in the population.
        """
        return [self.get_fitness(gid) for gid in self.evaluated_genomes]
    
    def get_fitness(self, genome_id: int): 
        """
        Get the fitness of a genome.
            
        :param genome_id: The ID of the genome to get the fitness of.
        :return: The fitness of the genome.
        """
        return self.get_genome(genome_id).fitness
    
    def get_genome_fitnesses_for_species(self, species_id) -> List[Tuple[int, float]]:
        """
        Get the fitnesses of all genomes in a species.

        :param species_id: The ID of the species to get the genome fitnesses of.
        :return: A list of all genome fitnesses in the species.
        """
        return [(gid, self.get_fitness(gid)) for gid in self.get_evaluated_genomes_in_species(species_id)]
    
    def get_genomes_sorted_by_fitness(self, species_id) -> List['DefaultGenome']:
        """
        Get all the evaluated genomes the species, sorted from highest to lowest fitness.
        """
        gids = sorted(self.get_evaluated_genomes_in_species(species_id), key=self.get_fitness, reverse=True)
        return [self.get_genome(gid) for gid in gids]
    
    def get_evaluated_genomes_in_species(self, species_id) -> List[int]:
        """
        Get all evaluated genomes in a species.
        
        :param species_id: The ID of the species to get the evaluated genomes of.
        :return: A list of all evaluated genome ids in the species.
        """
        return [gid for gid in self.get_genomes_in_species(species_id) if gid in self.evaluated_genomes]
    
    def get_available_genomes(self) -> List['DefaultGenome']:
        """
        Get all available genomes in the population.
        """
        return [self.genomes[gid] for gid in self.free_genomes]

    def get_elite_genomes(self) -> List['DefaultGenome']:
        """
        Get all elite genomes in the population.

        :return: A list of all elite genomes in the population.
        """
        return [self.genomes[gid] for gid in self.elite_genomes]

    def get_representative(self, species_id: int) -> Optional['DefaultGenome']:
        """
        Get the representative genome of a species.

        :param species_id: The ID of the species to get the representative genome of.
        :return: The representative genome of the species, or None if the species does not exist.
        """
        if species_id not in self.representative_genomes:
            return None
        return self.genomes.get(self.representative_genomes.get(species_id))

    def get_genomes_in_species(self, species_id: int) -> List[int]:
        """
        Get all genomes in a species.

        :param species_id: The ID of the species to get the genomes of.
        :return: A list of all genomes in the species.
        """
        return [gid for gid, sid in self.genome_species_map.items() if sid == species_id]

    def get_alive_genomes_count(self) -> int:
        """
        Get the number of alive genomes in the population.

        :return: The number of alive genomes in the population.
        """
        return self.alive_genomes_count
    
    def get_unspeciated_genomes(self) -> List['DefaultGenome']:
        """
        Get all genomes that have not been assigned to a species.

        :return: A list of all genomes that have not been assigned to a species.
        """
        return [self.genomes[gid] for gid in self.unspeciated_genomes]
