"""Divides the population into species based on genomic distances."""
from typing import Dict, List, Set, Tuple, Callable, TYPE_CHECKING
from neuroevolution.evolution.species import MixedGenerationSpecies

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
        self.species: Dict[int, 'MixedGenerationSpecies'] = {}
        self.genome_to_species: Dict[int, int] = {}

    def reset(self) -> None:
        """
        Clears all species from the set.
        """
        self.species.clear()

    def create_new_species(self, generation) -> int:
        """
        Adds a new species to the set.

        :param generation: The generation in which the species was created.
        :return: The ID of the newly created species.
        """
        new_species_id = (
                        max(self.get_all_species_ids(), default=0) + 1
                    )
        species_instance = species_factory(new_species_id, generation)
        self.species[new_species_id] = species_instance
        return new_species_id
    
    def update_species_representative(self, species_id: int, rep_id: int, rep: 'DefaultGenome') -> None:
        """
        Updates the representative genome of the specified species.

        :param species_id: An integer representing the ID of the species.
        :param rep_id: An integer representing the ID of the genome to be set as the representative.
        :param rep: The genome instance to be set as the representative.
        """
        member = (rep_id, rep)
        self.add_member(species_id, member)
        self.set_representative(species_id, member)

    def add_member(self, species_id: int, member: Member) -> None:
        """
        Adds a genome as a member of the specified species.

        :param species_id: An integer representing the ID of the species.
        :param member: A tuple containing the genome ID and the genome instance to be added to the species.
        """
        if species_id in self.species:
            self.species[species_id].add_member(member)
            self.genome_to_species[member[0]] = species_id
        else:
            raise ValueError(f"Species ID {species_id} does not exist.")

    def set_representative(self, species_id: int, representative: Member) -> None:
        """
        Sets the representative genome for a specified species.

        :param species_id: An integer representing the ID of the species.
        :param representative: The genome instance to be set as the representative of the species.
        """
        self.get_species(species_id).set_representative(representative)

    def remove_species(self, species_id: int) -> None:
        """
        Removes a species from the set.
        :param species_id: An integer representing the ID of the species to be removed.
        """
        to_delete = [member_id for member_id, sid in self.genome_to_species.items() if sid == species_id]
        for member_id in to_delete:
            del self.genome_to_species[member_id]
        if species_id in self.species:
            del self.species[species_id]
        else:
            raise ValueError(f"No species found with ID {species_id}")

    def get_species(self, species_id: int) -> 'MixedGenerationSpecies':
        """
        Retrieves a species by its ID.

        :param species_id: An integer representing the ID of the species to retrieve.
        :return: The species instance corresponding to the given ID.
        """
        return self.species[species_id]
    
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
    
    def get_active_species(self) -> List['MixedGenerationSpecies']:
        """
        Returns the IDs of all active species currently in the set.
        """
        return [species for sid, species in self.species.items() if species.is_active()]
    
    def get_unspeciated(self, population: Dict[int, "DefaultGenome"]) -> Set[int]: 
        """
        Returns the IDs of all genomes that have not been assigned to any species.
        """
        return set([
                genome_id
                for genome_id in population.keys()
                if not self.is_genome_speciated(genome_id)
        ])
    
    def get_representative_ids(self) -> Dict[int, int]: 
        """
        Retrieves the representative genome IDs for all species.

        :return: A dictionary mapping species IDs to their representative genome IDs.
        """
        return {species_id: species.get_representative_id() for species_id, species in self.species.items()}
    
    def get_species_id_for_genome(self, individual_id: int) -> int:
        """
        Finds the species ID for a given genome.

        :param individual_id: An integer representing the ID of the genome.
        :return: The species ID to which the genome belongs, or None if it doesn't belong to any species.
        """
        for species_id, species in self.species.items():
            if species.is_member(individual_id):
                return species_id
        return None
    
    def get_compatible_genomes(self, 
        species_ids: List[int], 
        genome_id: int, 
        population: Dict[int, 'DefaultGenome'], 
        compatibility_fn: Callable[['DefaultGenome', 'DefaultGenome'], bool]
    ) -> List[Tuple[float, int]]:
        """Returns a list of (compatibility, group_id) tuples for each group in groups."""
        return [(compatibility_fn(population[group_id], population[genome_id]), group_id)
                for group_id in species_ids
                if compatibility_fn(population[group_id], population[genome_id])]
    
    def is_genome_speciated(self, individual_id: int) -> bool:
        """
        Checks if a genome has been assigned to a species.

        :param individual_id: An integer representing the ID of the genome.
        :return: True if the genome is part of a species, False otherwise.
        """
        return individual_id in self.genome_to_species
        # for _, species in self.species.items():
        #     if species.is_member(individual_id):
        #         return True
        # return False
    
    def mark_stagnant(self, species_id: int) -> None:
        """
        Marks a species as stagnant.
        """
        self.get_species(species_id).mark_stagnant()

    def remove_stagnant_species(self) -> None:
        """
        Removes all stagnant species from the set.
        """
        to_remove = []
        for id, species in self.species.items():
            if not species.is_active():
                to_remove.append(id)
        for id in to_remove:
            self.remove_species(id)
    
    

    
