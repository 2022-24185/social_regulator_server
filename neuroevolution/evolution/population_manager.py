"""Implements the core evolution algorithm."""
import random
from typing import Dict, List, TYPE_CHECKING

from neat.genome import DefaultGenome
from neat.config import Config

from neuroevolution.evolution.species_set import MixedGenerationSpeciesSet
from neuroevolution.evolution.stagnation import MixedGenerationStagnation
from neuroevolution.evolution.speciation import Speciation

if TYPE_CHECKING:
    from neuroevolution.server.models import UserData

# Type aliases for better readability
Population = Dict[int, DefaultGenome]

class CompleteExtinctionException(Exception):
    """Exception to raise when a population has no members."""
    def __init__(self, message):
        super().__init__(message)

class PopulationManager:
    """Manages the members of the population"""

    def __init__(self, config: Config):
        self.population = {}
        self.generation = 0
        self.speciation = self.create_speciation(config)
        self.available_genomes = []

    def create_speciation(self, config): 
        return Speciation(config)
    
    def set_new_population(self, new_population: Population):
        """Set the population to a new one."""
        self.population = new_population
        self.speciation.species_set.reset()
        self.speciation.speciate(self.population, self.generation)
        self.available_genomes = self.get_all_genome_ids()

    def update_generation(self, offspring: Dict[int, DefaultGenome]):
        """Incorporate offspring into the population and update generation count."""
        self.generation += 1
        self.population.update(offspring)
        self.update_speciation()
        self.refresh_available_genomes()

    def update_speciation(self):
        """Handle speciation for the current population and generation."""
        self.speciation.speciate(self.population, self.generation)

    def update_genome_data(self, genome_id: int, data: 'UserData'):
        """Update specific data for a given genome."""
        if genome_id in self.population:
            self.population[genome_id].data = data
            return self.get_genome(genome_id)
        else:
            raise ValueError(f"Genome ID {genome_id} not found in the population.")
        
    def refresh_available_genomes(self):
        """
        Refresh the list of available genomes based on the current population.
        """
        self.available_genomes = self.get_all_genome_ids()

    def mark_genome_as_unavailable(self, genome_id: int):
        """
        Mark a genome as unavailable, removing it from the list of available genomes.
        
        :param genome_id: The ID of the genome to mark as unavailable.
        """
        try:
            self.available_genomes.remove(genome_id)
        except ValueError:
            raise ValueError(f"Genome ID {genome_id} is not in the available genomes list.")

    def update_stagnation(self, stagnation: MixedGenerationStagnation, evaluated_ids: List[int]):
        """Evaluate stagnation and collect fitnesses from active genomes."""
        stagnation_mapping = stagnation.update(self.get_species_set(), evaluated_ids, self.generation)
        self.speciation.update_stagnant_species(stagnation_mapping)

    def get_species_set(self) -> MixedGenerationSpeciesSet: 
        """
        Returns the species set containing all species in the current population.
        
        :return: The species set instance.
        """
        return self.speciation.species_set
    
    def get_active_species(self, stagnation, evaluated_ids: List[int]): 
        """
        Returns the active species in the current population.
        
        :param stagnation: The stagnation instance to use for updating species.
        :param evaluated_ids: A list of genome IDs that have been evaluated.
        :return: A list of species instances that are active.
        """
        self.update_stagnation(stagnation, evaluated_ids)
        return self.speciation.species_set.get_active_species()

    def get_genome(self, genome_id: int) -> DefaultGenome:
        """
        Get a genome from the population.
        
        :param genome_id: The ID of the genome to retrieve.
        :return: The genome instance corresponding to the given ID.
        """
        return self.population[genome_id]
    
    def get_all_genome_ids(self) -> List[int]: 
        """
        Get a list of all genome IDs in the population.

        :return: A list of genome IDs.
        """
        return list(self.population.keys())
    
    def get_available(self):
        """
        Get a list of genomes that have not been sent to the client.
        
        :return: A list of genome IDs that are available.
        """
        return self.available_genomes
    
    def get_random_available_genome(self) -> DefaultGenome:
        """
        Send a random member to the user.
        
        :return: A random genome from the available genomes.
        """
        if not self.available_genomes:
            raise RuntimeError("No more genomes to send.")
        genome_id = random.choice(self.available_genomes)
        self.mark_genome_as_unavailable(genome_id)
        return self.get_genome(genome_id)
    
    