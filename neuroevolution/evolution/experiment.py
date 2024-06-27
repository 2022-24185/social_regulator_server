"""This module contains the Experiment class, which is used to run NEAT experiments."""
from typing import Dict, Tuple

from neat.genome import DefaultGenome
from neat.config import Config
from neat.nn.recurrent import RecurrentNetwork

from neuroevolution.evolution.population_evolver import PopulationEvolver
from neuroevolution.evolution.speciation import Speciation
from neuroevolution.evolution.reproduction import MixedGenerationReproduction
from neuroevolution.evolution.stagnation import MixedGenerationStagnation
from neuroevolution.evolution.fitness_functions.user_evaluated_fitness import UserEvaluatedFitness
from neuroevolution.evolution.phenotype_creator import PhenotypeCreator

# Type aliases
Genome = DefaultGenome
Population = Dict[int, DefaultGenome]

class Experiment:
    def __init__(self, config_path):
        self.config_path = config_path
        self.population = None
        self.config = Config(
            DefaultGenome,
            MixedGenerationReproduction,
            Speciation,
            MixedGenerationStagnation,
            self.config_path
        )
        self.create_phenotype_creator()

    def create_phenotype_creator(self):
        """Create a phenotype creator."""
        self.phenotype_creator = PhenotypeCreator(self.config)
        
    def start(self):
        """Start the population evolver."""
        self.population = PopulationEvolver(self.config, self.get_fitness_function)

    def receive_evaluation(self, data):
        """Receive and process evaluation data for a genome."""
        self.population.process_user_evaluation(data)

    def get_random_individual(self) -> Tuple[int, RecurrentNetwork]:
        """Create a random individual."""
        random_ind = self.population.return_random_individual()
        gid = random_ind.key
        return (gid, self.phenotype_creator.create_network_from_genome(random_ind))

    def get_fitness_function(self):
        """Define or get the fitness function used by the population."""
        return UserEvaluatedFitness

    def reset(self):
        """Reset the population evolver to its initial state."""
        self.population = PopulationEvolver(self.config, self.get_fitness_function)
