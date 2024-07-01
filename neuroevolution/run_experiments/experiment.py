"""This module contains the Experiment class, which is used to run NEAT experiments."""
from typing import Dict, Tuple, Optional
import random

from neat.genome import DefaultGenome
from neat.nn.recurrent import RecurrentNetwork

from neuroevolution.server.models import UserData
from neuroevolution.run_experiments.basic_experiment import BasicExperiment
from neuroevolution.evolution.population_evolver import PopulationEvolver
from neuroevolution.evolution.speciation import Speciation
from neuroevolution.evolution.reproduction import MixedGenerationReproduction
from neuroevolution.evolution.stagnation import MixedGenerationStagnation
from neuroevolution.evolution.fitness_functions.user_evaluated_fitness import UserEvaluatedFitness
from neuroevolution.evolution.phenotype_creator import PhenotypeCreator

# Type aliases
Genome = DefaultGenome
Population = Dict[int, DefaultGenome]

class SimulatedUserEvalExperiment(BasicExperiment):
    """Class for running experiments with user evaluation."""
    def __init__(self, config_path):
        super().__init__(
            config_path, 
            genome_class=DefaultGenome, 
            reproduction_class=MixedGenerationReproduction, 
            speciation_class=Speciation, 
            stagnation_class=MixedGenerationStagnation
            )        
        self.iids_in_evaluation = []

    def create_fitness_function(self) -> UserEvaluatedFitness:
        """Create a fitness function."""
        return UserEvaluatedFitness(self.config)
    
    def create_evolver(self) -> PopulationEvolver:
        """Create a population evolver."""
        return PopulationEvolver(self.config, self.fitness_function)
    
    def create_phenotype_creator(self) -> PhenotypeCreator:
        return PhenotypeCreator(self.config)
    
    def instanciate(self): 
        self.evolver.create_new_population()
        for i in range(10):
            self.simulate_user_request()

    def run_simulation(self): 
        self.simulate_user_request()
        data = self.simulate_user_evaluation()
        self.receive_evaluation(data)

    def receive_evaluation(self, data: UserData):
        """Receive and process evaluation data for a genome."""
        self.evolver.handle_receive_user_data(data)

    def get_random_individual(self) -> Tuple[int, RecurrentNetwork]:
        """Create a random individual."""
        random_ind = self.evolver.return_random_individual()
        gid = random_ind.key
        return (gid, self.phenotype_creator.create_network_from_genome(random_ind))
    
    def simulate_user_evaluation(self) -> UserData:
        """Create a random user evaluation within a range."""
        received = random.choice(self.iids_in_evaluation)
        print(f"iids in evaluation: {self.iids_in_evaluation}")
        self.iids_in_evaluation.remove(received)
        return UserData(
            genome_id = received,
            time_since_startup = random.randint(0, 1000),
            user_rating = random.randint(0, 5),
            last_message = None,
            last_message_time = None,
            last_response = None,
            last_response_time = None,
        )

    def simulate_user_request(self): 
        iid = (self.get_random_individual())[0]
        self.iids_in_evaluation.append(iid)

    


