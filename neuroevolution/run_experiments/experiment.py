"""This module contains the Experiment class, which is used to run NEAT experiments."""
from typing import Dict, Tuple, List, Any
import random
from pydantic import BaseModel

from neat.genome import DefaultGenome
from neat.nn.recurrent import RecurrentNetwork

from neuroevolution.run_experiments.basic_experiment import BasicExperiment
from neuroevolution.evolution.population_evolver import PopulationEvolver
from neuroevolution.evolution.speciation import Speciation
from neuroevolution.evolution.reproduction import MixedGenerationReproduction
from neuroevolution.evolution.stagnation import MixedGenerationStagnation
from neuroevolution.phenotype.phenotype_creator import PhenotypeCreator

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
            stagnation_class=MixedGenerationStagnation, 
            strategy = 'xor'
            )
        self.eval_pool_size = 10
        self.evaluation_pool: List[Tuple[int, RecurrentNetwork]] = []
    
    def create_evolver(self) -> PopulationEvolver:
        """Create a population evolver."""
        return PopulationEvolver(self.config, self.fitness_function, evaluation_threshold=150)
    
    def create_phenotype_creator(self) -> PhenotypeCreator:
        """Create a phenotype creator."""
        return PhenotypeCreator(self.config, self.gym.input_coords, self.gym.output_coords, self.gym.params)
    
    def instanciate(self): 
        """Instanciate the experiment."""
        self.evolver.create_new_population()
        for _ in range(self.eval_pool_size-1):
            self.simulate_user_request()

    def run_simulation(self): 
        """Run the simulation."""
        if self.evolver.stop: 
            self.stop()
        self.simulate_user_request()
        received = random.choice(self.evaluation_pool)
        #print("received is: ", received)
        self.evaluation_pool.remove(received)
        data = self.gym.run(received)
        self.receive_evaluation(data)

    def receive_evaluation(self, data: BaseModel) -> None:
        """Receive and process evaluation data for a genome."""
        self.evolver.handle_receive_user_data(data)

    def get_random_individual(self) -> Tuple[int, RecurrentNetwork]:
        """Create a random individual."""
        random_ind = self.evolver.return_random_individual()
        #print(f"RANDOM IND = {random_ind.key}")
        gid = random_ind.key
        #print(f"GID = {gid}")
        network = self.phenotype_creator.create_network_from_genome(random_ind)
        #print(f"NETWORK = {network}")
        return (gid, network)

    def simulate_user_request(self): 
        """Create a random user request."""
        individual = self.get_random_individual()
        self.evaluation_pool.append(individual)