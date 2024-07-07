"""Base class for running experiments."""
from typing import Optional
from abc import ABC, abstractmethod
from neat.config import Config
import random

from neuroevolution.fitness_functions.basic_fitness import BasicFitness
from neuroevolution.fitness_functions.user_evaluated_fitness import UserEvaluatedFitness
from neuroevolution.fitness_functions.xor_fitness import XORFitness
from neuroevolution.run_experiments.basic_gym import BasicGym
from neuroevolution.run_experiments.chat_gym import ChatGym
from neuroevolution.run_experiments.xor_gym import XORGym
from neuroevolution.phenotype.phenotype_creator import PhenotypeCreator
from neuroevolution.evolution.population_evolver import PopulationEvolver

class BasicExperiment(ABC): 
    """"Base class for running experiments."""
    def __init__(self, config_path, genome_class, reproduction_class, speciation_class, stagnation_class, strategy: str):
        """Initialize the experiment."""
        self.config_path = config_path
        self.config = Config(
            genome_class,
            reproduction_class,
            speciation_class,
            stagnation_class,
            self.config_path
        )
        self.stop_experiment = False
        self.gym = self.create_gym(strategy)
        self.fitness_function = self.create_fitness_function(strategy)
        self.evolver = self.create_evolver()
        self.phenotype_creator = self.create_phenotype_creator()

    @abstractmethod
    def create_evolver(self) -> PopulationEvolver:
        """Create a population evolver."""
        raise NotImplementedError()
    
    @abstractmethod
    def create_phenotype_creator(self) -> PhenotypeCreator:
        """Create a phenotype creator."""
        raise NotImplementedError()
    
    @abstractmethod
    def run_simulation(self):
        """Run a simulation."""
        raise NotImplementedError()

    @abstractmethod
    def instanciate(self):
        """Create a new population."""
        raise NotImplementedError()
    
    def create_gym(self, strategy: str = 'chat') -> BasicGym: 
        """Create a gym environment."""
        if strategy == 'chat':
            return ChatGym()
        elif strategy == 'xor':
            return XORGym()

    def create_fitness_function(self, strategy: str = 'chat') -> BasicFitness:
        """Create a fitness function."""
        if strategy == 'chat':
            return UserEvaluatedFitness(self.config)
        elif strategy == 'xor':
            return XORFitness(self.config)
    
    def start(self, num_generations: Optional[int] = None):
        """Run the population evolver until it terminates."""
        self.instanciate()
        if num_generations:
            if self.stop_experiment:
                return
            while self.evolver.get_current_generation() < num_generations and not self.stop_experiment:
                self.run_simulation()
        else: 
            while not self.stop_experiment: 
                self.run_simulation()

    def stop(self): 
        """Stop the population evolver."""
        self.stop_experiment = True
        self.evolver.terminate_evolution()

    def reset(self):
        """Reset the population evolver to its initial state."""
        self.stop()
        self.instanciate()
        self.start()
        