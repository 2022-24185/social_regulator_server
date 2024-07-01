"""Base class for running experiments."""
from typing import Optional
from abc import ABC, abstractmethod
from neat.config import Config

from neuroevolution.evolution.population_evolver import PopulationEvolver

class BasicExperiment(ABC): 
    """"Base class for running experiments."""
    def __init__(self, config_path, genome_class, reproduction_class, speciation_class, stagnation_class):
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
        self.fitness_function = self.create_fitness_function()
        self.evolver = self.create_evolver()
        self.phenotype_creator = self.create_phenotype_creator()

    @abstractmethod
    def create_evolver(self) -> PopulationEvolver:
        """Create a population evolver."""
        raise NotImplementedError()
    
    @abstractmethod
    def create_phenotype_creator(self):
        """Create a phenotype creator."""
        raise NotImplementedError()
    
    @abstractmethod
    def create_fitness_function(self):
        """Create a fitness function."""
        raise NotImplementedError()
    
    @abstractmethod
    def run_simulation(self):
        """Run a simulation."""
        raise NotImplementedError()

    @abstractmethod
    def instanciate(self):
        """Create a new population."""
        raise NotImplementedError()

    def start(self, num_generations: Optional[int] = None):
        """Run the population evolver until it terminates."""
        self.instanciate()
        if num_generations:
            if self.stop_experiment:
                return
            while self.evolver.get_current_generation() < num_generations:
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
        