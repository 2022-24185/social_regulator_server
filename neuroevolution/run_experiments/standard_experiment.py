"""This module contains the Experiment class, which is used to run NEAT experiments."""
from typing import Dict, Tuple, List, Any

from neuroevolution.run_experiments.basic_experiment import BasicExperiment


class StandardExperiment(BasicExperiment):
    """Class for running experiments with user evaluation."""
    def __init__(self, experiment_config: Dict[str, Any]):
        super().__init__(experiment_config)
    
    def instanciate(self): 
        """Instanciate the experiment."""
        self.evolution.create_new_population()
        self.evaluation.set_threshold(self.manager.genomes.get_alive_genomes_count())

    def run_simulation(self): 
        """Run the simulation."""
        if self.evolution.stop: 
            self.stop()
        else: 
            for _ in range(self.manager.genomes.get_alive_genomes_count()): 
                individual = self.get_random_individual()
                data = self.gym.run(individual)
                self.evaluation.process_gym_data(data)
            self.try_evolve()
