"""This module contains the Experiment class, which is used to run NEAT experiments."""
from typing import Dict, Tuple, List, Any
import random

from neat.nn.recurrent import RecurrentNetwork
from pydantic import BaseModel

from neuroevolution.run_experiments.basic_experiment import BasicExperiment


class AsyncExperiment(BasicExperiment):
    """Class for running experiments with user evaluation."""
    def __init__(self,experiment_config: Dict[str, Any], experiment_id: int):
        super().__init__(experiment_config, experiment_id=experiment_id)
        self.eval_config = self.experiment_config['async_eval']
        self.eval_pool_size = self.eval_config['eval_pool_size']
        self.evaluation_pool: List[Tuple[int, RecurrentNetwork]] = []
        
    def instanciate(self): 
        """Instanciate the experiment."""
        self.evaluation.set_threshold(self.eval_config['eval_threshold'])
        self.evolution.create_new_population()
        for _ in range(self.eval_pool_size-1):
            self.simulate_request()

    def receive_evaluation(self, data: BaseModel):
        super().receive_evaluation(data)

    def run_simulation(self): 
        """Run the simulation."""
        if self.evolution.stop: 
            self.stop()
        else: 
            self.simulate_request()
            received = random.choice(self.evaluation_pool)
            self.evaluation_pool.remove(received)
            data = self.gym.run(received)
            self.receive_evaluation(data)

    def simulate_request(self): 
        """Create a random evaluation request."""
        individual = self.get_random_individual()
        self.evaluation_pool.append(individual)

    def clear_pool(self):
        """Clear the evaluation pool."""
        self.evaluation_pool.clear()
