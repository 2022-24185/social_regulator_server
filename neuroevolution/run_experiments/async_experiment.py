"""This module contains the Experiment class, which is used to run NEAT experiments."""
from typing import Dict, Tuple, List, Any
import random
import logging

from neat.nn.recurrent import RecurrentNetwork
from pydantic import BaseModel

from neuroevolution.run_experiments.basic_experiment import BasicExperiment
from neuroevolution.server.errors import ExperimentError, PopulationError


class AsyncExperiment(BasicExperiment):
    """Class for running experiments with user evaluation."""
    def __init__(self,experiment_config: Dict[str, Any], experiment_id: int):
        super().__init__(experiment_config, experiment_id=experiment_id)
        self.eval_config = self.experiment_config['async_eval']
        self.eval_pool_size = self.eval_config['eval_pool_size']
        self.evaluation_pool: List[Tuple[int, RecurrentNetwork]] = []
        
    def instanciate(self):
        """Initialize the experiment and start the evaluation pool."""
        try:
            self.evaluation.set_threshold(self.eval_config['eval_threshold'])
            self.evolution.create_new_population()
            for _ in range(self.eval_pool_size):
                self.simulate_request()
            logging.info(f"Experiment {self.experiment_id} instanciated successfully.")
        except Exception as e:
            logging.error(f"Error instanciating experiment: {e}")
            raise ExperimentError(f"Instanciation error: {e}")

    def receive_evaluation(self, data: BaseModel):
        super().receive_evaluation(data)

    def run_simulation(self):
        """Run the simulation and handle evaluation requests."""
        try:
            if self.evolution.stop:
                self.stop()
            else:
                self.simulate_request()
                if self.evaluation_pool:
                    received = random.choice(self.evaluation_pool)
                    self.evaluation_pool.remove(received)
                    data = self.gym.run(received)
                    self.receive_evaluation(data)
        except Exception as e:
            logging.error(f"Error during simulation: {e}")
            raise ExperimentError(f"Simulation error: {e}")

    def simulate_request(self):
        """Generate a random evaluation request."""
        try:
            individual = self.get_random_individual()
            self.evaluation_pool.append(individual)
            logging.info(f"New individual added to evaluation pool in experiment {self.experiment_id}.")
        except PopulationError as e:
            logging.warning(f"Cannot simulate request: {e}")
        except Exception as e:
            logging.error(f"Error during simulate request: {e}")
            raise ExperimentError(f"Simulation request error: {e}")

    def clear_pool(self):
        """Clear the evaluation pool."""
        self.evaluation_pool.clear()
