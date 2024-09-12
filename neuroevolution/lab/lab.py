from typing import List, Dict, Any, Tuple
from itertools import product
from copy import deepcopy
import random
import logging

from neat.reporting import ReporterSet

from neuroevolution.run_experiments.basic_experiment import BasicExperiment
from neuroevolution.lab.note_taker import NoteTaker
# from neuroevolution.lab.visualiser import ExperimentVisualizer
from neuroevolution.server.models import ExperimentData, UserData, PhenotypeData, FullExperimentConfig

# pylint: disable=logging-fstring-interpolation

class LabError(Exception):
    """Base class for all lab-related errors."""
    pass

class ExperimentNotFoundError(LabError):
    """Raised when an experiment with a given ID does not exist."""
    def __init__(self, experiment_id):
        super().__init__(f"Experiment {experiment_id} does not exist.")

class ExperimentCreationError(LabError):
    """Raised when an experiment cannot be created."""
    def __init__(self, message="Failed to create experiment"):
        super().__init__(message)

class ParameterizationError(LabError):
    """Raised when there's an error in parameter configuration."""
    def __init__(self, message="Error in experiment parameterization"):
        super().__init__(message)

class EvaluationError(LabError):
    """Raised when there's an error in returning or processing evaluation data."""
    def __init__(self, experiment_id):
        super().__init__(f"Error processing evaluation for experiment {experiment_id}.")

class Lab:
    def __init__(self):
        self.analyst = None
        self.visualiser = None
        self.reporters = ReporterSet()
        self.experiments: Dict[int, BasicExperiment] = {}
        self.next_experiment_id = 0
        logging.info("Lab initialized.")

    def add_simple_experiment(self, config: FullExperimentConfig):
        """Add a simple experiment to the lab."""
        try:
            experiment = self._create_experiment(config)
            self.experiments[experiment.experiment_id] = experiment
            reporter = NoteTaker(experiment.experiment_id)
            self.reporters.add(reporter)
            experiment.add_reporter(reporter)
            logging.info(f"Experiment {experiment.experiment_id} added.")
        except Exception as e:
            logging.error(f"Error adding experiment: {e}")
            raise ExperimentCreationError()

    def sample_random_experiment(self) -> 'PhenotypeData':
        """Sample a genome from a random experiment."""
        if not self.experiments:
            logging.error("No experiments available to sample from.")
            raise LabError("No experiments available to sample from.")
        
        try:
            experiment_id = random.choice(list(self.experiments.keys()))
            experiment = self.experiments[experiment_id]
            logging.info(f"Sampling individual from experiment {experiment_id}.")
            return experiment.get_random_individual()
        except Exception as e:
            logging.error(f"Error sampling random experiment: {e}")
            raise LabError("Failed to sample a random experiment.")

    def sample_individual_from_experiment(self, experiment_id: int) -> Tuple[ExperimentData, Any]:
        """Sample a genome from a specific experiment."""
        if experiment_id not in self.experiments:
            logging.error(f"Experiment {experiment_id} not found.")
            raise ExperimentNotFoundError(experiment_id)

        try:
            experiment = self.experiments[experiment_id]
            logging.info(f"Sampling individual from experiment {experiment_id}.")
            return experiment.get_random_individual()
        except Exception as e:
            logging.error(f"Error sampling from experiment {experiment_id}: {e}")
            raise LabError(f"Failed to sample from experiment {experiment_id}")
    
    def return_individual_to_experiment(self, data: 'UserData'):
        """Return an individual to the experiment for evaluation."""
        experiment_id = data.experiment_data.experiment_id
        if experiment_id not in self.experiments:
            logging.error(f"Experiment {experiment_id} not found.")
            raise ExperimentNotFoundError(experiment_id)

        try:
            experiment = self.experiments[experiment_id]
            experiment.receive_evaluation(data)
            logging.info(f"Evaluation returned to experiment {experiment_id}.")
        except Exception as e:
            logging.error(f"Error processing evaluation for experiment {experiment_id}: {e}")
            raise EvaluationError(experiment_id) from e

    def add_parameterized_experiment(self, base_config: FullExperimentConfig, parameters: Dict[str, List[Any]]):
        """Add parameterized experiments based on a base config and multiple parameters."""
        try:
            param_combinations = self._generate_param_combinations(parameters)
            for combination in param_combinations:
                combined_config = self._merge_configs(base_config, combination)
                self.add_simple_experiment(combined_config)
                logging.info(f"Added parameterized experiment with parameters: {combination}")
        except Exception as e:
            logging.error(f"Error parameterizing experiment: {e}")
            raise ParameterizationError() from e

    def add_comparison_experiment(self, experiment_a_config: FullExperimentConfig, experiment_b_config: FullExperimentConfig):
        """Add two experiments for comparison."""
        try:
            self.add_simple_experiment(experiment_a_config)
            self.add_simple_experiment(experiment_b_config)
            logging.info("Comparison experiments added.")
        except ExperimentCreationError as e:
            logging.error(f"Failed to create comparison experiment: {e}")
            raise

    def instanciate_experiments(self):
        """Instantiate all experiments."""
        try:
            for experiment in self.experiments.values():
                experiment.instanciate()
                logging.info(f"Instanciated experiment {experiment.experiment_id}.")
        except Exception as e:
            logging.error(f"Error instantiating experiments: {e}")
            raise LabError("Failed to instantiate all experiments.") from e

    def run_scenarios(self):
        """Run all experiments for their configured number of generations."""
        try:
            for experiment in self.experiments.values():
                num_gens = experiment.experiment_config['num_generations']
                eval_threshold = experiment.experiment_config['async_eval']['eval_threshold']
                logging.info(f"Running experiment {experiment.experiment_id} for {num_gens} generations with threshold {eval_threshold}.")
                experiment.start(num_gens)
                rep: NoteTaker = experiment.get_reporter()
                datamodels = rep.get_data()
                # Visualization logic or further data processing can go here.
        except Exception as e:
            logging.error(f"Error running scenarios: {e}")
            raise LabError("Failed to run all scenarios.") from e
        
    def get_experiment_statuses(self) -> Dict[int, Dict[str, Any]]:
        """Fetch the statuses of all running experiments."""
        statuses = {}
        for experiment_id, experiment in self.experiments.items():
            try:
                reporter = experiment.get_reporter()
                statuses[experiment_id] = reporter.get_data()
            except Exception as e:
                logging.error(f"Error fetching experiment status for {experiment_id}: {e}")
                raise LabError(f"Failed to get status for experiment {experiment_id}") from e
        return statuses

    def get_experiment_status(self, experiment_id: int) -> Dict[str, Any]:
        """Fetch the status of a running experiment."""
        if experiment_id not in self.experiments:
            logging.error(f"Experiment {experiment_id} not found.")
            raise ExperimentNotFoundError(experiment_id)

        try:
            reporter = self.experiments[experiment_id].get_reporter()
            return reporter.get_data()
        except Exception as e:
            logging.error(f"Error fetching experiment status for {experiment_id}: {e}")
            raise LabError(f"Failed to get status for experiment {experiment_id}")
        
    def reset_experiment(self):
        """Reset all experiments and clear the population."""
        try:
            logging.info("Resetting all experiments and clearing population.")
            for experiment in self.experiments.values():
                experiment.reset()  # Assuming BasicExperiment has a reset method
            logging.info("All experiments reset successfully.")
        except Exception as e:
            logging.error(f"Error resetting experiments: {e}")
            raise LabError("Failed to reset experiments.")

    # ---------- Private Methods ---------- #

    def _create_experiment(self, config: FullExperimentConfig) -> BasicExperiment:
        """Create and return a new experiment based on the given configuration."""
        try:
            experiment_class = config.experiment_config.experiment_class
            experiment = experiment_class(config, self.next_experiment_id)
            self.next_experiment_id += 1
            return experiment
        except Exception as e:
            logging.error(f"Error creating experiment: {e}")
            raise ExperimentCreationError()

    def _generate_param_combinations(self, parameters: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate combinations of parameters."""
        try:
            keys, values = zip(*parameters.items())
            combinations = [dict(zip(keys, v)) for v in product(*values)]
            return combinations
        except Exception as e:
            logging.error(f"Error generating parameter combinations: {e}")
            raise ParameterizationError()
    
    def _merge_configs(self, base_config: FullExperimentConfig, param_config: Dict[str, Any]) -> FullExperimentConfig:
        """Merge the base config with parameterized config using Pydantic's copy method."""
        try:
            return base_config.model_copy(update=param_config)
        except Exception as e:
            logging.error(f"Error merging configurations: {e}")
            raise ParameterizationError()

    # def _merge_configs(self, base_config: Dict[str, Any], param_config: Dict[str, Any]) -> Dict[str, Any]:
    #     """Merge the base config with parameterized config."""
    #     merged_config = deepcopy(base_config)

    #     def update_nested_dict(d, key_path, value):
    #         current = d
    #         for k in key_path[:-1]:
    #             current = current.setdefault(k, {})
    #         current[key_path[-1]] = value

    #     for key, value in param_config.items():
    #         if '.' in key:
    #             key_path = key.split('.')
    #             update_nested_dict(merged_config, key_path, value)
    #         else:
    #             merged_config[key] = value
    #     return merged_config


    