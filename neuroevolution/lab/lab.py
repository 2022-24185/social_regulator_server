from typing import List, Dict, Any, Tuple
from itertools import product
from copy import deepcopy
import random

from neat.reporting import ReporterSet

from neuroevolution.run_experiments.basic_experiment import BasicExperiment
from neuroevolution.lab.note_taker import NoteTaker
# from neuroevolution.lab.visualiser import ExperimentVisualizer
from neuroevolution.server.models import ExperimentData, UserData, PhenotypeData


class Lab:
    def __init__(self):
        self.analyst = None
        self.visualiser = None
        self.reporters = ReporterSet()
        self.experiments: Dict[int, BasicExperiment] = {}
        self.next_experiment_id = 0

    def add_simple_experiment(self, config: Dict[str, Any]):
        """Add a simple experiment to the lab."""
        experiment = self._create_experiment(config)
        self.experiments.update({experiment.experiment_id: experiment})
        reporter = NoteTaker(experiment.experiment_id)
        self.reporters.add(reporter)
        experiment.add_reporter(reporter)

    def sample_random_experiment(self) -> 'PhenotypeData':
        """Sample a genome from a random experiment."""
        experiment_id = random.choice(list(self.experiments.keys()))
        experiment = self.experiments[experiment_id]
        individual = experiment.get_random_individual()
        return individual

    def sample_individual_from_experiment(self, experiment_id: int) -> Tuple[ExperimentData, Any]:
        """Sample a genome from the experiment."""
        experiment = self.experiments[experiment_id]
        individual = experiment.get_random_individual()
        return individual
    
    def return_individual_to_experiment(self, data: 'UserData'):
        """Return an individual to the experiment."""
        experiment = self.experiments[data.experiment_data.experiment_id]
        experiment.receive_evaluation(data)

    def add_parameterized_experiment(self, base_config: Dict[str, Any], parameters: Dict[str, List[Any]]):
        """Add a parameterized experiment to the lab."""
        param_combinations = self._generate_param_combinations(parameters)
        for combination in param_combinations:
            print(f"👾 combination is {combination}")
            combined_config = self._merge_configs(base_config, combination)
            self.add_simple_experiment(combined_config)

    def add_comparison_experiment(self, experiment_a_config: Dict[str, Any], experiment_b_config: Dict[str, Any]):
        """Add a comparison experiment to the lab."""
        self.add_simple_experiment(experiment_a_config)
        self.add_simple_experiment(experiment_b_config)

    def instanciate_experiments(self):
        for experiment in self.experiments.values():
            experiment.instanciate()
            print(f"🧪 Instanciated experiment {experiment.experiment_id}")

    def run_scenarios(self):
        for scenario in self.experiments.values():
            num_gens = scenario.experiment_config['num_generations']
            eval_threshold = scenario.experiment_config['async_eval']['eval_threshold']
            print(f"📒📒📒 Running experiment {scenario.experiment_id} for {num_gens} generations with eval threshold {eval_threshold}")
            scenario.start(num_gens)
            rep: NoteTaker = scenario.get_reporter()
            datamodels = rep.get_data()
            # for model in datamodels: 
            #     viz = ExperimentVisualizer(model)
            #     viz.visualize_fitness_stats()
            #     viz.visualize_generation_summary()

    def _create_experiment(self, config: Dict[str, Any]) -> BasicExperiment:
        experiment_class = config['experiment_config']['experiment_class']
        experiment = experiment_class(config, self.next_experiment_id)
        self.next_experiment_id += 1
        return experiment

    def _generate_param_combinations(self, parameters: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        keys, values = zip(*parameters.items())
        combinations = [dict(zip(keys, v)) for v in product(*values)]
        return combinations

    def _merge_configs(self, base_config: Dict[str, Any], param_config: Dict[str, Any]) -> Dict[str, Any]:
        merged_config = deepcopy(base_config)

        def update_nested_dict(d, key_path, value):
            current = d
            for k in key_path[:-1]:  # Traverse all keys but the last
                current = current.setdefault(k, {})  # Get or create nested dict
            current[key_path[-1]] = value  # Set the value in the deepest dict

        for key, value in param_config.items():
            if '.' in key:  # Handle dot notation for nested updates
                key_path = key.split('.')
                update_nested_dict(merged_config, key_path, value)
            elif isinstance(value, dict) and key in merged_config:
                merged_config[key].update(value)
            else:
                merged_config[key] = value
        return merged_config


    