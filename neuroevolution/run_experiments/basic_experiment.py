"""Base class for running experiments."""
from typing import Optional, Any, Dict, Tuple
from abc import ABC, abstractmethod
from neat.config import Config
import logging

from neuroevolution.lab.note_taker import NoteTaker, BaseReporter
from neuroevolution.server.models import ExperimentData, UserData, FullExperimentConfig
# pylint: disable=logging-fstring-interpolation

# Custom Exception Classes
class InitializationError(Exception):
    """Raised when there is an error initializing components of the experiment."""
    pass

class EvolutionError(Exception):
    """Raised when there is an error during the evolutionary process."""
    pass

class ConfigHelper:
    def __init__(self, config: FullExperimentConfig):
        self.config = config

    def setup_neat_config(self) -> Config:
        neat_config = self.config.neat_config
        try:
            neat_config = Config(
                genome_type=neat_config.genome_class,
                reproduction_type=neat_config.reproduction_class,
                stagnation_type=neat_config.stagnation_class,
                species_set_type=neat_config.speciation_class,
                filename=neat_config.config_path
            )
            # Override NEAT config with experiment config if necessary
            neat_config.pop_size = self.config.experiment_config.pop_size
            neat_config.fitness_criterion = self.config.experiment_config.fitness_criterion
            neat_config.fitness_threshold = self.config.experiment_config.fitness_threshold
            neat_config.reset_on_extinction = self.config.experiment_config.reset_on_extinction
            return neat_config
        except KeyError as e:
            logging.error(f"Error setting up NEAT config: missing key {e}")
            raise InitializationError(f"Missing key in NEAT config: {e}") from e

    def get_experiment_config(self):
        return self.config.experiment_config

    def get_class_config(self):
        return self.config.class_config
    
    def get_phenotype_config(self):
        return self.config.phenotype_config


class BasicExperiment(ABC): 
    """"Base class for running experiments."""
    def __init__(self, config: FullExperimentConfig, experiment_id: int):
        """Initialize the experiment."""
        self.experiment_id = experiment_id
        self.config_helper = ConfigHelper(config)
        self.stop_experiment = False

        try:
            # Initialize configuration and components
            self.config = self.config_helper.setup_neat_config()
            self.experiment_config = self.config_helper.get_experiment_config()
            self.class_config = self.config_helper.get_class_config()
            self.phenotype_config = self.config_helper.get_phenotype_config()

            # Initialize components for experiment and evolution
            self.initialize_experiment_components()
            self.initialize_evolution_components()
        except InitializationError as e:
            logging.error(f"Initialization failed for experiment {experiment_id}: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during initialization: {e}")
            raise InitializationError(f"Unexpected error: {e}") from e

    def initialize_experiment_components(self):
        """Initialize experiment-specific components."""
        try:
            self.fitness_function = self.experiment_config.fitness_class(self.config)
            self.gym = self.experiment_config.gym_class()
            logging.info(f"Experiment components initialized successfully for experiment {self.experiment_id}.")
        except Exception as e:
            logging.error(f"Error initializing experiment components for experiment {self.experiment_id}: {e}")
            raise InitializationError(f"Failed to initialize experiment components: {e}") from e

    def initialize_evolution_components(self): 
        """Initialize components related to the evolutionary algorithm."""
        try:
            self.phenotype_creator = self.class_config.phenotype_creator(self.config, self.phenotype_config)
            self.manager = self.class_config.population_manager()
            self.evaluation = self.class_config.evaluation(self.config, self.fitness_function, self.manager.genomes)
            self.evolution = self.class_config.evolution(self.config, self.manager, self.evaluation)
            logging.info(f"Evolution components initialized successfully for experiment {self.experiment_id}.")
        except Exception as e:
            logging.error(f"Error initializing evolution components for experiment {self.experiment_id}: {e}")
            raise InitializationError(f"Failed to initialize evolution components: {e}") from e
    
    @abstractmethod
    def run_simulation(self):
        """Run a simulation."""
        raise NotImplementedError()

    @abstractmethod
    def instanciate(self):
        """Create a new population."""
        raise NotImplementedError()
    
    def add_reporter(self, reporter: BaseReporter):
        """Add reporters to the experiment."""
        try:
            self.evolution.add_reporter(reporter)
            logging.info(f"Reporter added to experiment {self.experiment_id}.")
        except Exception as e:
            logging.error(f"Error adding reporter to experiment {self.experiment_id}: {e}")
            raise EvolutionError(f"Failed to add reporter: {e}") from e

    def get_reporter(self) -> NoteTaker:
        """Get a reporter by name."""
        try:
            return self.evolution.get_reporter()
        except Exception as e:
            logging.error(f"Error retrieving reporter for experiment {self.experiment_id}: {e}")
            raise EvolutionError(f"Failed to retrieve reporter: {e}") from e
    
    def try_evolve(self): 
        try:
            if self.evaluation.threshold_reached() and not self.evolution.is_evolving: 
                self.evolution.advance_population()
                logging.info(f"Population advanced for experiment {self.experiment_id}.")
        except Exception as e:
            logging.error(f"Error during population evolution for experiment {self.experiment_id}: {e}")
            raise EvolutionError(f"Population evolution failed: {e}") from e

    def get_random_individual(self) -> Tuple[ExperimentData, Any]:
        """Create a random individual."""
        try:
            random_ind = self.evolution.return_random_individual()
            gid = random_ind.key
            network = self.phenotype_creator.create_network_from_genome(random_ind)
            phenotype = (ExperimentData(experiment_id=self.experiment_id, genome_id=gid), network)
            return phenotype
        except Exception as e:
            logging.error(f"Error retrieving random individual for experiment {self.experiment_id}: {e}")
            raise EvolutionError(f"Failed to retrieve random individual: {e}") from e
    
    def receive_evaluation(self, data: 'UserData'):
        """Receive and process evaluation data for a genome."""
        try:
            logging.info(f"Received evaluation for genome {data.experiment_data.genome_id}.")
            self.evaluation.process_gym_data(data)
            self.try_evolve()
        except Exception as e:
            logging.error(f"Error processing evaluation for genome {data.experiment_data.genome_id}: {e}")
            raise EvolutionError(f"Failed to process evaluation data for genome {data.experiment_data.genome_id}: {e}") from e
    
    def start(self, num_generations: Optional[int] = None):
        """Run the population evolver until it terminates."""
        try:
            if num_generations:
                if self.stop_experiment:
                    return
                while self.evolution.get_current_generation() < num_generations and not self.stop_experiment:
                    self.run_simulation()
            else: 
                while not self.stop_experiment: 
                    self.run_simulation()
        except Exception as e:
            logging.error(f"Error running experiment {self.experiment_id}: {e}")
            raise EvolutionError(f"Failed to run experiment {self.experiment_id}: {e}") from e

    def stop(self): 
        """Stop the population evolver."""
        try:
            self.stop_experiment = True
            self.evolution.terminate_evolution()
            logging.info(f"Experiment {self.experiment_id} stopped.")
        except Exception as e:
            logging.error(f"Error stopping experiment {self.experiment_id}: {e}")
            raise EvolutionError(f"Failed to stop experiment {self.experiment_id}: {e}") from e

    def reset(self):
        """Reset the population evolver to its initial state."""
        try:
            self.stop()
            self.instanciate()
            self.start()
            logging.info(f"Experiment {self.experiment_id} reset.")
        except Exception as e:
            logging.error(f"Error resetting experiment {self.experiment_id}: {e}")
            raise EvolutionError(f"Failed to reset experiment {self.experiment_id}: {e}")
        