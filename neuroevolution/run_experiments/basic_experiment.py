"""Base class for running experiments."""
from typing import Optional, Any, Dict, Tuple
from abc import ABC, abstractmethod
from neat.config import Config
from pydantic import BaseModel

from neuroevolution.fitness_functions.basic_fitness import BasicFitness
from neuroevolution.phenotype.phenotype_creator import PhenotypeCreator
from neuroevolution.evolution.population_evolver import Evolution
from neuroevolution.evolution.population_manager import PopulationManager
from neuroevolution.evolution.evaluation import Evaluation
from neat.reporting import BaseReporter
from neuroevolution.gym.basic_gym import BasicGym
from neuroevolution.server.models import ExperimentData
from neuroevolution.server.models import UserData


class BasicExperiment(ABC): 
    """"Base class for running experiments."""
    def __init__(self, config: Dict[str, Any], experiment_id: int):
        """Initialize the experiment."""
        self.experiment_id = experiment_id
        self.default_config = config
        self.neat_config = self.default_config['neat_config']
        self.class_config = self.default_config['class_config']
        self.experiment_config = self.default_config['experiment_config']
        self.config = self.setup_neat_config()
        self.initialize_experiment_components()
        self.initialize_evolution_components()
        self.stop_experiment = False

    def setup_neat_config(self):
        return Config(
            genome_type=self.neat_config['genome_class'],
            reproduction_type=self.neat_config['reproduction_class'],
            stagnation_type=self.neat_config['stagnation_class'],
            species_set_type=self.neat_config['speciation_class'],
            filename=self.neat_config['config_path']
        )

    def initialize_evolution_components(self): 
        self.phenotype_creator: PhenotypeCreator = self.class_config['phenotype_creator'](self.config, self.gym.input_coords, self.gym.output_coords, self.gym.params)
        self.manager: PopulationManager = self.class_config['population_manager']()
        self.evaluation: Evaluation = self.class_config['evaluation'](self.config, self.fitness_function, self.manager.genomes)
        self.evolution: Evolution = self.class_config['evolution'](self.config, self.manager, self.evaluation)

    def initialize_experiment_components(self):
        self.fitness_function: BasicFitness = self.experiment_config['fitness_class'](self.config)
        self.gym: BasicGym = self.experiment_config['gym_class']()
        self.config.pop_size = self.experiment_config['pop_size']
        self.config.fitness_criterion = self.experiment_config['fitness_criterion']
        self.config.fitness_threshold = self.experiment_config['fitness_threshold']
        self.config.reset_on_extinction = self.experiment_config['reset_on_extinction']
    
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
        self.evolution.add_reporter(reporter)

    def get_reporter(self) -> BaseReporter:
        """Get a reporter by name."""
        return self.evolution.get_reporter()
    
    def try_evolve(self): 
        if self.evaluation.threshold_reached() and not self.evolution.is_evolving: 
            self.evolution.advance_population()
            print(f"🕊️ Free genomes: {len([g.key for g in self.manager.genomes.get_available_genomes()])}")

    def get_random_individual(self) -> Tuple[ExperimentData, Any]:
        """Create a random individual."""
        random_ind = self.evolution.return_random_individual()
        gid = random_ind.key
        network = self.phenotype_creator.create_network_from_genome(random_ind)
        phenotype = (ExperimentData(experiment_id=self.experiment_id, genome_id=gid), network)
        return phenotype
    
    def receive_evaluation(self, data: 'UserData'):
        """Receive and process evaluation data for a genome."""
        print(f"Received evaluation for genome {data.experiment_data.genome_id}")
        self.evaluation.process_gym_data(data)
        self.try_evolve()
    
    def start(self, num_generations: Optional[int] = None):
        """Run the population evolver until it terminates."""
        if num_generations:
            if self.stop_experiment:
                return
            while self.evolution.get_current_generation() < num_generations and not self.stop_experiment:
                self.run_simulation()
        else: 
            while not self.stop_experiment: 
                self.run_simulation()

    def stop(self): 
        """Stop the population evolver."""
        self.stop_experiment = True
        self.evolution.terminate_evolution()

    def reset(self):
        """Reset the population evolver to its initial state."""
        self.stop()
        self.instanciate()
        self.start()
        