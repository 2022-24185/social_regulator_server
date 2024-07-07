"""Implements the core evolution algorithm."""
from typing import Dict, Tuple, TYPE_CHECKING, Any
from pydantic import BaseModel

from neat.reporting import ReporterSet

from neat.config import Config


from neuroevolution.evolution.reproduction import MixedGenerationReproduction
from neuroevolution.evolution.stagnation import MixedGenerationStagnation
from neuroevolution.evolution.speciation import Speciation
from neuroevolution.evolution.evaluation import Evaluation
from neuroevolution.evolution.population_manager import PopulationManager


if TYPE_CHECKING:
    from neuroevolution.fitness_functions.basic_fitness import BasicFitness
    from neat.genome import DefaultGenome


# Type aliases for better readability
Population = Dict[int, 'DefaultGenome']

class CompleteExtinctionException(Exception):
    """Exception to raise when a population has no members."""
    def __init__(self, message):
        super().__init__(message)


class PopulationEvolver:
    """
    Manages the population lifecycle, including fitness evaluation, reproduction, and speciation.
    """

    def __init__(self, config: Config, fitness_function: 'BasicFitness', evaluation_threshold: int = 10):
        self.reporters = self.create_reporter_set()
        self.config = config
        self.is_evolving = False
        self.manager = self.create_population_manager()
        self.stagnation = self.create_stagnation()
        self.reproduction = self.create_reproduction()
        self.speciation = self.create_speciation()
        self.evaluation = self.create_evaluation(fitness_function, evaluation_threshold)
        self.best_genome = None
        self.stop = False

    def create_reporter_set(self) -> ReporterSet:
        """Create the reporter set."""
        return ReporterSet()
    
    def create_speciation(self) -> Speciation:
        return Speciation(self.config, self.manager)

    def create_stagnation(self) -> MixedGenerationStagnation:
        """Create the stagnation handler."""
        return self.config.stagnation_type(self.config.stagnation_config, self.manager, self.reporters)

    def create_reproduction(self) -> MixedGenerationReproduction:
        """Create the reproduction handler."""
        return self.config.reproduction_type(self.config, self.manager)

    def create_population_manager(self) -> PopulationManager:
        """Create the population manager."""
        return PopulationManager()

    def create_evaluation(self, fitness_function: 'BasicFitness', evaluation_threshold) -> Evaluation:
        """Create the evaluation handler."""
        return Evaluation(self.config, fitness_function, evaluation_threshold, self.manager.genomes)

    def create_new_population(self) -> Population:
        """
        Create a population from scratch, then partition into species.
        
        :return: The new population.
        """
        self.manager.reset()
        self.reproduction.create_new_genomes(self.config.pop_size)
        self.speciation.speciate()

    def get_current_generation(self) -> int:
        """Get the current generation."""
        return self.manager.generation

    def handle_receive_user_data(self, user_data: BaseModel) -> None:
        """
        Handle user data received from the server.
        
        :param user_data: The user data to process.
        """
        if user_data:
            self.process_user_evaluation(user_data)
        if self.evaluation.threshold_reached() and not self.is_evolving:
            self.advance_population()
            self.manager.genomes.clear_elites()
            self.manager.genomes.clear_evaluated()
            print(f"🕊️ Free genomes: {len([g.key for g in self.manager.genomes.get_available_genomes()])}")

    def process_user_evaluation(self, user_data: BaseModel) -> None:
        """
        Process user evaluations and update genome data.
        
        :param user_data: The user data to process.
        """
        if user_data.genome_id == 0:
            return  # Assume 0 is an invalid ID or a placeholder
        genome = self.manager.update_genome_data(user_data.genome_id, user_data)
        self.evaluation.evaluate(genome)

    def return_random_individual(self) -> 'DefaultGenome': 
        """
        Return a random genome from the population.

        :return: A random genome from the population.
        """
        return self.manager.get_random_available_genome()

    def advance_population(self):
        """Advance the population to the next generation, checking for fitness goals."""
        self.is_evolving = True
        best_genome = self.evaluation.get_best()
        self.track_best_genome(best_genome)
        if self.fitness_goal_reached(best_genome):
            print("🎉 Fitness goal reached!")
            self.terminate_evolution()
        else:
            self.reproduce_and_update_generation()
        self.is_evolving = False

    def reproduce_and_update_generation(self):
        """Manage the reproduction process and update generation information."""        
        # Initial population size
        initial_population_size = len(self.manager.genomes.get_all_genomes())
        print(f"Initial population size: {initial_population_size}")
        self.stagnation.update_active_species()
        self.check_and_handle_extinction()
        self.reproduction.reproduce()
        self.manager.update_generation()
        print(f"🌎 Current generation: {self.manager.generation}")
        self.speciation.speciate()
        self.report_generation_end()

    def fitness_goal_reached(self, best_genome: 'DefaultGenome') -> bool:
        """
        Checks if the evolution should terminate based on the fitness criterion.
        
        :param best_genome: The genome with highest fitness in the population.
        :return: True if the fitness threshold has been reached, False otherwise.
        """
        if not self.config.no_fitness_termination:
            if best_genome.fitness >= self.config.fitness_threshold:
                return True
        return False

    def track_best_genome(self, best_genome: 'DefaultGenome') -> None:
        """
        Tracks the best genome seen so far.
        
        :param best_genome: The genome with the highest fitness.
        """
        if self.best_genome is None or best_genome.fitness > self.best_genome.fitness:
            self.best_genome = best_genome

    def check_and_handle_extinction(self):
        """Check for and handle potential extinction events."""
        if not self.manager.species.get_active_species():
            self.handle_extinction()

    def handle_extinction(self):
        """Handle all species going extinct and possibly reset population."""
        self.reporters.complete_extinction()
        if self.config.reset_on_extinction:
            self.create_new_population()
        else:
            raise CompleteExtinctionException("All species have gone extinct.")
            
    def terminate_evolution(self):
        """Terminate the evolutionary process upon reaching the fitness goal."""
        if self.stop == False: 
            self.stop = True
            self.reporters.found_solution(self.config, self.manager.generation, self.best_genome)

    def add_reporter(self, reporter) -> None:
        """Add a reporter to the set of reporters."""
        self.reporters.add(reporter)

    def remove_reporter(self, reporter) -> None:
        """Remove a reporter from the set of reporters."""
        self.reporters.remove(reporter)

    def start_reporting(self) -> None:
        """Start reporting the evolution process."""
        self.reporters.start_generation(self.manager.generation)

    def report_generation_end(self):
        """Report the end of a generation to all configured reporters."""
        self.reporters.end_generation(self.config, self.manager.genomes, self.manager.species)