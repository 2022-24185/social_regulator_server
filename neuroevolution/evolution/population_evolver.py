"""Implements the core evolution algorithm."""
from typing import Dict, Tuple, TYPE_CHECKING, Any
from pydantic import BaseModel

from neat.config import Config


from neuroevolution.evolution.reproduction import MixedGenerationReproduction
from neuroevolution.evolution.stagnation import MixedGenerationStagnation
from neuroevolution.evolution.speciation import Speciation
from neuroevolution.evolution.evaluation import Evaluation
from neuroevolution.evolution.population_manager import PopulationManager
from neuroevolution.data_models.experiment_data_models import GenerationSummaryData, FitnessStats

if TYPE_CHECKING:
    from neuroevolution.fitness_functions.basic_fitness import BasicFitness
    from neat.genome import DefaultGenome
    from neuroevolution.lab.note_taker import NoteTaker


# Type aliases for better readability
Population = Dict[int, 'DefaultGenome']

class CompleteExtinctionException(Exception):
    """Exception to raise when a population has no members."""
    def __init__(self, message):
        super().__init__(message)


class Evolution:
    """
    Manages the population lifecycle, including fitness evaluation, reproduction, and speciation.
    """

    def __init__(self, config: Config, manager: 'PopulationManager', evaluation: 'Evaluation'):
        #self.reporters = ReporterSet()
        self.reporter: 'NoteTaker' = None
        self.config = config
        self.is_evolving = False
        self.manager = manager # self.create_population_manager()
        self.evaluation = evaluation
        self.stagnation = self.create_stagnation()
        self.reproduction = self.create_reproduction()
        self.speciation = self.create_speciation()
        self.best_genome = None
        self.stop = False
    
    def create_speciation(self) -> Speciation:
        """Create the speciation handler."""
        return self.config.species_set_type(self.config, self.manager)

    def create_stagnation(self) -> MixedGenerationStagnation:
        """Create the stagnation handler."""
        return self.config.stagnation_type(self.config.stagnation_config, self.manager, self.reporter)

    def create_reproduction(self) -> MixedGenerationReproduction:
        """Create the reproduction handler."""
        return self.config.reproduction_type(self.config, self.manager)

    def create_new_population(self) -> Population:
        """
        Create a population from scratch, then partition into species.
        
        :return: The new population.
        """
        self.manager.reset()
        self.reproduction.create_new_genomes(self.config.pop_size)
        self.speciation.speciate()
        self.report_generation_start()
        #print(f"Evaluated genomes: {len(self.manager.genomes.get_evaluated_genomes())}")

    def get_current_generation(self) -> int:
        """Get the current generation."""
        return self.manager.generation

    def return_random_individual(self) -> 'DefaultGenome': 
        """
        Return a random genome from the population.

        :return: A random genome from the population.
        """
        return self.manager.get_random_available_genome()

    def advance_population(self):
        """Advance the population to the next generation, checking for fitness goals."""
        self.is_evolving = True
        stats = self.evaluation.get_fitness_stats()
        self.reporter.post_evaluate(self.get_current_generation(), stats)
        #self.track_best_genome(best_genome)
        if self.fitness_goal_reached(stats.best_genome_fitness):
            print("🎉 Fitness goal reached!")
            self.terminate_evolution()
        else:
            self.report_generation_end()
            self.reproduce_and_update_generation()
        self.manager.genomes.clear_elites()
        self.manager.genomes.clear_evaluated()
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
        self.report_generation_start()

    #def fitness_goal_reached(self, best_genome: 'DefaultGenome') -> bool:
    def fitness_goal_reached(self, best_genome_fitness: float) -> bool:
        """
        Checks if the evolution should terminate based on the fitness criterion.
        
        :param best_genome: The genome with highest fitness in the population.
        :return: True if the fitness threshold has been reached, False otherwise.
        """
        if not self.config.no_fitness_termination:
            if best_genome_fitness >= self.config.fitness_threshold:
                return True
        return False

    # def track_best_genome(self, best_genome: 'DefaultGenome') -> None:
    #     """
    #     Tracks the best genome seen so far.
        
    #     :param best_genome: The genome with the highest fitness.
    #     """
    #     if self.best_genome is None or best_genome.fitness > self.best_genome.fitness:
    #         self.best_genome = best_genome

    def check_and_handle_extinction(self):
        """Check for and handle potential extinction events."""
        if not self.manager.species.get_active_species():
            self.handle_extinction()

    def handle_extinction(self):
        """Handle all species going extinct and possibly reset population."""
        self.reporter.complete_extinction()
        if self.config.reset_on_extinction:
            self.create_new_population()
        else:
            raise CompleteExtinctionException("All species have gone extinct.")
            
    def terminate_evolution(self):
        """Terminate the evolutionary process upon reaching the fitness goal."""
        if self.stop == False: 
            self.stop = True
            self.reporter.found_solution(self.config, self.manager.generation, self.best_genome)

    def add_reporter(self, reporter) -> None:
        """Add a reporter to the set of reporters."""
        self.reporter = reporter

    def get_reporter(self) -> 'NoteTaker':
        """Get the reporter."""
        return self.reporter

    # def start_reporting(self) -> None:
    #     """Start reporting the evolution process."""
        

    def report_generation_start(self): 
        summary = GenerationSummaryData(
            generation=self.manager.generation,
            population_start_size=self.manager.genomes.get_alive_genomes_count(),
            fitness_summary=FitnessStats()
        )
        self.reporter.start_generation(summary)

    def report_generation_end(self):
        """Report the end of a generation to all configured reporters."""
        summary = GenerationSummaryData(
            generation=self.manager.generation,
            population_end_size=len(self.manager.genomes.get_all_genomes()),
            active_species_count=len(self.manager.species.get_active_species()),
        )
        self.reporter.end_generation(summary)