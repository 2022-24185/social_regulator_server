"""Implements the core evolution algorithm."""

from __future__ import print_function
import random, logging
from typing import Dict, Callable, Tuple, Optional, List, TYPE_CHECKING

from neat.reporting import ReporterSet
from neat.math_util import mean
from neat.six_util import itervalues
from neat.genome import DefaultGenome
from neat.config import Config


from neuroevolution.evolution.tournament_reproduction import (
    TournamentReproduction,
)
from neuroevolution.evolution.tournament_species import (
    TournamentSpeciesSet,
)

from neuroevolution.evolution.tournament_stagnation import (
    TournamentStagnation,
)

if TYPE_CHECKING:
    from neuroevolution.server.models import UserData

# Type aliases for better readability
FitnessSummarizer = Callable[[List[float]], float]
Population = Dict[int, DefaultGenome]
SpeciesSet = TournamentSpeciesSet
Stagnation = TournamentStagnation
State = Tuple[Population, SpeciesSet, int]
FitnessFunction = Callable[[Population, Config], None]


class CompleteExtinctionException(Exception):
    """Exception to raise when a population has no members."""
    def __init__(self, message):
        super().__init__(message)

class TournamentPopulation:
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
    """

    def __init__(self, config: Config, fitness_function: FitnessFunction, initial_state: Optional[State] = None, evaluation_threshold: int = 25):
        self.reporters = ReporterSet()
        self.config: Config = config
        self.fitness_function = fitness_function
        self.evaluation_threshold: int = evaluation_threshold
        self.reproduction: TournamentReproduction = self.setup_reproduction(config)
        self.fitness_summarizer: FitnessSummarizer = self.setup_fitness_summarizer(
            config
        )
        self.population: Population
        self.evaluated_genomes: Population = {}  # Store evaluated genomes
        self.species: TournamentSpeciesSet
        self.generation: int
        self.population, self.species, self.generation = self.setup_initial_state(
            config, initial_state
        )
        self.unsent_genomes = list(self.population.keys())  # Store unsent genomes
        self.best_genome: Optional[DefaultGenome] = None

    def setup_reproduction(self, config: Config) -> TournamentReproduction:
        """Setup the reproduction scheme based on the configuration."""
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        return config.reproduction_type(
            config.reproduction_config, self.reporters, stagnation
        )
    
    def get_available_genomes(self): 
        return self.unsent_genomes
    
    def get_evaluated_genomes(self):
        return self.evaluated_genomes

    def setup_fitness_summarizer(self, config: Config) -> FitnessSummarizer:
        """Setup the fitness summarizer based on the configuration."""
        fitness_summarizers = {"max": max, "min": min, "mean": mean}
        if config.fitness_criterion in fitness_summarizers:
            return fitness_summarizers[config.fitness_criterion]
        elif not config.no_fitness_termination:
            raise RuntimeError(
                f"Unexpected fitness_criterion: {config.fitness_criterion!r}"
            )

    def setup_initial_state(
        self, config: Config, initial_state: Optional[State]
    ) -> State:
        """Setup the initial state of the population."""
        if initial_state is None:
            population: Population = self.create_new_population()
            #  species_set_type: Class. Specifies the species set class used, such as species.DefaultSpeciesSet.
            species : SpeciesSet = config.species_set_type(config.species_set_config, self.reporters)
            generation = 0
            logging.info("Initial population created. Speciating...")
            species.set_new_population(population)
            species.speciate(generation, config)
            logging.info(f"Species initialized: {species.species}")
        else:
            population, species, generation = initial_state
        return population, species, generation

    def add_reporter(self, reporter) -> None:
        """Add a reporter to the set of reporters."""
        self.reporters.add(reporter)

    def remove_reporter(self, reporter) -> None:
        """Remove a reporter from the set of reporters."""
        self.reporters.remove(reporter)

    def start_reporting(self) -> None:
        """Start reporting the evolution process."""
        self.reporters.start_generation(self.generation)
    
    def get_random_non_evaluated_member(self) -> DefaultGenome:
        """Send a random member to the user."""
        if not self.unsent_genomes:
            raise RuntimeError("No more genomes to send.")
        genome_id = random.choice(self.unsent_genomes)
        self.unsent_genomes.remove(genome_id)
        return self.population[genome_id]

    def create_new_population(self) -> Population:
        """Create a population from scratch, then partition into species."""
        return self.reproduction.create_new_genomes(
            self.config.genome_type, self.config.genome_config, self.config.pop_size
        )
    
    def advance_population(self) -> None:
        """Advance the population to the next generation."""
        # Evaluate the fitness of the evaluated genomes
        logging.info("\033[91madvancing population...\033[0m")
        best_genome = self.evaluate_fitness(self.fitness_function)
        self.track_best_genome(best_genome)

        # Check if the termination criterion is met
        if self.should_terminate(best_genome):
            print("TERMINATING...")
            self.reporters.found_solution(
                self.config, self.generation, self.best_genome
            )
            return

        # Create the next generation
        print("\033[93mreproducing...\033[0m")
        self.reproduce_evaluated()
        self.reporters.end_generation(self.config, self.population, self.species)
        self.generation += 1

    def receive_evaluation(self, user_data: 'UserData') -> None:
        """Receive an evaluation of a member from the user."""
        logging.info("\033[96mevaluation received\033[0m")
        logging.info(f"Available genomes: {self.get_available_genomes()}")
        if user_data.genome_id == 0: 
            logging.info(f"Genome id is 0, skipping...")
            return
        self.population[user_data.genome_id].data = user_data
        logging.info( f"genome_id: {user_data.genome_id}, data: {user_data.model_dump()}")
        self.evaluated_genomes[user_data.genome_id] = self.population[user_data.genome_id]
        logging.info(f"Evaluated genomes: {self.get_evaluated_genomes()}")
        if self.is_evaluated_threshold_reached():
            #print in green text
            print("\033[92mthreshold reached\033[0m")
            self.advance_population()

    def evaluate_fitness(self, fitness_function: FitnessFunction) -> DefaultGenome:
        """Evaluate the fitness of the entire population."""
        print("\033[91mevaluating fitness...\033[0m")
        fitness_function(self.evaluated_genomes, self.config)
        best = max(itervalues(self.evaluated_genomes), key=lambda g: g.fitness)
        logging.info(f"Best genome: {best.key}, fitness: {best.fitness}")
        self.reporters.post_evaluate(self.config, self.population, self.species, best)
        return best

    def is_evaluated_threshold_reached(self) -> bool:
        """Check if the evaluated threshold is reached."""
        return len(self.evaluated_genomes) >= self.evaluation_threshold

    def reproduce_evaluated(self) -> None:
        """Allow evaluated members to reproduce when a certain threshold is reached."""
        offspring = self.reproduction.reproduce_selected(
            self.config, self.species, self.generation, list(self.evaluated_genomes.keys())
        )
        logging.info(f"\033[93mOffspring: {offspring}\033[0m")
        print("\033[93mdeleting evaluated...\033[0m")
        logging.info(f"Population before deletion: {len(self.population)}")
        for genome_id in self.evaluated_genomes:
            del self.population[genome_id]
        logging.info(f"Population after deletion: {len(self.population)}")
        print("\033[93mupdating population...\033[0m")
        self.population.update(offspring)
        self.evaluated_genomes = {}
        self.unsent_genomes = list(self.population.keys())
        #print all updated fields
        print(f"population: {self.population}, evaluated_genomes: {self.evaluated_genomes}, unsent_genomes: {self.unsent_genomes}")

        # Re-speciate the population
        self.species.set_new_population(self.population)
        self.species.speciate(self.generation, self.config)

        if not self.species.species:
            self.reporters.complete_extinction()
            if self.config.reset_on_extinction:
                self.population = self.create_new_population()
            else:
                raise CompleteExtinctionException(f"Population has been completely extinguished after deleting {len(self.evaluated_genomes.keys())}")

    def track_best_genome(self, best_genome: DefaultGenome) -> None:
        """Tracks the best genome seen so far."""
        print("tracking...")
        if self.best_genome is None or best_genome.fitness > self.best_genome.fitness:
            self.best_genome = best_genome

    def should_terminate(self, best_genome: DefaultGenome) -> bool:
        """Checks if the evolution should terminate based on the fitness criterion."""
        if not self.config.no_fitness_termination:
            fv = self.fitness_summarizer(g.fitness for g in itervalues(self.evaluated_genomes))
            if fv >= self.config.fitness_threshold:
                self.reporters.found_solution(self.config, self.generation, best_genome)
                return True
        return False

