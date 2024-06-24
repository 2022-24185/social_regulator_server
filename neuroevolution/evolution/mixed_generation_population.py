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
    MixedGenerationReproduction,
)
from neuroevolution.evolution.mixed_generation_species import (
    MixedGenerationSpeciesSet,
)

from neuroevolution.evolution.tournament_stagnation import (
    TournamentStagnation,
)

if TYPE_CHECKING:
    from neuroevolution.server.models import UserData

# Type aliases for better readability
FitnessSummarizer = Callable[[List[float]], float]
Population = Dict[int, DefaultGenome]
SpeciesSet = MixedGenerationSpeciesSet
Stagnation = TournamentStagnation
State = Tuple[Population, SpeciesSet, int]
FitnessFunction = Callable[[Population, Config], None]


class CompleteExtinctionException(Exception):
    """Exception to raise when a population has no members."""
    def __init__(self, message):
        super().__init__(message)

class MixedGenerationPopulation:
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
    """

    def __init__(self, config: Config, fitness_function: FitnessFunction, evaluation_threshold: int = 50, initial_state: Optional[State] = None):
        self.reporters = ReporterSet()
        self.config = config
        self.evaluation_threshold = evaluation_threshold
        self.fitness_function = fitness_function
        self.reproduction = self.setup_reproduction()
        self.population, self.species, self.generation = self.setup_initial_state(initial_state)
        self.fitness_summarizer = self.setup_fitness_summarizer()
        self.evaluated_genomes = {}
        self.available_genomes = list(self.population.keys())  # Cache unsent genomes
        self.best_genome = None

    def setup_reproduction(self) -> MixedGenerationReproduction:
        """Instantiate reproduction strategy with configured stagnation."""
        stagnation = self.config.stagnation_type(self.config.stagnation_config, self.reporters)
        return self.config.reproduction_type(self.config.reproduction_config, self.reporters, stagnation)

    def setup_fitness_summarizer(self) -> FitnessSummarizer:
        """Choose a fitness summarizer based on configuration, ensuring all conditions are handled."""
        fitness_summarizers = {"max": max, "min": min, "mean": mean}
        criterion = self.config.fitness_criterion
        if criterion in fitness_summarizers:
            return fitness_summarizers[criterion]
        if not self.config.no_fitness_termination:
            raise RuntimeError(f"Unexpected fitness_criterion: {criterion!r}")

    def setup_initial_state(self, initial_state: Optional[State]) -> State:
        """Initialize or reuse state based on input configuration."""
        if initial_state:
            print(f"using initial state, pop is {initial_state[0]}")
            return initial_state
        population = self.create_new_population()
        species = self.config.species_set_type(self.config.species_set_config, self.reporters)
        species.set_new_population(population)
        species.speciate(0, self.config)
        return population, species, 0
    
    def create_new_population(self) -> Population:
        """Create a population from scratch, then partition into species."""
        return self.reproduction.create_new_genomes(
            self.config.genome_type, self.config.genome_config, self.config.pop_size
        )

    def receive_evaluation(self, user_data: 'UserData') -> None:
        """Receive an evaluation of a member from the user."""
        if user_data.genome_id == 0: 
            return
        self.population[user_data.genome_id].data = user_data
        self.evaluated_genomes[user_data.genome_id] = self.population[user_data.genome_id]
        if self.is_evaluated_threshold_reached():
            self.advance_population()

    def advance_population(self) -> None:
        """Advance the population to the next generation."""
        best_genome = self.evaluate_fitness(self.fitness_function)
        self.track_best_genome(best_genome)

        if self.should_terminate(best_genome):
            print("TERMINATING...")
            self.reporters.found_solution(self.config, self.generation, self.best_genome)
            return

        self.reproduce_evaluated()
        self.reporters.end_generation(self.config, self.population, self.species)
        self.generation += 1

    def evaluate_fitness(self, fitness_function: FitnessFunction) -> DefaultGenome:
        """Evaluate the fitness of the entire population."""
        fitness_function(self.evaluated_genomes, self.config)
        best = max(itervalues(self.evaluated_genomes), key=lambda g: g.fitness)
        self.reporters.post_evaluate(self.config, self.population, self.species, best)
        return best

    def track_best_genome(self, best_genome: DefaultGenome) -> None:
        """Tracks the best genome seen so far."""
        if self.best_genome is None or best_genome.fitness > self.best_genome.fitness:
            self.best_genome = best_genome

    def should_terminate(self, best_genome: DefaultGenome) -> bool:
        """Checks if the evolution should terminate based on the fitness criterion."""
        print(f"no terminate? {self.config.no_fitness_termination}")
        if not self.config.no_fitness_termination:
            print(f"threshold is {self.config.fitness_threshold}")
            if best_genome.fitness_value >= self.config.fitness_threshold:
                return True
        return False

    def is_evaluated_threshold_reached(self) -> bool:
        """Check if the evaluated threshold is reached."""
        return len(self.evaluated_genomes) >= self.evaluation_threshold

    def reproduce_evaluated(self) -> None:
        """Facilitate reproduction based on evaluated genomes and manage speciation."""
        offspring = self._generate_offspring()
        self._remove_evaluated_genomes()
        self._update_population_with_offspring(offspring)
        self._respeciate_population()
        self._handle_possible_extinction()

    def _generate_offspring(self) -> Dict[int, DefaultGenome]:
        """Generate offspring from evaluated genomes."""
        return self.reproduction.reproduce_selected(
            self.config, self.species, self.generation, list(self.evaluated_genomes.keys())
        )

    def _remove_evaluated_genomes(self) -> None:
        """Remove evaluated genomes from the population."""
        for genome_id in self.evaluated_genomes:
            del self.population[genome_id]

    def _update_population_with_offspring(self, offspring: Dict[int, DefaultGenome]) -> None:
        """Update the population with newly created offspring."""
        print(f"updating pop {self.population}, with offspring {offspring}")
        self.population.update(offspring)
        self.available_genomes = list(self.population.keys())  # Refresh available genomes list.

    def _respeciate_population(self) -> None:
        """Re-speciate the population after updating it."""
        self.species.set_new_population(self.population)
        self.species.speciate(self.generation, self.config)

    def _handle_possible_extinction(self) -> None:
        """Handle the case where all species might have gone extinct."""
        if not self.species.species:
            self.reporters.complete_extinction()
            if self.config.reset_on_extinction:
                self.population = self.create_new_population()
                self.species.set_new_population(self.population)
                self.species.speciate(self.generation, self.config)
            else:
                raise CompleteExtinctionException("All species have gone extinct.")
    
    def get_available_genomes(self): 
        return self.available_genomes
    
    def get_evaluated_genomes(self):
        return self.evaluated_genomes
    
    def get_random_non_evaluated_member(self) -> DefaultGenome:
        """Send a random member to the user."""
        if not self.available_genomes:
            raise RuntimeError("No more genomes to send.")
        genome_id = random.choice(self.available_genomes)
        self.available_genomes.remove(genome_id)
        return self.population[genome_id]

    def add_reporter(self, reporter) -> None:
        """Add a reporter to the set of reporters."""
        self.reporters.add(reporter)

    def remove_reporter(self, reporter) -> None:
        """Remove a reporter from the set of reporters."""
        self.reporters.remove(reporter)

    def start_reporting(self) -> None:
        """Start reporting the evolution process."""
        self.reporters.start_generation(self.generation)