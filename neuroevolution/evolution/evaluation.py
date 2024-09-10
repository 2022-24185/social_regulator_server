"""Implements the core evolution algorithm."""
from typing import List, Callable
from pydantic import BaseModel

from neat.math_util import mean
from neat.genome import DefaultGenome
from neuroevolution.fitness_functions.basic_fitness import BasicFitness
from neuroevolution.evolution.genome_manager import GenomeManager
from neuroevolution.data_models.experiment_data_models import FitnessStats
from neuroevolution.server.models import UserData

# Type aliases for better readability
FitnessSummarizer = Callable[[List[float]], float]

class CompleteExtinctionException(Exception):
    """Exception to raise when a population has no members."""
    def __init__(self, message):
        super().__init__(message)

class Evaluation:
    """
    Manages the evaluation of genomes and tracks their fitness.
    """
    def __init__(self, config, fitness_function: BasicFitness, genome_manager: GenomeManager, threshold: int = 0):
        """
        Initialize the evaluation manager.

        :param config: The configuration object.
        :param fitness_function: The fitness function to evaluate genomes.
        :param evaluation_threshold: The number of evaluations to perform before advancing the population.
        """
        self.threshold = threshold # default is to always advance to next generation
        self.config = config
        self.fitness_function = fitness_function
        self.summarizer = self.get_fitness_summarizer()
        self.genomes = genome_manager

    def set_threshold(self, threshold: int):
        """
        Set the threshold for the number of evaluations to perform before advancing the population.
        """
        self.threshold = threshold
        print("new threshold is ", self.threshold)

    def get_fitness_summarizer(self) -> FitnessSummarizer:
        """
        Choose a fitness summarizer based on configuration, ensuring all conditions are handled.
        """
        fitness_summarizers = {"max": max, "min": min, "mean": mean}
        criterion = self.config.fitness_criterion
        if criterion not in fitness_summarizers:
            if not self.config.no_fitness_termination:
                raise ValueError(f"Invalid fitness criterion: {criterion}")
            return None  # or some default behavior
        return fitness_summarizers[criterion]
        
    def get_fitness_stats(self) -> FitnessStats:
        """
        Get the best genome from the evaluated genomes.
        """
        evaluated = self.genomes.get_evaluated_genomes()
        self.best_genome = max(evaluated, key=lambda genome: genome.fitness)
        self.worst_genome = min(evaluated, key=lambda genome: genome.fitness)
        stats = FitnessStats(
            best_genome_fitness=self.best_genome.fitness,
            worst_genome_fitness=self.worst_genome.fitness,
            median_fitness=self.calculate_median_fitness(evaluated),
            mean_fitness=sum(g.fitness for g in evaluated) / len(evaluated),
            fitness_variance=self.calculate_variance_fitness(evaluated), 
            fitness_quartiles=self.calculate_quartiles_fitness(evaluated)
        )
        return stats
    
    def process_gym_data(self, gym_data: 'UserData') -> None:
        """
        Process user evaluations and update genome data.
        
        :param user_data: The user data to process.
        """
        if not gym_data: 
            raise ValueError("No data received from the server.")
        if gym_data.experiment_data.genome_id == 0:
            return  # Assume 0 is an invalid ID or a placeholder
        genome = self.genomes.update_genome_data(gym_data.experiment_data.genome_id, gym_data)
        self.evaluate(genome)

    def evaluate(self, genome: DefaultGenome, **kwargs):
        """
        Evaluate a genome and store its fitness.
        
        :param genome: The genome to evaluate.
        """
        self.fitness_function(genome, **kwargs)  # Assuming each genome has a fitness attribute
        self.genomes.set_evaluated(genome.key)
        #print(f"🧬 Genome {genome.key} evaluated with fitness {genome.fitness}")
        #print(f"total evaluated: {len(self.genomes.get_evaluated_genomes())}")
    
    def threshold_reached(self) -> bool:
        """
        Check if the evaluation threshold has been reached.
        
        :return: True if the threshold has been reached, False otherwise.
        """
        if len(self.genomes.get_evaluated_genomes()) >= self.threshold * self.genomes.get_alive_genomes_count():
            print(f"🕊️ Threshold reached! {len(self.genomes.get_evaluated_genomes())} genomes evaluated")
            return True
        else:
            return False
        
    def calculate_median_fitness(self, genomes: List[DefaultGenome]):
        fitnesses = sorted(g.fitness for g in genomes)
        mid = len(fitnesses) // 2
        if len(fitnesses) % 2 == 0:
            return (fitnesses[mid - 1] + fitnesses[mid]) / 2
        else:
            return fitnesses[mid]

    def calculate_variance_fitness(self, genomes: List[DefaultGenome]):
        fitnesses = [g.fitness for g in genomes]
        mean_fitness = sum(fitnesses) / len(fitnesses)
        return sum((x - mean_fitness) ** 2 for x in fitnesses) / len(fitnesses)

    def calculate_quartiles_fitness(self, genomes: List[DefaultGenome]):
        fitnesses = sorted(g.fitness for g in genomes)
        q1 = fitnesses[len(fitnesses) // 4]
        q2 = fitnesses[len(fitnesses) // 2]
        q3 = fitnesses[3 * len(fitnesses) // 4]
        return [q1, q2, q3]