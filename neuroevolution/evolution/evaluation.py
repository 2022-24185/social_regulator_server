"""Implements the core evolution algorithm."""
from typing import List, Callable, Dict

from neat.math_util import mean
from neat.genome import DefaultGenome
from neuroevolution.evolution.fitness_functions.basic_fitness import BasicFitness
from neuroevolution.evolution.genome_manager import GenomeManager

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
    def __init__(self, config, fitness_function: BasicFitness, evaluation_threshold, genome_manager: GenomeManager):
        """
        Initialize the evaluation manager.

        :param config: The configuration object.
        :param fitness_function: The fitness function to evaluate genomes.
        :param evaluation_threshold: The number of evaluations to perform before advancing the population.
        """
        self.config = config
        self.fitness_function = fitness_function
        self.evaluation_threshold = evaluation_threshold
        self.summarizer = self.get_fitness_summarizer()
        self.genomes = genome_manager

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
        
    def get_best(self) -> DefaultGenome:
        """
        Get the best genome from the evaluated genomes.
        """
        evaluated = self.genomes.get_evaluated_genomes()
        best = max(evaluated, key=lambda genome: genome.fitness)
        return best

    def evaluate(self, genome: DefaultGenome, **kwargs):
        """
        Evaluate a genome and store its fitness.
        
        :param genome: The genome to evaluate.
        """
        self.fitness_function(genome, **kwargs)  # Assuming each genome has a fitness attribute
        self.genomes.set_evaluated(genome.key)
    
    def threshold_reached(self) -> bool:
        """
        Check if the evaluation threshold has been reached.
        
        :return: True if the threshold has been reached, False otherwise.
        """
        if len(self.genomes.get_evaluated_genomes()) > self.evaluation_threshold:
            return True
        else:
            return False
        
