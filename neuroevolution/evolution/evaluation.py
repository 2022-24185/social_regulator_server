"""Implements the core evolution algorithm."""
from typing import List, Callable, Dict

from neat.math_util import mean
from neat.genome import DefaultGenome
from neuroevolution.evolution.fitness_functions.basic_fitness import BasicFitness

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
    def __init__(self, config, fitness_function: BasicFitness, evaluation_threshold):
        """
        Initialize the evaluation manager.

        :param config: The configuration object.
        :param fitness_function: The fitness function to evaluate genomes.
        :param evaluation_threshold: The number of evaluations to perform before advancing the population.
        """
        self.config = config
        self.fitness_function = fitness_function
        self.evaluation_threshold = evaluation_threshold
        self.evaluated_genomes : Dict[int, DefaultGenome] = {}
        self.summarizer = self.get_fitness_summarizer()

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
        best = max(self.evaluated_genomes, key=lambda x: self.evaluated_genomes[x].fitness)
        genome = self.evaluated_genomes[best]
        return genome

    def evaluate(self, genome_id: int, genome: DefaultGenome, **kwargs):
        """
        Evaluate a genome and store its fitness.
        
        :param genome: The genome to evaluate.
        """
        self.fitness_function(genome, **kwargs)  # Assuming each genome has a fitness attribute
        self.evaluated_genomes.update({genome_id: genome})
    
    def threshold_reached(self) -> bool:
        """
        Check if the evaluation threshold has been reached.
        
        :return: True if the threshold has been reached, False otherwise.
        """
        if len(self.evaluated_genomes) > self.evaluation_threshold:
            return True
        else:
            return False
        
    def get_evaluated(self) -> List[int]: 
        return list(self.evaluated_genomes.keys())

    def clear_evaluated(self): 
        self.evaluated_genomes.clear()
