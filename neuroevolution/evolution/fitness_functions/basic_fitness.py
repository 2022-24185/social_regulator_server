from abc import ABC, abstractmethod
from neat.config import Config
from neat.genome import DefaultGenome

class BasicFitness(ABC): 
    """
    Base class for fitness functions. 
    Call the class to calculate the fitness of a genome.
    
    :param: init:config: The configuration object.
    :param: call:genome: The genome to evaluate.
    :param: call:kwargs: Additional arguments to pass to the fitness function.
    """
    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def __call__(self, genome: DefaultGenome, **kwargs):
        """
        Calculate the fitness of a genome.
        
        :param genome: The genome to evaluate.
        :param kwargs: Additional arguments to pass to the fitness function.
        """
        pass