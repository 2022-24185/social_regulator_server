""" This module contains the abstract class for the gym environment. """
from abc import ABC, abstractmethod
from typing import Tuple
from neat.nn.recurrent import RecurrentNetwork
from pydantic import BaseModel

class BasicGym(ABC):
    def __init__(self):
        self.input_coords = None
        self.output_coords = None
        self.params = None

    @abstractmethod
    def run(self, individual: Tuple[int, RecurrentNetwork]) -> BaseModel:
        """
        Run the simulation environment.
        
        :param network: The phenotype network to evaluate.
        :return: The simulation data
        """
        pass