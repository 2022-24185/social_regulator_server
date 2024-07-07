from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
from neat.nn.recurrent import RecurrentNetwork

class BasicGym(ABC):
    def __init__(self):
        self.input_coords = None
        self.output_coords = None
        self.params = None

    @abstractmethod
    def run(self, individual: Tuple[int, RecurrentNetwork]) -> Dict[str, Any]:
        """
        Run the simulation environment.
        
        :param network: The phenotype network to evaluate.
        :return: The simulation data
        """
        pass