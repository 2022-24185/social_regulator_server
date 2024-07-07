""" The """
import random
from typing import Any, Dict, Tuple
from neat.nn.recurrent import RecurrentNetwork

from neuroevolution.server.models import UserData
from neuroevolution.run_experiments.basic_gym import BasicGym

INPUT_COORDINATES = [(-0.33, -1.), (0, 0), (0.33, 1.)]
OUTPUT_COORDINATES = [(-0.5, 1.), (0., -1.), (0., -1.), (0.5, 1.)]
PARAMS = {
    "initial_depth": 1,
    "max_depth": 2,
    "variance_threshold": 0.03,
    "band_threshold": 0.3,
    "iteration_level": 1,
    "division_threshold": 0.5,
    "max_weight": 8.0,
    "activation": "sigmoid"
}

class ChatGym(BasicGym):
    def __init__(self):
        self.input_coords = INPUT_COORDINATES
        self.output_coords = OUTPUT_COORDINATES
        self.params = PARAMS

    def run(self, individual: Tuple[int, RecurrentNetwork]) -> Dict[str, Any]:
        """
        Run the simulation environment.
        
        :param network: The phenotype network to evaluate.
        :return: The simulation data
        """
        gid = individual[0]
        return self.simulate_user_evaluation(gid)

    def simulate_user_evaluation(self, gid: int) -> UserData:
        """
        Create a random user evaluation within a range.
        
        :param gid: The genome id.
        :return: The user data.
        """
        return UserData(
            genome_id=gid,
            time_since_startup=random.randint(0, 1000),
            user_rating=random.randint(0, 5),
            last_message=None,
            last_message_time=None,
            last_response=None,
            last_response_time=None,
        )