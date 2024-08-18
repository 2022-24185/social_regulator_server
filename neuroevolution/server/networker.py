"""This module contains the network class that is used to communicate with the server."""
import pickle
import base64
from typing import TYPE_CHECKING, Tuple

from neuroevolution.run_experiments.basic_experiment import BasicExperiment

if TYPE_CHECKING:
    from neuroevolution.server.models import UserData
    from neat.nn import RecurrentNetwork

class Network:
    def __init__(self, experiment: BasicExperiment):
        self.experiment = experiment

    def receive_evaluation(self, user_data: 'UserData'):
        """Receive and process evaluation data for a genome."""
        print(f"Received evaluation for genome {user_data.genome_id}")
        self.experiment.receive_evaluation(user_data)

    def get_serialized_mediator(self) -> bytes:
        """Serialize the network associated with a genome."""
        genome_id, mediator = self.get_mediator()
        if mediator is None:
            return None
        pickled = pickle.dumps((genome_id, mediator))
        return base64.b64encode(pickled).decode('utf-8')

    def get_mediator(self) -> Tuple[int, 'RecurrentNetwork']:
        """Create a mediator network from a genome."""
        gid, mediator = self.experiment.get_random_individual()
        return (gid, mediator)
