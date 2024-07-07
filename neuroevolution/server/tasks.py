# neuroevolution/server/tasks.py
import logging
from typing import TYPE_CHECKING
from neuroevolution.server.networker import Network
from neuroevolution.run_experiments.experiment import SimulatedUserEvalExperiment
from neuroevolution.server.errors import MissingGenomeError
from neuroevolution.server.data_storage import SessionData

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from .models import UserData

logging.basicConfig(level=logging.INFO)

experiment = SimulatedUserEvalExperiment('neuroevolution/evolution/config_cppn_social_brain')
network = Network(experiment)
session_data = SessionData('session_data.csv')

def process_user_data(data: 'UserData'):
    """
    Process user data by storing it in the session data and sending it to the network for evaluation.
    """
    print(f"Processing user data: {data}")
    session_data.store_session_data(data)
    network.receive_evaluation(data)
    print(f"User data processed: {data}")

def swap_out_mediator(user_data: 'UserData') -> bytes:
    """
    Request a new mediator genome to be generated.
    """
    print(f"Requesting new mediator genome for mediator ID: {user_data.genome_id}")
    session_data.store_session_data(user_data)
    print(f"User data stored: {user_data}")
    network.receive_evaluation(user_data)
    print(f"User data evaluated: {user_data}"[:100])
    new_mediator = network.get_serialized_mediator()
    if not new_mediator:
        logging.error("Failed to fetch new genome")
        raise MissingGenomeError("Failed to fetch new genome")
    print(f"New mediator provided.")
    return new_mediator

def run_evolution():
    """
    Start the evolutionary process.
    """
    logging.info("Starting the evolutionary process")
    experiment.start()
    logging.info("Evolutionary process finished")

def reset_population():
    """
    Restart the population.
    """
    logging.info("Restarting the population")
    experiment.reset()
    logging.info("Population restarted")