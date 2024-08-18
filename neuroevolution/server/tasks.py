# neuroevolution/server/tasks.py
import logging
from typing import TYPE_CHECKING, Tuple
from neuroevolution.server.networker import Network
from neuroevolution.run_experiments.async_experiment import AsyncExperiment
from neuroevolution.server.errors import MissingGenomeError
from neuroevolution.server.data_storage import SessionData
from neuroevolution.lab.lab import Lab
from neuroevolution.server.config import Config

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from .models import UserData
    from .models import PhenotypeData

logging.basicConfig(level=logging.INFO)

parameters = {
        'experiment_config.async_eval.eval_threshold': [0.4, 0.8]
        }
lab = Lab()
lab.add_parameterized_experiment(Config.EXPERIMENT, parameters)
lab.instanciate_experiments()
session_data = SessionData('session_data.csv')

def process_user_data(data: 'UserData'):
    """
    Process user data by storing it in the session data and sending it to the network for evaluation.
    """
    print(f"Processing user data: {data}")
    session_data.store_session_data(data)
    lab.return_individual_to_experiment(data)
    print(f"User data processed: {data}")

def swap_out_mediator(user_data: 'UserData') -> 'PhenotypeData':
    """
    Request a new mediator genome to be generated.
    """
    print(f"Requesting new mediator genome for mediator ID: {user_data.experiment_data.genome_id}")
    session_data.store_session_data(user_data)
    print(f"User data stored: {user_data}")
    process_user_data(user_data)
    new_mediator = lab.sample_random_experiment()
    print(f"User data evaluated: {user_data}"[:100])
    if not new_mediator.new_mediator: 
        logging.error("Failed to fetch new genome")
        raise MissingGenomeError("Failed to fetch new genome")
    print(f"New mediator provided.")
    return new_mediator

def get_new_mediator() -> 'PhenotypeData':
    """
    Request a new mediator genome to be generated.
    """
    print("Requesting new mediator genome")
    new_mediator = lab.sample_random_experiment()
    if not new_mediator.new_mediator: 
        logging.error("Failed to fetch new genome")
        raise MissingGenomeError("Failed to fetch new genome")
    print(f"New mediator provided.")
    return new_mediator

def run_evolution():
    """
    Start the evolutionary process.
    """
    logging.info("Starting the evolutionary process")
    #experiment.start()
    logging.info("Evolutionary process finished")

def reset_population():
    """
    Restart the population.
    """
    logging.info("Restarting the population")
    #experiment.reset()
    logging.info("Population restarted")