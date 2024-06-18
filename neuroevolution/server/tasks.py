# neuroevolution/server/tasks.py
import logging
from neuroevolution.evolution.manager import EvolutionManager
from typing import TYPE_CHECKING
from neuroevolution.server.errors import MissingGenomeError
from neuroevolution.server.data_storage import SessionData

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from .models import UserData

logging.basicConfig(level=logging.INFO)

evolution_manager = EvolutionManager('neuroevolution/evolution/config_cppn_social_brain')
session_data = SessionData('session_data.csv')

def process_user_data(data: 'UserData'):
    logging.info(f"Processing user data: {data}")
    session_data.store_session_data(data)
    evolution_manager.receive_evaluation(data)
    logging.info(f"User data processed: {data}")

def swap_out_mediator(user_data: 'UserData') -> bytes:
    logging.info(f"Requesting new mediator genome for mediator ID: {user_data.genome_id}")
    session_data.store_session_data(user_data)
    logging.info(f"User data stored: {user_data}")
    evolution_manager.receive_evaluation(user_data)
    logging.info(f"User data evaluated: {user_data}"[:100])
    new_genome = evolution_manager.fetch_new_genome()
    if not new_genome:
        logging.error("Failed to fetch new genome")
        raise MissingGenomeError("Failed to fetch new genome")
    new_mediator = evolution_manager.get_serialized_network(new_genome)
    logging.info(f"New mediator provided.")
    return new_mediator

def run_evolution():
    logging.info("Starting the evolutionary process")
    evolution_manager.advance_population()
    logging.info("Evolutionary process finished")

def reset_population():
    logging.info("Restarting the population")
    evolution_manager.reset()
    logging.info("Population restarted")