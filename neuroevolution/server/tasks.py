# neuroevolution/server/tasks.py
import logging
from typing import TYPE_CHECKING, Tuple, Optional 
from neuroevolution.server.networker import Network
from neuroevolution.run_experiments.async_experiment import AsyncExperiment
from neuroevolution.server.errors import MissingGenomeError
from neuroevolution.server.data_storage import SessionData
from neuroevolution.lab.lab import Lab, LabError
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

def process_user_data(user_data: 'UserData'):
    """
    Store and process user data, send it for evaluation.
    """
    try:
        logging.info(f"Processing user data: {user_data}")
        session_data.store_session_data(user_data)
        lab.return_individual_to_experiment(user_data)
        logging.info("User data processed successfully.")
    except Exception as e:
        logging.error(f"Error processing user data: {e}")

def swap_out_mediator(user_data: 'UserData') -> Optional['PhenotypeData']:
    """
    Replace the mediator genome based on user data.
    """
    try:
        logging.info(f"Swapping out mediator for genome ID: {user_data.experiment_data.genome_id}")
        session_data.store_session_data(user_data)
        process_user_data(user_data)
        new_mediator = lab.sample_random_experiment()
        
        if not new_mediator.new_mediator:
            raise MissingGenomeError("Failed to fetch new genome")
        
        logging.info("New mediator generated successfully.")
        return new_mediator
    
    except MissingGenomeError as e:
        logging.error(f"Missing genome: {e}")
        return None

    except Exception as e:
        logging.error(f"Error swapping mediator: {e}")
        return None

def get_new_mediator() -> Optional['PhenotypeData']:
    """
    Get a new mediator genome.
    """
    try:
        logging.info("Requesting new mediator genome.")
        new_mediator = lab.sample_random_experiment()
        
        if not new_mediator.new_mediator:
            raise MissingGenomeError("Failed to fetch new genome")
        
        logging.info("New mediator provided successfully.")
        return new_mediator
    
    except MissingGenomeError as e:
        logging.error(f"Missing genome: {e}")
        return None

    except Exception as e:
        logging.error(f"Error getting new mediator: {e}")
        return None
    
def get_experiment_statuses() -> dict:
    """
    Retrieve the current experiment status, including experiment details.
    """
    try:
        # Get statuses as JSON from the Lab instance
        statuses = lab.get_experiment_statuses()
        return statuses  # FastAPI automatically converts it to JSON
    except LabError as e:
        logging.error(f"Lab error occurred: {e}")
        raise Exception(status_code=500, detail="Error fetching experiment statuses") from e
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}")
        raise Exception(status_code=500, detail="Unexpected server error") from e

def reset_experiment():
    """
    Reset the current experiment.
    """
    try:
        logging.info("Resetting the experiment...")
        lab.reset_experiment()  # Assuming this method exists in Lab
        logging.info("Experiment reset successfully.")
    except Exception as e:
        logging.error(f"Error resetting experiment: {e}")

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