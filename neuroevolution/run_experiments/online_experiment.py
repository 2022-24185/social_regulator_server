from typing import Dict, Tuple, List, Any, TYPE_CHECKING
import pickle
import base64
import logging


from neat.nn.recurrent import RecurrentNetwork

from neuroevolution.run_experiments.basic_experiment import BasicExperiment, InitializationError
from neuroevolution.server.models import PhenotypeData, AsyncEvalConfig
from neuroevolution.phenotype.phenotype_creator import PhenotypeCreationError
# pylint: disable=logging-fstring-interpolation



class OnlineExperiment(BasicExperiment):
    """Class for running experiments with user evaluation."""
    def __init__(self, experiment_config: Dict[str, Any], experiment_id: int):
        """Initialize the online experiment."""
        try:
            super().__init__(experiment_config, experiment_id=experiment_id)
            self.eval_config: AsyncEvalConfig = self.experiment_config.additional_config
            self.evaluation_pool: List[Tuple[int, RecurrentNetwork]] = []
            logging.info(f"Initialized OnlineExperiment with experiment_id: {experiment_id}")
        except KeyError as e:
            logging.error(f"Missing configuration key during initialization: {e}")
            raise InitializationError(f"Missing configuration key: {e}") from e
        except Exception as e:
            logging.error(f"Error initializing OnlineExperiment with id {experiment_id}: {e}")
            raise InitializationError(f"Error initializing experiment: {e}") from e

    def run_simulation(self):
        pass
        
    def instanciate(self):
        """Instanciate the experiment."""
        try:
            self.evaluation.set_threshold(self.eval_config.eval_threshold)
            logging.info(f"Set evaluation threshold to {self.eval_config.eval_threshold} for experiment {self.experiment_id}")
            
            self.evolution.create_new_population()
            logging.info(f"Created new population for experiment {self.experiment_id}")
        except AttributeError as e:
            logging.error(f"Error setting threshold or creating population: {e}")
            raise InitializationError(f"Error instanciating experiment {self.experiment_id}: {e}") from e
        except Exception as e:
            logging.error(f"Unexpected error in instanciation of experiment {self.experiment_id}: {e}")
            raise

    def get_random_individual(self) -> 'PhenotypeData':
        """Serialize the network associated with a genome and return it as PhenotypeData."""
        try:
            # Get a random individual from the parent class method
            (experiment_data, phenotype) = super().get_random_individual()
            self.evaluation_pool.append(experiment_data.genome_id)
            
            # Handle the case where phenotype is None
            if phenotype is None:
                logging.warning(f"No phenotype available for genome {experiment_data.genome_id}")
                return None
            
            # Serialize and encode the phenotype
            pickled = pickle.dumps(phenotype)
            encoded = base64.b64encode(pickled).decode('utf-8')
            
            logging.info(f"Serialized and encoded phenotype for genome {experiment_data.genome_id}")
            return PhenotypeData(experiment_data=experiment_data, new_mediator=encoded)
        
        except PhenotypeCreationError as e:
            logging.error(f"Error creating phenotype for genome {experiment_data.genome_id}: {e}")
            raise
        except (pickle.PicklingError, base64.binascii.Error) as e:
            logging.error(f"Error serializing or encoding phenotype for genome {experiment_data.genome_id}: {e}")
            raise PhenotypeCreationError(f"Error serializing phenotype for genome {experiment_data.genome_id}: {e}") from e
        except Exception as e:
            logging.error(f"Unexpected error retrieving random individual for experiment {self.experiment_id}: {e}")
            raise PhenotypeCreationError(f"Error retrieving random individual for experiment {self.experiment_id}: {e}") from e
    
    def clear_pool(self):
        """Clear the evaluation pool."""
        self.evaluation_pool.clear()