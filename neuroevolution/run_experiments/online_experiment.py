from typing import Dict, Tuple, List, Any, TYPE_CHECKING
import random
import pickle
import base64


from neat.nn.recurrent import RecurrentNetwork

from neuroevolution.run_experiments.basic_experiment import BasicExperiment
from neuroevolution.server.models import PhenotypeData

if TYPE_CHECKING:
    from neuroevolution.server.models import PhenotypeData


class OnlineExperiment(BasicExperiment):
    """Class for running experiments with user evaluation."""
    def __init__(self, experiment_config: Dict[str, Any], experiment_id: int):
        super().__init__(experiment_config, experiment_id=experiment_id)
        self.eval_config = self.experiment_config['async_eval']
        self.evaluation_pool: List[Tuple[int, RecurrentNetwork]] = [] # can be removed as is handled by genome data, but good for keeping track maybe?

    def run_simulation(self):
        pass
        
    def instanciate(self): 
        """Instanciate the experiment."""
        self.evaluation.set_threshold(self.eval_config['eval_threshold'])
        self.evolution.create_new_population()

    def get_random_individual(self) -> 'PhenotypeData':
        """Serialize the network associated with a genome."""
        (experiment_data, phenotype) = super().get_random_individual()
        self.evaluation_pool.append(experiment_data.genome_id)
        if phenotype is None:
            return None
        pickled = pickle.dumps(phenotype)
        encoded = base64.b64encode(pickled).decode('utf-8')
        return PhenotypeData(experiment_data=experiment_data, new_mediator=encoded)
    
    def clear_pool(self):
        """Clear the evaluation pool."""
        self.evaluation_pool.clear()