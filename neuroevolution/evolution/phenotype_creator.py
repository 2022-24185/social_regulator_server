"""This module contains the PhenotypeCreator class, which is responsible for creating a network from a genome using ES-HyperNEAT."""
from neat.config import Config
from neat.genome import DefaultGenome
from neat.nn.feed_forward import FeedForwardNetwork
from neat.nn.recurrent import RecurrentNetwork
from neuroevolution.evolution.es_hyperneat import ESNetwork
from neuroevolution.evolution.substrate import Substrate

# Constants
# interpolate so we have 3 THREEE coordinates
INPUT_COORDINATES = [(-0.33, -1.), (0, 0), (0.33, 1.)]
OUTPUT_COORDINATES = [(-0.5, 1.), (0., -1.), (0., -1.), (0.5, 1.)]
SUBSTRATE = Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES,)
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

class PhenotypeCreator:
    def __init__(self, config: Config):
        self.config = config

    def create_network_from_genome(self, genome: DefaultGenome) -> RecurrentNetwork:
        """Create a network from a genome using ES-HyperNEAT."""
        cppn = FeedForwardNetwork.create(genome, self.config)
        es_network = ESNetwork(SUBSTRATE, cppn, PARAMS)
        return es_network.create_phenotype_network()
