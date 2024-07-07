"""This module contains the PhenotypeCreator class, which is responsible for creating a network from a genome using ES-HyperNEAT."""
from neat.config import Config
from neat.genome import DefaultGenome
from neat.nn.feed_forward import FeedForwardNetwork
from neat.nn.recurrent import RecurrentNetwork
from neuroevolution.phenotype.es_hyperneat import ESNetwork
from neuroevolution.phenotype.substrate import Substrate


class PhenotypeCreator:
    def __init__(self, config: Config, input_coords, output_coords, params):
        self.config = config
        self.input_coords = input_coords
        self.output_coords = output_coords
        self.params = params

    def create_network_from_genome(self, genome: DefaultGenome) -> RecurrentNetwork:
        """Create a network from a genome using ES-HyperNEAT."""
        cppn = FeedForwardNetwork.create(genome, self.config)
        substrate = Substrate(self.input_coords, self.output_coords)
        es_network = ESNetwork(substrate, cppn, self.params)
        network = es_network.create_phenotype_network()
        return network