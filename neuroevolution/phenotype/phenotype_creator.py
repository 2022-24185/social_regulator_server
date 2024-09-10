"""This module contains the PhenotypeCreator class, which is responsible for creating a network from a genome using ES-HyperNEAT."""
import logging
from neat.config import Config
from neat.genome import DefaultGenome
from neat.nn.feed_forward import FeedForwardNetwork
from neat.nn.recurrent import RecurrentNetwork
from neuroevolution.phenotype.es_hyperneat import ESNetwork
from neuroevolution.phenotype.substrate import Substrate
from neuroevolution.server.models import PhenotypeConfig

# pylint: disable=logging-fstring-interpolation

# Custom Exception Classes
class PhenotypeCreationError(Exception):
    """Raised when the creation of a phenotype network fails."""
    pass

class PhenotypeCreator:
    def __init__(self, config: Config, phenotype_config: PhenotypeConfig):
        try:
            self.config = config
            self.input_coords = phenotype_config.input_coords
            self.output_coords = phenotype_config.output_coords
            self.params = phenotype_config.params.model_dump()

            # Validate the input and output coordinates
            if not self.input_coords or not self.output_coords:
                raise ValueError("Input or output coordinates are missing or invalid.")
            logging.info(f"PhenotypeCreator initialized successfully with input coords: {self.input_coords} and output coords: {self.output_coords}")
        
        except KeyError as e:
            logging.error(f"Error initializing PhenotypeCreator: Missing key {e}")
            raise PhenotypeCreationError(f"Missing configuration key: {e}") from e
        except Exception as e:
            logging.error(f"Error during PhenotypeCreator initialization: {e}")
            raise PhenotypeCreationError(f"PhenotypeCreator initialization failed: {e}") from e

    def create_network_from_genome(self, genome: DefaultGenome) -> RecurrentNetwork:
        """Create a network from a genome using ES-HyperNEAT."""
        
        # Step 1: Create CPPN
        try:
            logging.info(f"Creating CPPN from genome {genome.key}")
            cppn = FeedForwardNetwork.create(genome, self.config)
            if not cppn:
                raise ValueError(f"Failed to create CPPN for genome {genome.key}")
            logging.info(f"CPPN created successfully with {len(cppn.input_nodes)} input nodes and {len(cppn.output_nodes)} output nodes.")
        except Exception as e:
            logging.error(f"Error during CPPN creation for genome {genome.key}: {e}")
            raise PhenotypeCreationError(f"Error during CPPN creation for genome {genome.key}: {e}") from e
        
        # Step 2: Initialize Substrate
        try:
            substrate = Substrate(self.input_coords, self.output_coords)
            if not substrate:
                raise ValueError("Failed to initialize substrate with given input and output coordinates.")
            logging.info(f"Substrate initialized with expected: {len(self.input_coords)} inputs, created: {len(substrate.input_coordinates)}.")
        except Exception as e:
            logging.error(f"Error during substrate initialization for genome {genome.key}: {e}")
            raise PhenotypeCreationError(f"Error during substrate initialization for genome {genome.key}: {e}") from e
        
        # Step 3: Create ES-HyperNEAT Network
        try:
            logging.info(f"Creating ES-HyperNEAT network for genome {genome.key}")
            es_network = ESNetwork(substrate, cppn, self.params)
            
            if not es_network:
                raise ValueError(f"Failed to create ESNetwork for genome {genome.key}")
            
            network = es_network.create_phenotype_network()
            if not network:
                raise ValueError(f"Failed to create phenotype network for genome {genome.key}.")
            logging.info(f"ES-HyperNEAT network created successfully for genome {genome.key}")
        except Exception as e:
            logging.error(f"Error during ES-HyperNEAT network creation for genome {genome.key}: {e}")
            raise PhenotypeCreationError(f"Error during ES-HyperNEAT network creation for genome {genome.key}: {e}") from e
        
        # Step 4: Validate Network Inputs
        try:
            actual_inputs = len(network.input_nodes)
            logging.info(f"Phenotype network created with {actual_inputs} input nodes.")
            expected_inputs = len(self.input_coords)
            if actual_inputs != expected_inputs:
                raise ValueError(f"Expected {expected_inputs} inputs, got {actual_inputs}")
            
            logging.info(f"Successfully created network for genome {genome.key}")
            return network
        except Exception as e:
            logging.error(f"Error validating phenotype network inputs for genome {genome.key}: {e}")
            raise PhenotypeCreationError(f"Error validating phenotype network inputs for genome {genome.key}: {e}") from e