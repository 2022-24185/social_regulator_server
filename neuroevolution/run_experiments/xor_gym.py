""" The module for the XOR experiment gym. """

from typing import Any, Dict, Tuple

from neat.nn.recurrent import RecurrentNetwork

from neuroevolution.run_experiments.basic_gym import BasicGym
from neuroevolution.server.models import XORData

VERSION = "S"
VERSION_TEXT = "small" if VERSION == "S" else "medium" if VERSION == "M" else "large"

# Network inputs and expected outputs.
XOR_INPUTS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
XOR_OUTPUTS = [(0.0,), (1.0,), (1.0,), (0.0,)]

# Network coordinates and the resulting substrate.
INPUT_COORDINATES = [(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)]
OUTPUT_COORDINATES = [(0.0, 1.0)]

def params(version):
    """
    ES-HyperNEAT specific parameters.
    """
    return {"initial_depth": 0 if version == "S" else 1 if version == "M" else 2,
            "max_depth": 1 if version == "S" else 2 if version == "M" else 3,
            "variance_threshold": 0.03,
            "band_threshold": 0.3,
            "iteration_level": 1,
            "division_threshold": 0.5,
            "max_weight": 5.0,
            "activation": "sigmoid"}


DYNAMIC_PARAMS = params(VERSION)

class XORGym(BasicGym):
    def __init__(self):
        super().__init__()
        self.input_coords = INPUT_COORDINATES
        self.output_coords = OUTPUT_COORDINATES
        self.params = DYNAMIC_PARAMS

    def run(self, individual: Tuple[int, RecurrentNetwork]) -> Dict[str, Any]:
        """
        Run the simulation environment.
        
        :param network: The phenotype network to evaluate.
        :return: The simulation data
        """
        return self.simulate_xor(individual)
    
    def simulate_xor(self, individual: Tuple[int, RecurrentNetwork]) -> Dict[str, Any]:
        """
        Create a random user evaluation within a range.

        :param gid: The genome id.
        :return: The user data.
        """
        logged_data = XORData(
            genome_id=individual[0],
            inputs=[],
            outputs=[],
            expected_outputs=[]
        )
        network = individual[1]

        for xor_inputs, xor_expected in zip(XOR_INPUTS, XOR_OUTPUTS):
            new_xor_input = xor_inputs + (1.0,)
            network.reset()

            for _ in range(network.activations):
                xor_output = network.activate(new_xor_input)

            logged_data.inputs.append(new_xor_input)
            logged_data.outputs.append(xor_output)
            logged_data.expected_outputs.append(xor_expected)

        return logged_data