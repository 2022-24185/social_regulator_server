
from neuroevolution.fitness_functions.basic_fitness import BasicFitness
from neat.genome import DefaultGenome
from neat.config import Config


class XORFitness(BasicFitness): 
    def __init__(self, config: Config):
        self.config = config

    def __call__(self, genome: DefaultGenome, **kwargs):
        """
        Calculate the fitness of a genome.
        
        :param genome: The genome to evaluate.
        :param kwargs: Additional arguments to pass to the fitness function.
        """
        logged_data = genome.data
        sum_square_error = 0.0
        inputs = logged_data.inputs
        outputs = logged_data.outputs
        expected_outputs = logged_data.expected_outputs

        for xor_input, xor_output, xor_expected in zip(inputs, outputs, expected_outputs):
            sum_square_error += ((xor_output[0] - xor_expected[0]) ** 2.0) / len(inputs)

        genome.fitness = 1 - sum_square_error