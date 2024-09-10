from .basic_gym import BasicGym


# make a list of 10 input coordinates evenly distributed in a 2D space, and then 10 output coordinates also fitting in the same space. 
# it might be an idea to separate inputs to the left and outputs to the right, or vice versa.



PARAMS = {
    "initial_depth": 1,
    "max_depth": 6,
    "variance_threshold": 0.03,
    "band_threshold": 0.3,
    "iteration_level": 1,
    "division_threshold": 0.5,
    "max_weight": 8.0,
    "activation": "sigmoid"
}

class OnlineGym(BasicGym): 
    def __init__(self):
        super().__init__()

    def run(self, individual):
        """
        Run the simulation environment.
        
        :param network: The phenotype network to evaluate.
        :return: The simulation data
        """
        pass
    