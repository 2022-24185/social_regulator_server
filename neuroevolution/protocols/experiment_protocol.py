from typing import List
from neuroevolution.data_models.experiment_design import VaribleSetup

class ExperimentStep:
    def __init__(self, variables: VaribleSetup): 
        self.variables = variables

    def execute(self, population): 
        raise NotImplementedError("ExperimentStep execution not defined in subclass")

class ExperimentProtocol: 
    def __init__(self, steps: List[ExperimentStep]): 
        self.steps = steps

