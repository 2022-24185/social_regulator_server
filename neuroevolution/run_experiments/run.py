
from neuroevolution.run_experiments.experiment import SimulatedUserEvalExperiment


if __name__ == "__main__":
    experiment = SimulatedUserEvalExperiment(config_path='neuroevolution/run_experiments/config_cppn_xor')
    
    experiment.start(50)