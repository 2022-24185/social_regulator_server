
from neuroevolution.run_experiments.experiment import SimulatedUserEvalExperiment

if __name__ == "__main__":
    experiment = SimulatedUserEvalExperiment('neuroevolution/evolution/config_cppn_social_brain')
    experiment.start(10)