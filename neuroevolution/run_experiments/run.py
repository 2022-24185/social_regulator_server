
from neat.genome import DefaultGenome

# Default imports
from neuroevolution.lab.lab import Lab
from neuroevolution.evolution.speciation import Speciation
from neuroevolution.evolution.reproduction import MixedGenerationReproduction
from neuroevolution.evolution.stagnation import MixedGenerationStagnation
from neuroevolution.evolution.population_manager import PopulationManager
from neuroevolution.phenotype.phenotype_creator import PhenotypeCreator
from neuroevolution.evolution.evaluation import Evaluation
from neuroevolution.evolution.population_evolver import Evolution

# Experiment imports
from neuroevolution.run_experiments.async_experiment import AsyncExperiment
from neuroevolution.run_experiments.standard_experiment import StandardExperiment
from neuroevolution.fitness_functions.xor_fitness import XORFitness
from neuroevolution.gym.xor_gym import XORGym



experiment_config = {
    'neat_config': {
        'config_path':'neuroevolution/run_experiments/config_cppn_xor',
        'genome_class':DefaultGenome,
        'reproduction_class':MixedGenerationReproduction,
        'speciation_class':Speciation,
        'stagnation_class':MixedGenerationStagnation,
    },
    'class_config': {
        'population_manager':PopulationManager,
        'phenotype_creator':PhenotypeCreator,
        'evaluation':Evaluation,
        'evolution':Evolution,
    },
    'experiment_config': {
        'experiment_class':AsyncExperiment,
        'num_generations':5,
        'pop_size':150,
        'fitness_class':XORFitness,
        'gym_class':XORGym,
        'fitness_criterion':'max',
        'fitness_threshold':0.975,
        'reset_on_extinction':False,
        'async_eval':{
            'eval_pool_size': 10,
            'eval_threshold': 0.4
        },
    }
}


if __name__ == "__main__":
    parameters = {
        'experiment_config.async_eval.eval_threshold': [0.4, 0.8]
        }
    lab = Lab()
    lab.add_parameterized_experiment(experiment_config, parameters)
    lab.run_scenarios()