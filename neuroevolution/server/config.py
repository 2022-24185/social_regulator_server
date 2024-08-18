# server/config.py
from neuroevolution.run_experiments.async_experiment import AsyncExperiment
from neuroevolution.evolution.population_manager import PopulationManager
from neuroevolution.evolution.evaluation import Evaluation
from neuroevolution.evolution.population_evolver import Evolution
from neuroevolution.fitness_functions.xor_fitness import XORFitness
from neuroevolution.fitness_functions.user_evaluated_fitness import UserEvaluatedFitness
from neuroevolution.gym.xor_gym import XORGym
from neat.genome import DefaultGenome
from neuroevolution.evolution.reproduction import MixedGenerationReproduction
from neuroevolution.evolution.speciation import Speciation
from neuroevolution.evolution.stagnation import MixedGenerationStagnation
from neuroevolution.phenotype.phenotype_creator import PhenotypeCreator
from neuroevolution.run_experiments.online_experiment import OnlineExperiment


class Config:
    SERVER_HOST = '0.0.0.0'
    SERVER_PORT = 8000
    EXPERIMENT = {
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
        'experiment_class':OnlineExperiment,
        'num_generations':5,
        'pop_size':150,
        'fitness_class': UserEvaluatedFitness,
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