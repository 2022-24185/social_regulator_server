# server/config.py
from neuroevolution.run_experiments.async_experiment import AsyncExperiment
from neuroevolution.evolution.population_manager import PopulationManager
from neuroevolution.evolution.evaluation import Evaluation
from neuroevolution.evolution.population_evolver import Evolution
from neuroevolution.fitness_functions.xor_fitness import XORFitness
from neuroevolution.fitness_functions.user_evaluated_fitness import UserEvaluatedFitness
from neuroevolution.gym.online_gym import OnlineGym
from neat.genome import DefaultGenome
from neuroevolution.evolution.reproduction import MixedGenerationReproduction
from neuroevolution.evolution.speciation import Speciation
from neuroevolution.evolution.stagnation import MixedGenerationStagnation
from neuroevolution.phenotype.phenotype_creator import PhenotypeCreator
from neuroevolution.run_experiments.online_experiment import OnlineExperiment
from neuroevolution.server.models import ExperimentConfig, PhenotypeConfig, AsyncEvalConfig, ClassConfig, FullExperimentConfig, NEATConfig, PhenotypeParams

INPUT_COORDINATES = [
    (-1.0, -1.0), (-0.75, -1.0), (-0.5, -1.0), (-0.25, -1.0), (0.0, -1.0), 
    (0.25, -1.0), (0.5, -1.0), (0.75, -1.0), (1.0, -1.0), (1.25, -1.0)
]

OUTPUT_COORDINATES = [
    (-1.0, 1.0), (-0.75, 1.0), (-0.5, 1.0), (-0.25, 1.0), (0.0, 1.0),
    (0.25, 1.0), (0.5, 1.0), (0.75, 1.0), (1.0, 1.0), (1.25, 1.0)
]

# class Config:
#     SERVER_HOST = '0.0.0.0'
#     SERVER_PORT = 8000
#     EXPERIMENT = {
#     'neat_config': {
#         'config_path':'neuroevolution/run_experiments/config_cppn_xor',
#         'genome_class':DefaultGenome,
#         'reproduction_class':MixedGenerationReproduction,
#         'speciation_class':Speciation,
#         'stagnation_class':MixedGenerationStagnation,
#     },
#     'class_config': {
#         'population_manager':PopulationManager,
#         'phenotype_creator':PhenotypeCreator,
#         'evaluation':Evaluation,
#         'evolution':Evolution,
#     },
#     'experiment_config': {
#         'experiment_class':OnlineExperiment,
#         'num_generations':5,
#         'pop_size':150,
#         'fitness_class': UserEvaluatedFitness,
#         'gym_class':OnlineGym,
#         'fitness_criterion':'max',
#         'fitness_threshold':0.975,
#         'reset_on_extinction':False,
#         'async_eval':{
#             'eval_pool_size': 10,
#             'eval_threshold': 0.4
#         },
#     },
#     'phenotype_config': {
#         'version': 'M',  # Example version for phenotype
#         'input_coords': INPUT_COORDINATES,
#         'output_coords': OUTPUT_COORDINATES,
#         'dynamic_params': {
#             "initial_depth": 1,
#             "max_depth": 2,
#             "variance_threshold": 0.03,
#             "band_threshold": 0.3,
#             "iteration_level": 1,
#             "division_threshold": 0.5,
#             "max_weight": 5.0,
#             "activation": "sigmoid"
#         }
#     }
# }
    

class Config: 
    SERVER_HOST = '0.0.0.0'
    SERVER_PORT = 8000

    EXPERIMENT = FullExperimentConfig(
        neat_config=NEATConfig(
            config_path='neuroevolution/run_experiments/config_cppn_xor',
            genome_class=DefaultGenome,
            reproduction_class=MixedGenerationReproduction,
            speciation_class=Speciation,
            stagnation_class=MixedGenerationStagnation
        ),
        class_config=ClassConfig(
            population_manager=PopulationManager,
            phenotype_creator=PhenotypeCreator,
            evaluation=Evaluation,
            evolution=Evolution
        ),
        experiment_config=ExperimentConfig(
            experiment_class=OnlineExperiment,
            num_generations=5,
            pop_size=30,
            fitness_class=UserEvaluatedFitness,
            gym_class=OnlineGym,
            fitness_criterion='max',
            fitness_threshold=5,
            reset_on_extinction=False,
            additional_config=AsyncEvalConfig(
                eval_pool_size=10,
                eval_threshold=0.4
            )
        ),
        phenotype_config=PhenotypeConfig(
            version='M',
            input_coords=INPUT_COORDINATES,
            output_coords=OUTPUT_COORDINATES,
            params=PhenotypeParams(
                initial_depth=1,
                max_depth=2,
                variance_threshold=0.03,
                band_threshold=0.3,
                iteration_level=1,
                division_threshold=0.5,
                max_weight=5.0,
                activation='sigmoid'
            )
        )
    )