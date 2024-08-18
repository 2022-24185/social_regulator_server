from typing import List, Dict, Any
from pydantic import BaseModel, Field

class NEATConfig(BaseModel):
    config_path: str
    genome_class: str
    reproduction_class: str
    speciation_class: str
    stagnation_class: str

class ClassConfig(BaseModel):
    population_manager: str
    phenotype_creator: str
    evaluation: str
    evolution: str

class AsyncEvalConfig(BaseModel):
    eval_pool_size: int
    eval_threshold: float

class ExperimentConfig(BaseModel):
    experiment_class: str
    num_generations: int
    pop_size: int
    fitness_class: str
    gym_class: str
    fitness_criterion: str
    fitness_threshold: float
    reset_on_extinction: bool
    async_eval: AsyncEvalConfig

class ExperimentConfigModel(BaseModel):
    neat_config: NEATConfig
    class_config: ClassConfig
    experiment_config: ExperimentConfig
