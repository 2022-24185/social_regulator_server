# server/models.py
from pydantic import BaseModel, Field, PositiveInt, conint
from typing import Optional, Type, List, Tuple

class ExperimentData(BaseModel):
    experiment_id: int
    genome_id: int

class UserData(BaseModel):
    experiment_data: ExperimentData
    time_since_startup: float
    user_rating: int
    last_message: Optional[str] = None
    last_message_time: Optional[float] = None
    last_response: Optional[str] = None
    last_response_time: Optional[float] = None

class XORData(BaseModel):
    experiment_data: ExperimentData
    inputs: list
    outputs: list
    expected_outputs: list

class PhenotypeData(BaseModel):
    experiment_data: ExperimentData
    new_mediator: str

class ResponseModel(BaseModel):
    phenotype: PhenotypeData
    message: str



class ServerConfig(BaseModel):
    host: str = '0.0.0.0'
    port: PositiveInt = 8000

class NEATConfig(BaseModel):
    config_path: str
    genome_class: Type
    reproduction_class: Type
    speciation_class: Type
    stagnation_class: Type

# Model for the class configuration
class ClassConfig(BaseModel):
    population_manager: Type
    phenotype_creator: Type
    evaluation: Type
    evolution: Type

class PhenotypeParams(BaseModel):
    initial_depth: conint(ge=0, le=5) = 1  # Constrained int between 0 and 5
    max_depth: conint(ge=1, le=5) = 2
    variance_threshold: float = 0.03
    band_threshold: float = 0.3
    iteration_level: PositiveInt = 1
    division_threshold: float = 0.5
    max_weight: float = 5.0
    activation: str = 'sigmoid'

class PhenotypeConfig(BaseModel):
    version: str = 'M'
    input_coords: List[Tuple[float, float]] = Field(..., description="Coordinates of input neurons")
    output_coords: List[Tuple[float, float]] = Field(..., description="Coordinates of output neurons")
    params: PhenotypeParams

class AsyncEvalConfig(BaseModel):
    eval_pool_size: int = Field(..., gt=0)
    eval_threshold: float = Field(..., gt=0.0, le=1.0)

# Model for the main experiment configuration
class ExperimentConfig(BaseModel):
    experiment_class: Type
    num_generations: int = Field(..., gt=0)
    pop_size: int = Field(..., gt=0)
    fitness_class: Type
    gym_class: Type
    fitness_criterion: str
    fitness_threshold: float = Field(..., gt=0.0, le=1.0)
    reset_on_extinction: bool
    additional_config: Optional[BaseModel] = None

# Main configuration model to combine all the sub-configs
class FullExperimentConfig(BaseModel):
    neat_config: NEATConfig
    class_config: ClassConfig
    experiment_config: ExperimentConfig
    phenotype_config: PhenotypeConfig
