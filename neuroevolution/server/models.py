# server/models.py
from pydantic import BaseModel
from typing import Optional

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