from pydantic import BaseModel, Field
from typing import List, Union
import time
import uuid

class EventData(BaseModel):
    """Base class for events in the evolutionary process."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    generation: int

class FitnessStats(BaseModel):
    """Statistics related to fitness within the population."""
    best_genome_fitness: float = 0.0
    worst_genome_fitness: float = 0.0
    mean_fitness: float = 0.0
    median_fitness: float = 0.0
    fitness_variance: float = 0.0
    fitness_quartiles: List[float] = Field(default_factory=list)  # 25th, 50th, and 75th percentiles

class SpeciesFitnessStats(FitnessStats):
    """Fitness statistics for a specific species."""
    species_id: int

class GenerationSummaryData(EventData):
    """Summary data for a generation."""
    population_start_size: int = 0
    population_end_size: int = 0
    active_species_count: int = 0
    fitness_summary: FitnessStats = Field(default_factory=FitnessStats)

class SpeciesPopulationDynamics(BaseModel):
    """Population dynamics for a species."""
    species_id: int
    initial_size: int
    final_size: int
    new_genomes: int
    elites_preserved: int

class SpeciesStagnationMetrics(EventData):
    """Metrics related to species stagnation."""
    species_id: int
    stagnant_generations: int
    fitness_change: float

class FitnessContributionData(BaseModel):
    """Data on individual genome's contribution to fitness."""
    genome_id: int
    contribution: Union[str, float]
    description: str = ""

class SelectiveEvaluationData(EventData):
    """Data on selective evaluation of genomes."""
    genome_id: int
    fitness: float
    fitness_contributions: List[FitnessContributionData] = Field(default_factory=list)

class SpeciesReproductionData(EventData):
    """Data on species reproduction events."""
    species_id: int
    offspring_count: int
    population_dynamics: SpeciesPopulationDynamics

class ReproductionData(EventData):
    """Data on reproduction events in the population."""
    population_size: int
    species_reproduction: List[SpeciesReproductionData] = Field(default_factory=list)

class SolutionData(EventData):
    """Data on the best solution found."""
    best_genome: int
    fitness: float

class StagnationData(EventData):
    """Data on species stagnation."""
    species_id: int

class ExtinctionData(EventData):
    """Data on extinction events."""
    extinction_count: int

class InfoData(EventData):
    """General information messages."""
    message: str

class SelectionPressureMetricsData(BaseModel):
    """Metrics related to selection pressure."""
    selection_intensity: float
    selection_differential: float
    proportion_selected: float

class SelectionPressureData(EventData):
    """Data on selection pressure during the process."""
    selected_individuals: int
    metrics: SelectionPressureMetricsData

class ConvergenceMetricsData(EventData):
    """Metrics related to population convergence."""
    fitness_std: float
    genetic_diversity: float
    best_to_mean_fitness_ratio: float
    fitness_improvement_rate: float
    population_entropy: float

class FailureExceptionData(EventData):
    """Data on any exceptions or failures."""
    exception_message: str

class ExperimentDataModel(BaseModel):
    """Overall model to track evolutionary algorithm experiment data."""
    generations: List[GenerationSummaryData] = []
    selective_evaluations: List[SelectiveEvaluationData] = Field(default_factory=list)
    reproductions: List[ReproductionData] = Field(default_factory=list)
    species_fitness_stats: List[SpeciesFitnessStats] = Field(default_factory=list)
    species_population_dynamics: List[SpeciesPopulationDynamics] = Field(default_factory=list)
    species_stagnation_metrics: List[SpeciesStagnationMetrics] = Field(default_factory=list)
    extinction_events: List[ExtinctionData] = Field(default_factory=list)
    solutions: List[SolutionData] = Field(default_factory=list)
    stagnations: List[StagnationData] = Field(default_factory=list)
    info: List[InfoData] = Field(default_factory=list)
    selection_pressure: List[SelectionPressureData] = Field(default_factory=list)
    convergence_metrics: List[ConvergenceMetricsData] = Field(default_factory=list)
    failures_exceptions: List[FailureExceptionData] = Field(default_factory=list)