from typing import List, Dict, Optional
from pydantic import BaseModel

class SummaryStatistics(BaseModel):
    """Summary statistics for a specific metric."""
    mean: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    std_dev: Optional[float] = None

class QuartileStatistics(BaseModel):
    """Quartile statistics for a specific metric."""
    q1: Optional[float] = None
    median: Optional[float] = None
    q3: Optional[float] = None

class GenerationStats(BaseModel):
    """Aggregated statistics for generation data."""
    generation: int
    population_start_size: SummaryStatistics
    population_end_size: SummaryStatistics
    active_species_count: SummaryStatistics
    new_species_count: Optional[SummaryStatistics] = None
    extinct_species_count: Optional[SummaryStatistics] = None

class FitnessStatsSummary(BaseModel):
    """Aggregated statistics for fitness data."""
    best_fitness: SummaryStatistics
    worst_fitness: SummaryStatistics
    mean_fitness: SummaryStatistics
    median_fitness: SummaryStatistics
    fitness_variance: SummaryStatistics
    fitness_quartiles: QuartileStatistics
    best_fitness_improvement_rate: Optional[SummaryStatistics] = None

class DiversityMetrics(BaseModel):
    """Aggregated metrics for diversity."""
    genetic_diversity: Optional[SummaryStatistics] = None
    population_entropy: Optional[SummaryStatistics] = None

class SelectionPressureMetrics(BaseModel):
    """Aggregated metrics for selection pressure."""
    selection_intensity: Optional[SummaryStatistics] = None
    selection_differential: Optional[SummaryStatistics] = None
    proportion_selected: Optional[SummaryStatistics] = None

class ConvergenceMetrics(BaseModel):
    """Aggregated metrics for convergence."""
    convergence_rate: Optional[SummaryStatistics] = None
    fitness_std_dev: Optional[SummaryStatistics] = None
    best_to_mean_fitness_ratio: Optional[SummaryStatistics] = None

class AggregatedStatistics(BaseModel):
    """Overall aggregated statistics across multiple experiments."""
    total_experiments: int
    generation_statistics: Dict[str, GenerationStats]
    fitness_statistics: FitnessStatsSummary
    diversity_metrics: Optional[DiversityMetrics] = None
    selection_pressure_metrics: Optional[SelectionPressureMetrics] = None
    convergence_metrics: Optional[ConvergenceMetrics] = None