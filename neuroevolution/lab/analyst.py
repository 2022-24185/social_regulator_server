from typing import List, Dict
import numpy as np

from neuroevolution.data_models.experiment_data_models import ExperimentDataModel
from neuroevolution.data_models.aggregated_experiments import AggregatedStatistics, SummaryStatistics, QuartileStatistics, GenerationStats, FitnessStatsSummary

# Reusing your existing data models for single run and aggregated data

class DataAnalyst:
    def __init__(self, experiment_data: List[ExperimentDataModel]):
        self.experiment_data = experiment_data

    def compute_summary_statistics(self, data: List[float]) -> SummaryStatistics:
        if not data:
            return SummaryStatistics()
        return SummaryStatistics(
            mean=np.mean(data),
            min=np.min(data),
            max=np.max(data),
            std_dev=np.std(data)
        )

    def compute_quartile_statistics(self, data: List[float]) -> QuartileStatistics:
        if not data:
            return QuartileStatistics()
        return QuartileStatistics(
            q1=np.percentile(data, 25),
            median=np.median(data),
            q3=np.percentile(data, 75)
        )

    def aggregate_generation_stats(self) -> Dict[str, GenerationStats]:
        generation_data = {}
        for exp in self.experiment_data:
            for gen in exp.generations:
                gen_num = gen.generation
                if gen_num not in generation_data:
                    generation_data[gen_num] = {
                        'population_start_size': [],
                        'population_end_size': [],
                        'active_species_count': []
                    }
                generation_data[gen_num]['population_start_size'].append(gen.population_start_size)
                generation_data[gen_num]['population_end_size'].append(gen.population_end_size)
                generation_data[gen_num]['active_species_count'].append(gen.active_species_count)

        aggregated_stats = {}
        for gen_num, data in generation_data.items():
            aggregated_stats[gen_num] = GenerationStats(
                generation=gen_num,
                population_start_size=self.compute_summary_statistics(data['population_start_size']),
                population_end_size=self.compute_summary_statistics(data['population_end_size']),
                active_species_count=self.compute_summary_statistics(data['active_species_count'])
            )
        return aggregated_stats

    def aggregate_fitness_stats(self) -> FitnessStatsSummary:
        best_fitness = []
        worst_fitness = []
        mean_fitness = []
        median_fitness = []
        fitness_variance = []
        fitness_quartiles = []

        for exp in self.experiment_data:
            for fitness_stat in exp.species_fitness_stats:
                best_fitness.append(fitness_stat.best_genome_fitness)
                worst_fitness.append(fitness_stat.worst_genome_fitness)
                mean_fitness.append(fitness_stat.mean_fitness)
                median_fitness.append(fitness_stat.median_fitness)
                fitness_variance.append(fitness_stat.fitness_variance)
                fitness_quartiles.extend(fitness_stat.fitness_quartiles)

        return FitnessStatsSummary(
            best_fitness=self.compute_summary_statistics(best_fitness),
            worst_fitness=self.compute_summary_statistics(worst_fitness),
            mean_fitness=self.compute_summary_statistics(mean_fitness),
            median_fitness=self.compute_summary_statistics(median_fitness),
            fitness_variance=self.compute_summary_statistics(fitness_variance),
            fitness_quartiles=self.compute_quartile_statistics(fitness_quartiles)
        )

    def compute_aggregated_statistics(self) -> AggregatedStatistics:
        return AggregatedStatistics(
            total_experiments=len(self.experiment_data),
            generation_statistics=self.aggregate_generation_stats(),
            fitness_statistics=self.aggregate_fitness_stats()
        )

# Example usage:
# analyst = Analyst(list_of_experiment_data_models)
# aggregated_stats = analyst.compute_aggregated_statistics()
