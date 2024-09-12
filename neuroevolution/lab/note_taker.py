from typing import Dict, Any, List
import pandas as pd
from neat.reporting import BaseReporter
from neat.genome import DefaultGenome

from neuroevolution.data_models.experiment_data_models import ExperimentDataModel, GenerationSummaryData, FitnessStats, FitnessContributionData, SelectiveEvaluationData, PoolStatusData

import pandas as pd
from typing import List, Optional
from pydantic import BaseModel
import logging
from uuid import uuid4
import time

def pydantic_to_dataframe(data: List[BaseModel], fields: List[str]) -> pd.DataFrame:
    """
    Convert a list of Pydantic models to a Pandas DataFrame.

    Args:
        data: List of Pydantic models.
        fields: List of fields to extract from each model.

    Returns:
        A Pandas DataFrame containing the specified fields from the Pydantic models.
    """
    data_dict = [{field: getattr(item, field, None) for field in fields} for item in data]
    return pd.DataFrame(data_dict)

def collect_data(models: List[BaseModel], field: str, fields: List[str], nested_field: Optional[str] = None) -> pd.DataFrame:
    """
    Collect data from a list of Pydantic models and convert it to a Pandas DataFrame.

    Args:
        models: List of Pydantic models.
        field: The field to collect from each model.
        fields: List of fields to extract from the models.
        nested_field: Optional nested field to extract if present.

    Returns:
        A Pandas DataFrame containing the collected data.
    """
    data = []
    for model in models:
        items = getattr(model, field, [])
        if nested_field:
            items = [getattr(item, nested_field) for item in items if getattr(item, nested_field, None)]
        data.extend(items)
    return pydantic_to_dataframe(data, fields)


class NoteTaker(BaseReporter):
    """Reports several experiments to the Lab."""
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.current_generation = 0
        self.current_data = ExperimentDataModel()
        self.data_models: List[ExperimentDataModel] = []
        logging.info(f"NoteTaker initialized for experiment {experiment_id}")

    def start_generation(self, generation_summary: GenerationSummaryData):
        """Called at the start of each generation."""
        try:
            self.current_generation = generation_summary.generation
            self.current_data.generations.append(generation_summary)
            logging.info(f"Started generation {self.current_generation} for experiment {self.experiment_id}")
        except Exception as e:
            logging.error(f"Error starting generation {self.current_generation}: {e}")
            raise

    def end_generation(self, generation_update: GenerationSummaryData):
        """Called at the end of each generation."""
        generation_summary = self.current_data.generations[self.current_generation]
        generation_summary.population_end_size = generation_update.population_end_size
        generation_summary.active_species_count = generation_update.active_species_count
        logging.info(f"Generation {self.current_generation} completed. Population size: {generation_update.population_end_size}")

    def post_evaluate(self, generation: int, fitness_stats: FitnessStats):
        generation_summary = self.current_data.generations[generation]
        generation_summary.fitness_summary = fitness_stats
        print(f"Best fitness: {fitness_stats.best_genome_fitness}")
        print(f"Worst fitness: {fitness_stats.worst_genome_fitness}")
        print(f"Mean fitness: {fitness_stats.mean_fitness}")
        print(f"Median fitness: {fitness_stats.median_fitness}")
        print(f"Fitness variance: {fitness_stats.fitness_variance}")
        print(f"Fitness quartiles: {fitness_stats.fitness_quartiles}")

    def track_evaluated_genome(self, genome_id: int, fitness: float, contributions: List[Dict[str, Any]] = None):
        """Track each evaluated genome along with its fitness and contributions."""
        try:
            fitness_contributions = [
                FitnessContributionData(**contribution) for contribution in contributions
            ] if contributions else []
            evaluation_data = SelectiveEvaluationData(
                event_id=str(uuid4()),
                timestamp=time.time(),
                genome_id=genome_id,
                generation=self.current_generation,
                fitness=fitness,
                fitness_contributions=fitness_contributions
            )
            self.current_data.selective_evaluations.append(evaluation_data)
            logging.info(f"Tracked genome {genome_id} with fitness {fitness}.")
        except Exception as e:
            logging.error(f"Error tracking genome {genome_id}: {e}")
            raise

    def update_pool_overview(self, available: int, interacting: int, evaluated: int):
        """Update the current pool overview (available, interacting, evaluated)."""
        try:
            logging.info(f"Updating pool overview for experiment {self.experiment_id}.")
            self.current_data.pool_status.available_individuals = available
            self.current_data.pool_status.interacting_individuals = interacting
            self.current_data.pool_status.evaluated_individuals = evaluated
        except Exception as e:
            logging.error(f"Error updating pool overview for experiment {self.experiment_id}: {e}")
            raise

    def post_reproduction(self, config, population, species):
        pass

    def complete_extinction(self):
        pass

    def found_solution(self, config, generation, best):
        pass

    def species_stagnant(self, sid, species):
        pass

    def info(self, msg):
        pass

    def end_experiment(self):
        """Finalize and store experiment data when the experiment ends."""
        self.data_models.append(self.current_data)
        self.current_data = ExperimentDataModel()
        logging.info(f"Experiment {self.experiment_id} ended.")

    def get_statistics(self) -> Dict[str, Any]:
        """Retrieve summarized statistics for the experiment."""
        try:
            stats = {
                "experiment_id": self.experiment_id,
                "total_generations": len(self.current_data.generations),
                "fitness_statistics": [gen.fitness_summary.model_dump() for gen in self.current_data.generations if gen.fitness_summary]
            }
            logging.info(f"Statistics for experiment {self.experiment_id} retrieved.")
            return stats
        except Exception as e:
            logging.error(f"Error retrieving statistics for experiment {self.experiment_id}: {e}")
            raise

    def get_data(self) -> ExperimentDataModel:
        """Retrieve all stored data models."""
        try:
            logging.info(f"Retrieving data for experiment {self.experiment_id}")
            return self.current_data
        except Exception as e:
            logging.error(f"Error retrieving data for experiment {self.experiment_id}: {e}")
            raise
    
    def save_to_file(self, filename: str):
        """Save the current experiment data to a file."""
        try:
            with open(filename, 'w') as f:
                f.write(self.current_data.model_dump_json())
            logging.info(f"Data saved to {filename}")
        except IOError as e:
            logging.error(f"Error saving data to file {filename}: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error saving data to file {filename}: {e}")
            raise

    def __repr__(self):
        return f"NoteTaker(experiment_id={self.experiment_id})"