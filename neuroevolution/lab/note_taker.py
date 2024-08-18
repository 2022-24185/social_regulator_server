from typing import Dict, Any, List
import pandas as pd
from neat.reporting import BaseReporter
from neat.genome import DefaultGenome

from neuroevolution.data_models.experiment_data_models import ExperimentDataModel, GenerationSummaryData, FitnessStats

import pandas as pd
from typing import List, Optional
from pydantic import BaseModel

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
        items = getattr(model, field)
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

    def start_generation(self, generation_summary: GenerationSummaryData):
        """Called at the start of each generation."""
        self.current_generation = generation_summary.generation
        self.current_data.generations.append(generation_summary)

    def end_generation(self, generation_update: GenerationSummaryData):
        """Called at the end of each generation."""
        this_gen = generation_update.generation
        generation_summary = self.current_data.generations[this_gen]
        generation_summary.population_end_size = generation_update.population_end_size
        generation_summary.active_species_count = generation_update.active_species_count
        print(f"Generation {this_gen} complete.")
        print(f"Species: {generation_update.active_species_count}")
        print(f"Population size: {generation_update.population_end_size}")

    def post_evaluate(self, generation: int, fitness_stats: FitnessStats):
        generation_summary = self.current_data.generations[generation]
        generation_summary.fitness_summary = fitness_stats
        print(f"Best fitness: {fitness_stats.best_genome_fitness}")
        print(f"Worst fitness: {fitness_stats.worst_genome_fitness}")
        print(f"Mean fitness: {fitness_stats.mean_fitness}")
        print(f"Median fitness: {fitness_stats.median_fitness}")
        print(f"Fitness variance: {fitness_stats.fitness_variance}")
        print(f"Fitness quartiles: {fitness_stats.fitness_quartiles}")

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
        self.data_models.append(self.current_data)
        self.current_data = None

    def get_statistics(self): 
        pass

    def get_data(self) -> List[ExperimentDataModel]:
        return self.data_models
    
    def save_to_file(self, filename: str):
        """Saves the collected data to a file."""
        with open(filename, 'w') as f:
            f.write(self.current_data.model_dump_json())

    def __repr__(self):
        return f"NoteTaker(experiment_id={self.experiment_id})"