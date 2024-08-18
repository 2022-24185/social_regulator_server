
from manim import Text, Write, Scene, Transform
from neuroevolution.data_models.experiment_data_models import ExperimentDataModel

class ExperimentVisualizer(Scene):
    def __init__(self, experiment_data: ExperimentDataModel):
        self.experiment_data = experiment_data
        super().__init__()

    def visualize_generation_summary(self):
        for gen_summary in self.experiment_data.generations:
            generation_text = Text(f"Generation {gen_summary.generation}")
            population_text = Text(f"Population: Start={gen_summary.population_start_size}, End={gen_summary.population_end_size}")
            active_species_text = Text(f"Active Species: {gen_summary.active_species_count}")

            self.play(Write(generation_text))
            self.wait(1)
            self.play(Transform(generation_text, population_text))
            self.wait(1)
            self.play(Transform(generation_text, active_species_text))
            self.wait(1)
            self.clear()

    def visualize_fitness_stats(self):
        for fitness_stat in self.experiment_data.species_fitness_stats:
            fitness_text = Text(f"Species {fitness_stat.species_id} - Best Fitness: {fitness_stat.best_genome_fitness}")
            self.play(Write(fitness_text))
            self.wait(1)
            self.clear()

    def construct(self):
        self.visualize_generation_summary()
        self.visualize_fitness_stats()

# Example usage with Manim
# visualizer = ExperimentVisualizer(single_experiment_data_model)
# visualizer.render()
