import unittest
from itertools import count
from unittest.mock import MagicMock
from unittest.mock import patch

from neuroevolution.evolution.species_reproduction import SpeciesReproduction
from neuroevolution.evolution.species import MixedGenerationSpecies
from neuroevolution.evolution.fitness_manager import FitnessManager
from neuroevolution.evolution.offspring_generator import OffspringGenerator

class TestableSpeciesReproduction(SpeciesReproduction):
    def __init__(self, active_species, selected_genome_ids, min_species_size, config) -> None:
        self.mock_fitness_manager = MagicMock(spec=FitnessManager)
        self.mock_offspring_generator = MagicMock(spec=OffspringGenerator)
        super().__init__(active_species, selected_genome_ids, min_species_size, config)
        self.elites = MagicMock()

    def create_fitness_manager(self):
        return self.mock_fitness_manager
    
    def create_offspring_generator(self, g_type, g_config, r_config) -> OffspringGenerator:
        return self.mock_offspring_generator

class TestSpeciesReproduction(unittest.TestCase):
    def setUp(self):
        self.reprod_config = MagicMock()
        self.reprod_config.survival_threshold = 0.5
        self.active_species = [MagicMock(spec=MixedGenerationSpecies) for _ in range(3)]
        self.selected_genome_ids = [1, 2, 3]
        self.min_species_size = 2
        self.species_reproduction = TestableSpeciesReproduction(self.active_species, self.selected_genome_ids, self.min_species_size, self.reprod_config)

    def test_init(self):
        self.assertEqual(self.species_reproduction.min_species_size, 2)
        self.assertEqual(self.species_reproduction.active_species, self.active_species)
        self.assertEqual(self.species_reproduction.evaluated_genome_ids, self.selected_genome_ids)
        self.assertIsInstance(self.species_reproduction.fitness_collector, FitnessManager)

    def test_create_fitness_manager(self):
        self.species_reproduction.create_fitness_manager()
        self.assertIsInstance(self.species_reproduction.fitness_collector, FitnessManager)

    def test_get_total_adjusted_fitness(self):
        for species in self.active_species:
            species.adjusted_fitness = 1.0
        self.assertEqual(self.species_reproduction.get_total_adjusted_fitness(), 1.0)

    def test_get_total_death_counts(self):
        for i, species in enumerate(self.active_species):
            species.dying_count = i
        self.assertEqual(self.species_reproduction.get_total_death_counts(), sum(range(3)))

    def test_get_evaluated_genome_ids(self):
        self.assertEqual(self.species_reproduction.get_evaluated_genome_ids(), self.selected_genome_ids)

    def test_process_dying_parents(self):
        species = MagicMock(spec=MixedGenerationSpecies, members=MagicMock())
        sorted_parents = [(1, MagicMock()), (2, MagicMock())]
        elites = {1: MagicMock()}
        num_dying = self.species_reproduction.process_dying_parents(species, sorted_parents, elites)
        self.assertEqual(num_dying, 1)

    def test_process_dying_parents_no_elites(self):
        species = MagicMock(spec=MixedGenerationSpecies, key=1, members = [MagicMock() for _ in range(3)])
        sorted_parents = [(1, MagicMock()), (2, MagicMock())]
        elites = {}
        num_dying = self.species_reproduction.process_dying_parents(species, sorted_parents, elites)
        self.assertEqual(num_dying, 2)

    def test_adjust_fitnesses(self):
        with patch.object(self.species_reproduction.fitness_collector, 'adjust_fitnesses') as mock_adjust:
            self.species_reproduction.adjust_fitnesses()
            mock_adjust.assert_called_once()

    def test_normalize_spawn_counts(self):
        # Assuming min_species_size might be set to 4 in the setUp method.
        self.species_reproduction.min_species_size = 4
        normalized_counts = self.species_reproduction.normalize_spawn_counts(10, [1, 2, 3])
        expected_counts = [4, 4, 5]  # As calculated based on your logic and min_species_size.
        self.assertEqual(normalized_counts, expected_counts, "Normalized counts should match expected distribution.")

    def test_normalize_spawn_counts_less_dying_than_deficit(self):
        self.species_reproduction.min_species_size = 1
        normalized_counts = self.species_reproduction.normalize_spawn_counts(5, [2, 3, 4])
        expected_counts = [1, 2, 2]  # Should only replace the number of dying population despite bigger deficit
        self.assertEqual(normalized_counts, expected_counts, "Should assign minimum possible spawn counts.")

    def test_normalize_spawn_counts_zero_deficits(self):
        self.species_reproduction.min_species_size = 1
        normalized_counts = self.species_reproduction.normalize_spawn_counts(10, [0, 0, 0])
        expected_counts = [1, 1, 1]  # Minimum size should still be assigned
        self.assertEqual(normalized_counts, expected_counts, "Zero deficits should still assign minimum spawns.")

    def test_normalize_spawn_counts_high_min_species_size(self):
        self.species_reproduction.min_species_size = 5
        normalized_counts = self.species_reproduction.normalize_spawn_counts(10, [1, 1, 1])
        expected_counts = [5, 5, 5]  # Minimum size overrides the calculated sizes
        self.assertEqual(normalized_counts, expected_counts, "High minimum species size should dominate the spawn counts.")

    def test_compute_offspring_counts(self):
        with patch.object(self.species_reproduction, 'get_total_adjusted_fitness', return_value=1.0), \
             patch.object(self.species_reproduction, 'get_total_death_counts', return_value=10):
            counts = self.species_reproduction.compute_offspring_counts()
            self.assertEqual(len(counts), len(self.active_species))