import unittest
from unittest.mock import MagicMock
from neuroevolution.evolution.fitness_manager import FitnessManager

class TestFitnessManager(unittest.TestCase):
    def setUp(self):
        self.fm = FitnessManager()

    def test_collect_new_fitnesses_with_active_species_no_evaluated_genomes(self):
        species = MagicMock()
        species.members = {1: MagicMock(fitness=10)}
        self.fm.collect_new_fitnesses([species], [])
        self.assertEqual(self.fm.all_new_fitnesses, [], "No genomes should be evaluated, hence no new fitnesses.")

    def test_collect_new_fitnesses_with_active_species_and_evaluated_genomes(self):
        species1 = MagicMock()
        species1.members = {1: MagicMock(fitness=10), 2: MagicMock(fitness=20)}
        species2 = MagicMock()
        species2.members = {3: MagicMock(fitness=30)}
        self.fm.collect_new_fitnesses([species1, species2], [1, 3])
        self.assertIn(10, self.fm.all_new_fitnesses)
        self.assertIn(30, self.fm.all_new_fitnesses)
        self.assertNotIn(20, self.fm.all_new_fitnesses)

    def test_adjust_fitnesses_no_active_species(self):
        adjusted_fitnesses = self.fm.adjust_fitnesses([], [])
        self.assertEqual(adjusted_fitnesses, [], "No species means no fitnesses to adjust.")

    def test_adjust_fitnesses_with_active_species_no_evaluated_genomes(self):
        species = MagicMock()
        species.get_fitnesses.return_value = []
        species.set_adjusted_fitness = MagicMock()
        adjusted_fitnesses = self.fm.adjust_fitnesses([species], [])
        self.assertEqual(adjusted_fitnesses, [], "No genomes evaluated, so no adjusted fitnesses.")

    def test_adjust_fitnesses_with_active_species_and_evaluated_genomes(self):
        species1 = MagicMock()
        species1.get_fitnesses.return_value = [10, 20]
        species2 = MagicMock()
        species2.get_fitnesses.return_value = [30, 40]
        species1.members = {1: MagicMock(fitness=10), 2: MagicMock(fitness=20)}
        species2.members = {3: MagicMock(fitness=30), 4: MagicMock(fitness=40)}
        adjusted_fitnesses = self.fm.adjust_fitnesses([species1, species2], [1, 2, 3, 4])
        self.assertGreater(len(adjusted_fitnesses), 0, "Adjusted fitnesses should be calculated.")

    def test_adjust_fitnesses_with_single_species_single_genome(self):
        species = MagicMock()
        species.members = {1: MagicMock(fitness=50)}
        species.get_fitnesses.return_value = [50]
        adjusted_fitnesses = self.fm.adjust_fitnesses([species], [1])
        self.assertEqual(len(adjusted_fitnesses), 1, "One adjusted fitness should be calculated.")

if __name__ == '__main__':
    unittest.main()
