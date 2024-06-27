import unittest
from neuroevolution.evolution.species import MixedGenerationSpecies, Member

class TestMixedGenerationSpecies(unittest.TestCase):

    def setUp(self):
        self.species = MixedGenerationSpecies(key=1, generation=0)

    def test_initialization(self):
        self.assertEqual(self.species.key, 1)
        self.assertTrue(self.species.active)
        self.assertEqual(self.species.created, 0)
        self.assertEqual(self.species.last_improved, 0)
        self.assertIsNone(self.species.representative)
        self.assertEqual(self.species.members, {})
        self.assertIsNone(self.species.fitness)
        self.assertIsNone(self.species.adjusted_fitness)
        self.assertEqual(self.species.dying_count, 0)
        self.assertEqual(self.species.expected_offspring, 0)
        self.assertEqual(self.species.fitness_history, [])

    def test_add_member(self):
        member = (2, 'genome_instance')
        self.species.add_member(member)
        self.assertIn(2, self.species.members)
        self.assertEqual(self.species.members[2], 'genome_instance')

    def test_set_representative(self):
        representative = ('genome_instance',)
        self.species.set_representative(representative)
        self.assertEqual(self.species.representative, representative)

    def test_get_representative_id(self):
        self.species.representative = (3, 'genome_instance')
        self.assertEqual(self.species.get_representative_id(), 3)

    def test_get_fitnesses(self):
        self.species.members = {1: MockGenome(0.5), 2: MockGenome(1.0)}
        self.assertEqual(self.species.get_fitnesses(), [0.5, 1.0])
        self.assertEqual(self.species.get_fitnesses([2]), [1.0])

    def test_get_sorted_by_fitness(self):
        self.species.members = {1: MockGenome(0.5), 2: MockGenome(1.0), 3: MockGenome(0.75)}
        sorted_members = self.species.get_sorted_by_fitness([1, 2, 3])
        self.assertEqual([member[0] for member in sorted_members], [2, 3, 1])

    def test_set_adjusted_fitness(self):
        self.species.set_adjusted_fitness(0.75)
        self.assertEqual(self.species.adjusted_fitness, 0.75)

    def test_mark_stagnant(self):
        self.species.mark_stagnant()
        self.assertFalse(self.species.active)

    def test_kill_members(self):
        self.species.members = {1: 'genome1', 2: 'genome2'}
        self.species.kill_members({1})
        self.assertNotIn(1, self.species.members)
        self.assertIn(2, self.species.members)

    def test_compute_expected_size_positive_total_adjusted_fitness(self):
        self.species.adjusted_fitness = 1.0
        # Case where the formula is used and the result is exactly min_species_size
        self.assertEqual(self.species.compute_expected_size(5, 2.0, 10), 5)
        # Case where the formula is used and the result is greater than min_species_size
        self.assertEqual(self.species.compute_expected_size(5, 1.0, 10), 10)

    def test_compute_expected_size_zero_total_adjusted_fitness(self):
        self.species.adjusted_fitness = 1.0
        # Case where total_adjusted_fitness is zero, should default to min_species_size
        self.assertEqual(self.species.compute_expected_size(5, 0, 10), 5)

    def test_compute_expected_size_result_less_than_min_species_size(self):
        self.species.adjusted_fitness = 0.1
        # Case where the formula's result is less than min_species_size
        self.assertEqual(self.species.compute_expected_size(5, 10.0, 10), 5)

    def test_compute_expected_size_result_greater_than_min_species_size(self):
        self.species.adjusted_fitness = 5.0
        # Case where the formula's result is greater than min_species_size
        self.assertEqual(self.species.compute_expected_size(5, 10.0, 100), 50)

    def test_compute_pop_deficit(self):
        # Initialize species object's dying_count for context
        self.species.dying_count = 5

        # Test case where expected size equals dying count
        self.assertEqual(self.species.compute_pop_deficit(5), 5, "Equal size should not change deficit")

        # Test case where expected size is greater than dying count
        self.species.dying_count = 4
        self.assertEqual(self.species.compute_pop_deficit(6), 5, "Expected increase in deficit not observed")

        # Test case where expected size is less than dying count
        self.species.dying_count = 6
        self.assertEqual(self.species.compute_pop_deficit(4), 5, "Expected decrease in deficit not observed")

        # Test rounding effect: size_diff results in positive rounding
        self.species.dying_count = 4
        self.assertEqual(self.species.compute_pop_deficit(7), 6, "Rounding up did not increase deficit as expected")

        # Test rounding effect: size_diff results in negative rounding
        self.species.dying_count = 6
        self.assertEqual(self.species.compute_pop_deficit(3), 4, "Rounding down did not decrease deficit as expected")

        # Test no rounding but size_diff > 0, should increase by 1
        self.species.dying_count = 4
        self.assertEqual(self.species.compute_pop_deficit(5), 5, "Expected minimal increase not observed")

        # Test no rounding but size_diff < 0, should decrease by 1
        self.species.dying_count = 6
        self.assertEqual(self.species.compute_pop_deficit(5), 5, "Expected minimal decrease not observed")

    def test_is_member(self):
        self.species.members = {1: 'genome1'}
        self.assertTrue(self.species.is_member(1))
        self.assertFalse(self.species.is_member(2))

    def test_is_active(self):
        self.assertTrue(self.species.is_active())
        self.species.active = False
        self.assertFalse(self.species.is_active())

class MockGenome:
    def __init__(self, fitness):
        self.fitness = fitness

if __name__ == '__main__':
    unittest.main()