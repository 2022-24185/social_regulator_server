import unittest
from unittest import TestCase, mock
from neuroevolution.evolution.elites import Elites
from neat.genome import DefaultGenome

class TestElites(TestCase):
    def setUp(self):
        self.elitism_config = 2
        self.elites = Elites(self.elitism_config)

    def test_init(self):
        self.assertEqual(self.elites.elitism, self.elitism_config)
        self.assertEqual(self.elites.elitism_count, 0)
        self.assertEqual(self.elites.non_elites, 0)

    def test_set_elitism_stats_with_higher_offspring_count(self):
        offspring_count = 10
        self.elites.set_elitism_stats(offspring_count)
        self.assertEqual(self.elites.elitism_count, offspring_count)
        self.assertEqual(self.elites.non_elites, offspring_count - self.elites.elitism_count)

    def test_set_elitism_stats_with_lower_offspring_count(self):
        offspring_count = 1
        self.elites.set_elitism_stats(offspring_count)
        self.assertEqual(self.elites.elitism_count, self.elitism_config)
        self.assertEqual(self.elites.non_elites, 0)

    def test_preserve(self):
        sorted_parents = [(1, DefaultGenome(1)), (2, DefaultGenome(2)), (3, DefaultGenome(3))]
        offspring_count = 2
        expected_population = {1: sorted_parents[0][1], 2: sorted_parents[1][1]}
        new_population = self.elites.preserve(sorted_parents, offspring_count)
        self.assertEqual(new_population, expected_population)

    def test_preserve_with_more_offspring_than_parents(self):
        sorted_parents = [(1, DefaultGenome(1)), (2, DefaultGenome(2))]
        offspring_count = 3
        expected_population = {1: sorted_parents[0][1], 2: sorted_parents[1][1]}
        new_population = self.elites.preserve(sorted_parents, offspring_count)
        self.assertEqual(new_population, expected_population)

if __name__ == '__main__':
    unittest.main()
