import unittest
from unittest.mock import MagicMock, patch

from neat.math_util import mean
from neuroevolution.evolution.evaluation import Evaluation, CompleteExtinctionException

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.config.fitness_criterion = 'mean'
        self.config.no_fitness_termination = False
        self.fitness_function = MagicMock(return_value=1.0)
        self.evaluation = Evaluation(self.config, self.fitness_function, 10)

    def test_init(self):
        self.assertEqual(self.evaluation.config, self.config)
        self.assertEqual(self.evaluation.fitness_function, self.fitness_function)
        self.assertEqual(self.evaluation.evaluation_threshold, 10)
        self.assertEqual(self.evaluation.evaluated_genomes, {})
        self.assertIsNotNone(self.evaluation.summarizer)

    def test_get_fitness_summarizer_valid(self):
        self.assertEqual(self.evaluation.get_fitness_summarizer(), mean)

    def test_get_fitness_summarizer_invalid(self):
        self.config.fitness_criterion = 'invalid_criterion'
        with self.assertRaises(ValueError):
            self.evaluation.get_fitness_summarizer()

    @patch('neat.genome.DefaultGenome')
    def test_get_best(self, mock_genome):
        genome1 = mock_genome()
        genome1.fitness = 1
        genome2 = mock_genome()
        genome2.fitness = 2
        self.evaluation.evaluated_genomes = {genome1: genome1.fitness, genome2: genome2.fitness}
        best = self.evaluation.get_best()
        self.assertEqual(best, genome2)

    @patch('neat.genome.DefaultGenome')
    def test_evaluate(self, mock_genome):
        genome = mock_genome()
        self.evaluation.evaluate(1, genome)
        self.fitness_function.assert_called_once_with(genome, self.config)
        self.assertIn(genome, self.evaluation.evaluated_genomes.values())

    def test_threshold_reached(self):
        self.evaluation.evaluated_genomes = {i: 1.0 for i in range(11)}
        self.assertTrue(self.evaluation.threshold_reached())
        self.evaluation.evaluated_genomes = {i: 1.0 for i in range(10)}
        self.assertFalse(self.evaluation.threshold_reached())

    def test_get_evaluated(self):
        self.evaluation.evaluated_genomes = {1: 1.0, 2: 2.0}
        self.assertEqual(self.evaluation.get_evaluated(), [1, 2])

    def test_clear_evaluated(self):
        self.evaluation.evaluated_genomes = {1: 1.0, 2: 2.0}
        self.evaluation.clear_evaluated()
        self.assertEqual(self.evaluation.evaluated_genomes, {})

if __name__ == '__main__':
    unittest.main()