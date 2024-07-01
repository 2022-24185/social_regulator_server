import unittest
from unittest.mock import MagicMock, patch
from neuroevolution.evolution.offspring_generator import OffspringGenerator
from neat.genome import DefaultGenome

class TestOffspringGenerator(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.config.genome_type = DefaultGenome
        self.genome_config = MagicMock()
        self.reproduction_config = MagicMock()
        self.config.survival_threshold = 0.5
        self.config.min_species_size = 2
        self.offspring_generator = OffspringGenerator(self.config.genome_type, self.genome_config, self.reproduction_config)

    def test_init(self):
        self.assertEqual(self.offspring_generator.genome_factory.genome_config, self.genome_config)

    @patch('neat.genome.DefaultGenome.mutate')
    @patch('neat.genome.DefaultGenome.configure_crossover')
    @patch('neuroevolution.evolution.offspring_generator.DefaultGenome')
    def test_mate_parents(self, mock_genome, mock_configure_crossover, mock_mutate):
        parent1 = mock_genome()
        parent2 = mock_genome()
        child_id, child = self.offspring_generator.mate_parents(parent1, parent2)
        self.assertIsInstance(child_id, int)
        self.assertIsInstance(child, DefaultGenome)

    def test_create_offspring_insufficient_parents(self):
        with self.assertRaises(ValueError):
            self.offspring_generator.create_offspring([], 1, 2)

    @patch('neat.genome.DefaultGenome.mutate')
    @patch('neat.genome.DefaultGenome.configure_crossover')
    @patch('neuroevolution.evolution.offspring_generator.random.sample')
    @patch('neuroevolution.evolution.offspring_generator.DefaultGenome')
    def test_create_offspring(self, mock_genome, mock_sample, mock_configure, mock_mutate):
        parent1 = (1, mock_genome())
        parent2 = (2, mock_genome())
        pick_top = 2
        mock_sample.return_value = [parent1, parent2]
        offspring = self.offspring_generator.create_offspring([parent1, parent2], 3, pick_top)
        self.assertEqual(len(offspring), 3)
        for child_id, child in offspring.items():
            self.assertIsInstance(child_id, int)
            self.assertIsInstance(child, DefaultGenome)

if __name__ == '__main__':
    unittest.main()