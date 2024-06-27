import unittest
from unittest import TestCase, mock
from neuroevolution.evolution.reproduction import MixedGenerationReproduction
from neat.config import Config
from neat.genome import DefaultGenome
from neuroevolution.evolution.species import MixedGenerationSpecies
from neuroevolution.evolution.stagnation import MixedGenerationStagnation

class TestMixedGenerationReproduction(TestCase):
    def setUp(self):
        self.config_mock = mock.Mock(spec=Config)
        self.config_mock.genome_config = mock.Mock(spec=Config)
        self.config_mock.reproduction_config = mock.Mock(spec=Config)
        self.config_mock.reproduction_config.elitism = 1
        self.config_mock.reproduction_config.survival_threshold = 0.2
        self.config_mock.reproduction_config.min_species_size = 9
        self.stagnation_mock = mock.Mock(spec=MixedGenerationStagnation)
        self.reproduction = MixedGenerationReproduction(self.config_mock, self.stagnation_mock)

    def test_parse_config(self):
        param_dict = {
            "elitism": 5,
            "survival_threshold": 0.3,
            "min_species_size": 10
        }
        result = MixedGenerationReproduction.parse_config(param_dict)
        self.assertEqual(result.elitism, 5) # pylint: disable=no-member
        self.assertEqual(result.survival_threshold, 0.3) # pylint: disable=no-member
        self.assertEqual(result.min_species_size, 10) # pylint: disable=no-member

    def test_create_new_genomes(self):
        num_genomes = 3
        with mock.patch.object(self.reproduction.genome_factory, 'create_genome', autospec=True) as mock_create_genome:
            mock_create_genome.side_effect = [DefaultGenome(i) for i in range(num_genomes)]
            new_genomes = self.reproduction.create_new_genomes(num_genomes)
            self.assertEqual(len(new_genomes), num_genomes)
            for key, genome in new_genomes.items():
                self.assertIsInstance(genome, DefaultGenome)
                self.assertIn(key, self.reproduction.ancestors)
                self.assertEqual(self.reproduction.ancestors[key], tuple())

    def test_reproduce_evaluated(self):
        active_species = [mock.Mock(spec=MixedGenerationSpecies) for _ in range(2)]
        selected_genome_ids = [1, 2, 3]
        with mock.patch('neuroevolution.evolution.reproduction.SpeciesReproduction') as mock_species_reproduction:
            instance = mock_species_reproduction.return_value
            instance.reproduce.return_value = {1: DefaultGenome(1), 2: DefaultGenome(2)}
            new_population = self.reproduction.reproduce_evaluated(active_species, selected_genome_ids)
            self.assertEqual(len(new_population), 2)
            mock_species_reproduction.assert_called_once()

    def test_get_minimum_species_size(self):
        self.reproduction.config.min_species_size = 2
        self.reproduction.config.elitism = 1
        self.assertEqual(self.reproduction.get_minimum_species_size(), 2)
        self.reproduction.config.elitism = 3
        self.assertEqual(self.reproduction.get_minimum_species_size(), 3)

if __name__ == '__main__':
    unittest.main()