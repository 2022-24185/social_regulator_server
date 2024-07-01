import unittest
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch
from neuroevolution.evolution.reproduction import MixedGenerationReproduction
from neat.config import Config
from neat.genome import DefaultGenome
from neuroevolution.evolution.species import MixedGenerationSpecies
from neuroevolution.evolution.stagnation import MixedGenerationStagnation
from neuroevolution.evolution.offspring_generator import OffspringGenerator

class TestableReproduction(MixedGenerationReproduction):
    def __init__(self, config, stagnation) -> None:
        self.mock_offspring_generator = MagicMock(spec=OffspringGenerator)
        self.mock_offspring_generator.create_offspring.return_value = {1: MagicMock(spec=DefaultGenome), 2: MagicMock(spec=DefaultGenome)}
        super().__init__(config, stagnation)
        self.elites = MagicMock()
    
    def create_offspring_generator(self, g_type, g_config, r_config) -> OffspringGenerator:
        return self.mock_offspring_generator

class TestMixedGenerationReproduction(TestCase):
    def setUp(self):
        self.config_mock = Mock(spec=Config)
        self.config_mock.genome_config = Mock(spec=Config)
        self.config_mock.reproduction_config = Mock(spec=Config)
        self.config_mock.reproduction_config.elitism = 1
        self.config_mock.reproduction_config.survival_threshold = 0.2
        self.config_mock.reproduction_config.min_species_size = 9
        self.stagnation_mock = Mock(spec=MixedGenerationStagnation)
        self.reproduction = TestableReproduction(self.config_mock, self.stagnation_mock)

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
        self.reproduction.mock_offspring_generator.create_without_parents.return_value = {1: MagicMock(spec=DefaultGenome, key=1), 2: MagicMock(spec=DefaultGenome, key=2), 3: MagicMock(spec=DefaultGenome, key=3)}
        new_genomes = self.reproduction.create_new_genomes(num_genomes)
        self.assertEqual(len(new_genomes), num_genomes)
        for key, genome in new_genomes.items():
            self.assertIsInstance(genome, DefaultGenome)
            self.assertIn(key, self.reproduction.ancestors)
            self.assertEqual(self.reproduction.ancestors[key], tuple())

    def test_reproduce_evaluated(self):
        active_species = [MagicMock(spec=MixedGenerationSpecies), MagicMock(spec=MixedGenerationSpecies)]
        active_species[0].get_sorted_by_fitness.return_value = [MagicMock(spec=DefaultGenome, key=1), MagicMock(spec=DefaultGenome, key=2), MagicMock(spec=DefaultGenome, key=3)]
        active_species[1].get_sorted_by_fitness.return_value = [MagicMock(spec=DefaultGenome, key=4), MagicMock(spec=DefaultGenome, key=5), MagicMock(spec=DefaultGenome, key=6)]
        selected_genome_ids = [1, 2, 3]
        with patch('neuroevolution.evolution.reproduction.SpeciesReproduction') as mock_species_reproduction:
            instance = mock_species_reproduction.return_value
            instance.process_dying_parents.return_value = 2
            new_population = self.reproduction.reproduce_evaluated(active_species, selected_genome_ids)
            self.assertEqual(len(new_population), 2)
            mock_species_reproduction.assert_called_once()

    def test_get_minimum_species_size(self):
        self.config_mock.reproduction_config.min_species_size = 2
        self.config_mock.reproduction_config.elitism = 1
        self.assertEqual(self.reproduction.get_minimum_species_size(), 2)
        self.config_mock.reproduction_config.elitism = 3
        self.assertEqual(self.reproduction.get_minimum_species_size(), 3)

if __name__ == '__main__':
    unittest.main()