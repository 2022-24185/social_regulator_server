import unittest
from unittest.mock import Mock
from neat.genome import DefaultGenome
from neat.config import Config
from neuroevolution.evolution.population_manager import PopulationManager
from neuroevolution.evolution.speciation import Speciation

config = Mock(spec=Config)
config.genome_config = Mock()
config.reproduction_config = Mock()
config.species_set_config = Mock()

class TestablePopulationManager(PopulationManager):
    def __init__(self, *args, **kwargs):
        self.mock_speciation = Mock(spec=Speciation, update_speciation = Mock())
        self.mock_speciation.species_set = Mock()
        super().__init__(*args, **kwargs)

    def create_speciation(self, config):
        return self.mock_speciation

class TestPopulationManagerInitialization(unittest.TestCase):
    def setUp(self):
        self.config = config
        self.config.species_set_config.compatibility_threshold = 3.0
        self.manager = PopulationManager(self.config)

    def test_initial_conditions(self):
        self.assertEqual(self.manager.population, {})
        self.assertEqual(self.manager.generation, 0)
        self.assertIsInstance(self.manager.speciation, Speciation)

class TestPopulationManagement(unittest.TestCase):
    def setUp(self):
        self.config = config
        self.config.species_set_config.compatibility_threshold = 3.0
        
        self.manager = TestablePopulationManager(self.config)
        self.new_population = {1: DefaultGenome(1), 2: DefaultGenome(2)}

    def test_set_new_population(self):
        self.manager.set_new_population(self.new_population)
        self.assertEqual(self.manager.population, self.new_population)
        self.assertEqual(self.manager.available_genomes, list(self.new_population.keys()))

    def test_update_generation(self):
        self.manager.set_new_population({1: DefaultGenome(1), 2: DefaultGenome(2)})
        offspring = {3: DefaultGenome(3)}
        self.manager.update_generation(offspring)
        self.assertEqual(self.manager.generation, 1)
        self.assertIn(3, self.manager.population)

    def test_update_genome(self): 
        self.manager.set_new_population({1: DefaultGenome(1), 2: DefaultGenome(2)})
        genome_id = 1
        data = Mock()
        self.manager.update_genome_data(genome_id, data)
        self.assertEqual(self.manager.population[genome_id].data, data)

class TestGenomeHandling(unittest.TestCase):
    def setUp(self):
        self.config = config
        self.config.species_set_config.compatibility_threshold = 3.0
        self.manager = PopulationManager(self.config)
        self.manager.set_new_population({1: DefaultGenome(1), 2: DefaultGenome(2)})

    def test_mark_genome_as_unavailable(self):
        self.manager.mark_genome_as_unavailable(1)
        self.assertNotIn(1, self.manager.available_genomes)

    def test_mark_genome_as_unavailable_error(self): 
        with self.assertRaises(ValueError):
            self.manager.mark_genome_as_unavailable(3)

    def test_get_random_available_genome(self):
        genome = self.manager.get_random_available_genome()
        self.assertIsInstance(genome, DefaultGenome)
        self.assertNotIn(genome.key, self.manager.available_genomes)

    def test_no_more_genomes_raises(self):
        self.manager.mark_genome_as_unavailable(1)
        self.manager.mark_genome_as_unavailable(2)
        with self.assertRaises(RuntimeError):
            self.manager.get_random_available_genome()

    def test_get_available(self):
        self.manager.mark_genome_as_unavailable(1)
        self.manager.mark_genome_as_unavailable(2)
        genomes = self.manager.get_available()
        self.assertEqual(len(genomes), 0)

class TestStagnationHandling(unittest.TestCase):
    def setUp(self):
        self.config = config
        self.config.species_set_config.compatibility_threshold = 3.0
        self.config.stagnation_threshold = 5
        self.manager = TestablePopulationManager(self.config)
        self.manager.set_new_population({1: DefaultGenome(1), 2: DefaultGenome(2)})
        self.stagnation = Mock()
        self.stagnation.update.return_value = {1: True, 2: False}

    def test_update_stagnation(self):
        eval_ids = [1, 2]
        self.manager.update_stagnation(self.stagnation, eval_ids)
        self.manager.mock_speciation.update_stagnant_species.assert_called_with({1: True, 2: False})

    def test_get_active_species(self):
        eval_ids = [1, 2]
        self.manager.get_active_species(self.stagnation, eval_ids)
        self.manager.mock_speciation.update_stagnant_species.assert_called_with({1: True, 2: False})
        self.manager.mock_speciation.species_set.get_active_species.assert_called()

if __name__ == '__main__':
    unittest.main()
