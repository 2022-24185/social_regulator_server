import unittest, sys
from unittest.mock import MagicMock
from neuroevolution.evolution.stagnation import MixedGenerationStagnation

class TestMixedGenerationStagnationMethods(unittest.TestCase):
    def setUp(self):
        self.config_mock = MagicMock()
        self.config_mock.species_fitness_func = "mean"
        self.config_mock.max_stagnation = 15
        self.config_mock.species_elitism = 0
        self.reporters_mock = MagicMock()
        self.stagnation = MixedGenerationStagnation(self.config_mock, self.reporters_mock)

    def test_parse_config_valid_parameters(self):
        param_dict = {"species_fitness_func": "max", "max_stagnation": 10, "species_elitism": 2}
        config = MixedGenerationStagnation.parse_config(param_dict)
        self.assertEqual(config.species_fitness_func, "max") # pylint: disable=no-member
        self.assertEqual(config.max_stagnation, 10) # pylint: disable=no-member
        self.assertEqual(config.species_elitism, 2) # pylint: disable=no-member

    def test_parse_config_with_defaults(self):
        param_dict = {}
        config = MixedGenerationStagnation.parse_config(param_dict)
        self.assertEqual(config.species_fitness_func, "mean") # pylint: disable=no-member
        self.assertEqual(config.max_stagnation, 15) # pylint: disable=no-member
        self.assertEqual(config.species_elitism, 0) # pylint: disable=no-member

    def test_init_invalid_species_fitness_func(self):
        self.config_mock.species_fitness_func = "invalid_func"
        with self.assertRaises(RuntimeError):
            MixedGenerationStagnation(self.config_mock, self.reporters_mock)

    def testcalculate_prev_fitness_with_history(self):
        species_mock = MagicMock(fitness_history=[1, 2, 3])
        result = self.stagnation.calculate_prev_fitness(species_mock)
        self.assertEqual(result, 3)

    def testcalculate_prev_fitness_empty_history(self):
        species_mock = MagicMock(fitness_history=[])
        result = self.stagnation.calculate_prev_fitness(species_mock)
        self.assertEqual(result, -sys.float_info.max)

    def testupdate_species_fitness(self):
        species_mock = MagicMock()
        species_mock.fitness_history = []
        species_mock.get_subset_of_fitnesses.return_value = [1, 2, 3]
        self.stagnation.species_fitness_func = MagicMock(return_value=2)
        self.stagnation.update_species_fitness(species_mock, [1, 2, 3])
        self.assertEqual(species_mock.fitness, 2)
        self.assertIn(2, species_mock.fitness_history)

    def test_is_species_stagnant(self):
        species_mock = MagicMock(last_improved=0)
        result = self.stagnation._is_species_stagnant(species_mock, 20, 4, 5)
        self.assertTrue(result)

    def test_species_active_due_to_elitism(self):
        species_mock = MagicMock(last_improved=0)
        self.config_mock.species_elitism = 2
        result = self.stagnation._is_species_stagnant(species_mock, 20, 4, 5)
        self.assertFalse(result)

    def test_identify_stagnant_species(self):
        species_data = [(1, MagicMock(last_improved=0, fitness=2)), (2, MagicMock(last_improved=19, fitness=3))]
        self.config_mock.species_elitism = 0
        result = self.stagnation._identify_stagnant_species(species_data, 21)
        self.assertTrue(result[1])
        self.assertFalse(result[2])

    def test_update_fitness_history_for_species_empty_species_set(self):
        species_set_mock = MagicMock(species={})
        result = self.stagnation._update_fitness_history_for_species(species_set_mock, [], 1)
        self.assertEqual(result, [])

    def test_update_fitness_history_for_species_single_species_no_previous_fitness(self):
        species_mock = MagicMock(fitness_history=[], get_fitnesses=MagicMock(return_value=[1]))
        species_set_mock = MagicMock(species={1: species_mock})
        self.stagnation.species_fitness_func = MagicMock(return_value=2)
        result = self.stagnation._update_fitness_history_for_species(species_set_mock, [1], 1)
        self.assertEqual(result, [(1, species_mock)])
        self.assertEqual(species_mock.fitness_history, [2])
        self.assertEqual(species_mock.last_improved, 1)

    def test_update_fitness_history_for_species_multiple_species_with_previous_fitness(self):
        species_mock1 = MagicMock(fitness_history=[1], get_fitnesses=MagicMock(return_value=[2]), last_improved=0)
        species_mock2 = MagicMock(fitness_history=[2], get_fitnesses=MagicMock(return_value=[1]), last_improved=0)
        species_set_mock = MagicMock(species={1: species_mock1, 2: species_mock2})
        self.stagnation.species_fitness_func = MagicMock(side_effect=[3, 1])
        result = self.stagnation._update_fitness_history_for_species(species_set_mock, [1, 2], 2)
        self.assertEqual(result, [(1, species_mock1), (2, species_mock2)])
        self.assertEqual(species_mock1.fitness_history, [1, 3])
        self.assertEqual(species_mock2.fitness_history, [2, 1])
        self.assertEqual(species_mock1.last_improved, 2)
        self.assertEqual(species_mock2.last_improved, 0)  # Assuming last_improved was initialized to 0

    def test_update_fitness_history_for_species_improvement_in_fitness(self):
        species_mock = MagicMock(fitness_history=[1], get_fitnesses=MagicMock(return_value=[3]))
        species_set_mock = MagicMock(species={1: species_mock})
        self.stagnation.species_fitness_func = MagicMock(return_value=4)
        result = self.stagnation._update_fitness_history_for_species(species_set_mock, [1], 3)
        self.assertEqual(result, [(1, species_mock)])
        self.assertEqual(species_mock.fitness_history, [1, 4])
        self.assertEqual(species_mock.last_improved, 3)

    def test_update_fitness_history_for_species_no_improvement_in_fitness(self):
        species_mock = MagicMock(fitness_history=[5], get_fitnesses=MagicMock(return_value=[2]), last_improved=0)
        species_set_mock = MagicMock(species={1: species_mock})
        self.stagnation.species_fitness_func = MagicMock(return_value=2)
        result = self.stagnation._update_fitness_history_for_species(species_set_mock, [1], 4)
        self.assertEqual(result, [(1, species_mock)])
        self.assertEqual(species_mock.fitness_history, [5, 2])
        self.assertEqual(species_mock.last_improved, 0)  # Assuming last_improved was initialized to 0

if __name__ == '__main__':
    unittest.main()