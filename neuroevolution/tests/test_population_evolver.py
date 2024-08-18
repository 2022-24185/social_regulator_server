import unittest
from unittest.mock import Mock, MagicMock, patch
from neat.config import Config
from neat.genome import DefaultGenome
from neuroevolution.evolution.population_evolver import Evolution, CompleteExtinctionException
from neuroevolution.evolution.population_manager import PopulationManager
from neuroevolution.evolution.stagnation import MixedGenerationStagnation
from neuroevolution.evolution.reproduction import MixedGenerationReproduction

class TestablePopulationEvolver(Evolution):
    def __init__(self, *args, **kwargs):
        self.mock_reporters = Mock()
        self.mock_stagnation = Mock()
        self.mock_reproduction = Mock()
        self.mock_pop_manager = Mock()
        self.mock_evaluation = Mock()
        super().__init__(*args, **kwargs)

    def create_reporter_set(self):
        return self.mock_reporters

    def create_stagnation(self):
        return self.mock_stagnation

    def create_reproduction(self):
        return self.mock_reproduction

    def create_population_manager(self):
        return self.mock_pop_manager

    def create_evaluation(self, fitness_function, evaluation_threshold):
        return self.mock_evaluation

class TestPopulationEvolverInitialization(unittest.TestCase):
    def setUp(self):
        self.config = Mock(spec=Config)
        self.config.compatibility_threshold = 3.0
        self.config.fitness_criterion = Mock()
        self.config.no_fitness_termination = Mock()
        self.fitness_function = Mock()
        self.evolver = TestablePopulationEvolver(self.config, self.fitness_function)
        self.evolver.mock_pop_manager.population = {1: Mock(), 2: Mock()}

    def test_population_evolver_initialization(self):
        # Test that factory methods are called and set proper attributes
        self.assertIs(self.evolver.reporters, self.evolver.mock_reporters)
        self.assertIs(self.evolver.stagnation, self.evolver.mock_stagnation)
        self.assertIs(self.evolver.reproduction, self.evolver.mock_reproduction)
        self.assertIs(self.evolver.pm, self.evolver.mock_pop_manager)
        self.assertIs(self.evolver.evaluation, self.evolver.mock_evaluation)

    def test_create_new_population(self):
        # Setup: Mock the return value from create_new_genomes
        self.config.pop_size = 2
        self.evolver.mock_reproduction.create_new_genomes.return_value = {1: 'genome1', 2: 'genome2'}
        
        # Action: Call the method under test
        self.evolver.create_new_population()
        
        # Assert: Check that create_new_genomes was called with the correct population size
        self.evolver.mock_reproduction.create_new_genomes.assert_called_once_with(self.config.pop_size)
        # Assert: Check that the population manager's set_new_population was called with the correct data
        self.evolver.mock_pop_manager.set_new_population.assert_called_once_with({1: 'genome1', 2: 'genome2'})

    def test_handle_receive_user_data_with_valid_data(self):
        # Setup: Create mock user data
        user_data = MagicMock(genome_id=1)
        
        # Setup: Mock evaluation.threshold_reached to return True
        self.evolver.mock_evaluation.threshold_reached.return_value = True
        
        # Action: Call the method under test
        self.evolver.receive_evaluation(user_data)
        
        # Assert: Check that update_genome_data was called correctly
        self.evolver.mock_pop_manager.update_genome_data.assert_called_once_with(1, user_data)
        # Assert: Check that the evaluation was called
        self.evolver.mock_evaluation.evaluate.assert_called()
        # Assert: Check that advance_population was called if threshold reached
        self.evolver.mock_evaluation.get_best.assert_called()

    def test_handle_receive_user_data_with_invalid_id(self):
        # Setup: Create mock user data with invalid ID
        user_data = MagicMock(genome_id=0)
        
        # Action: Call the method under test
        self.evolver.receive_evaluation(user_data)
        
        # Assert: Check that update_genome_data was not called
        self.evolver.mock_pop_manager.update_genome_data.assert_not_called()

    def test_advance_population_with_fitness_goal_not_met(self):
        # Setup: Mock get_best to return a genome with low fitness
        best_genome = MagicMock(fitness=0.5)
        self.evolver.mock_evaluation.get_best.return_value = best_genome
        
        # Setup: Config settings for no termination on fitness
        self.config.no_fitness_termination = False
        self.config.fitness_threshold = 0.6
        
        # Action: Call the method under test
        self.evolver.advance_population()
        
        # Assert: Check reproduction and generation update
        self.evolver.mock_reproduction.reproduce_evaluated.assert_called()
        self.evolver.mock_pop_manager.update_generation.assert_called()
        # Assert: Check no termination occurred
        self.evolver.mock_reporters.found_solution.assert_not_called()

    def test_advance_population_with_extinction(self):
        # Setup: Mock get_active_species to return None or empty
        self.config.reset_on_extinction = False
        self.config.pop_size = Mock()
        self.evolver.mock_pop_manager.get_active_species.return_value = None
        
        # Action: Call the method under test
        with self.assertRaises(CompleteExtinctionException):
            self.evolver.advance_population()
        
        # Assert: Check if extinction handling was called
        self.evolver.mock_reporters.complete_extinction.assert_called()
