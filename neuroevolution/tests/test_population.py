# /path/to/test_mixed_generation_population.py

import unittest
from unittest.mock import MagicMock, patch, create_autospec
from neat.config import Config
from neat.genome import DefaultGenome, DefaultGenomeConfig
from neat.reporting import ReporterSet
from neuroevolution.evolution.tournament_reproduction import MixedGenerationReproduction
from neuroevolution.evolution.mixed_generation_species import MixedGenerationSpeciesSet
from neuroevolution.evolution.tournament_stagnation import TournamentStagnation
from neuroevolution.evolution.mixed_generation_population import MixedGenerationPopulation

class TestMixedGenerationPopulation(unittest.TestCase):
    def setUp(self):
        # Mock the configuration and fitness function
        self.config = MagicMock(spec=Config)
        self.config.fitness_criterion = 'max'
        self.config.no_fitness_termination = False
        self.config.fitness_threshold = 1.0
        self.config.pop_size = 50
        self.config.reset_on_extinction = 'false'
        self.config.species_set_type = MagicMock(return_value = create_autospec(MixedGenerationSpeciesSet, instance=True))
        self.config.species_set_config = MagicMock()
        self.config.reproduction_type = MagicMock(return_value = create_autospec(MixedGenerationReproduction, instance=True))
        self.config.stagnation_type = MagicMock()
        self.config.stagnation_config = MagicMock()
        self.config.reproduction_config = MagicMock()
        self.config.genome_type = MagicMock(spec=DefaultGenome)
        self.config.genome_config = MagicMock(spec=DefaultGenomeConfig)
        self.evaluation_threshold = 10
        self.fitness_function = MagicMock()

        # Define a small initial population
        self.population = {1: DefaultGenome(key=1), 2: DefaultGenome(key=2)}
        self.species = MagicMock(spec=MixedGenerationSpeciesSet)
        self.initial_state = (self.population, self.species, 0)

    def test_initialization_with_initial_state(self):
        """Test initialization with a given initial state."""
        tp = MixedGenerationPopulation(self.config, self.fitness_function, self.evaluation_threshold, self.initial_state)

        self.assertEqual(tp.population, self.initial_state[0])
        self.assertEqual(tp.species, self.initial_state[1])
        self.assertEqual(tp.generation, self.initial_state[2])
        self.assertIsInstance(tp.reproduction, MixedGenerationReproduction)
        self.assertEqual(tp.fitness_summarizer, max)
    
    @patch('neuroevolution.evolution.tournament_reproduction.TournamentReproduction.create_new_genomes', return_value={1: DefaultGenome(key=1), 2: DefaultGenome(key=2)})
    def test_initialization_without_initial_state(self, mock_create_new_genomes):
        """Test initialization without a given initial state, ensuring defaults are set."""
        # Mock species_set and reproduction return values
        mock_species_instance = MagicMock(spec=MixedGenerationSpeciesSet)
        mock_reproduction_instance = MagicMock(spec=MixedGenerationReproduction)
        
        # Ensure create_new_genomes returns a proper dictionary
        mock_reproduction_instance.create_new_genomes.return_value = {1: DefaultGenome(key=1), 2: DefaultGenome(key=2)}

        self.config.species_set_type.return_value = mock_species_instance
        self.config.reproduction_type.return_value = mock_reproduction_instance

        tp = MixedGenerationPopulation(self.config, self.fitness_function, self.evaluation_threshold)

        # Verify the population and species are set up correctly
        self.assertIsInstance(tp.population, dict)
        self.assertTrue(callable(tp.reproduction.create_new_genomes))
        self.assertEqual(tp.generation, 0)
        self.assertEqual(tp.species, mock_species_instance)
        self.assertEqual(tp.fitness_summarizer, max)

        # Verify species set-up method calls
        tp.species.set_new_population.assert_called_once_with(tp.population)
        tp.species.speciate.assert_called_once_with(0, self.config)

    def test_receive_evaluation(self):
        """Test receiving an evaluation of a genome."""
        tp = MixedGenerationPopulation(self.config, self.fitness_function, self.evaluation_threshold, self.initial_state)
        user_data = MagicMock()
        user_data.genome_id = 1
        tp.receive_evaluation(user_data)
        self.assertIn(1, tp.evaluated_genomes)
        self.assertEqual(tp.evaluated_genomes[1].data, user_data)
    
    @patch('neuroevolution.evolution.mixed_generation_population.MixedGenerationPopulation.advance_population')
    def test_receive_evaluation_threshold(self, mock_advance_population):
        """Test receiving an evaluation that reaches the threshold."""
        tp = MixedGenerationPopulation(self.config, self.fitness_function, self.evaluation_threshold, self.initial_state)
        tp.evaluation_threshold = 1
        user_data = MagicMock()
        user_data.genome_id = 1
        tp.receive_evaluation(user_data)
        mock_advance_population.assert_called_once()

    def test_evaluate_fitness(self):
        """Test the fitness evaluation of the population."""
        tp = MixedGenerationPopulation(self.config, self.fitness_function, self.evaluation_threshold, self.initial_state)
        genome = MagicMock(spec=DefaultGenome)
        genome.fitness = 1.0
        tp.evaluated_genomes = {1: genome}
        best_genome = tp.evaluate_fitness(self.fitness_function)
        self.assertEqual(best_genome, genome)
        self.fitness_function.assert_called_once_with(tp.evaluated_genomes, self.config)
    
    def test_track_best_genome(self):
        """Test tracking the best genome."""
        tp = MixedGenerationPopulation(self.config, self.fitness_function, self.evaluation_threshold, self.initial_state)
        genome = MagicMock(spec=DefaultGenome)
        genome.fitness = 1.0
        tp.track_best_genome(genome)
        self.assertEqual(tp.best_genome, genome)
    
    def test_should_terminate(self):
        """Test termination check based on fitness."""
        genome = MagicMock(spec=DefaultGenome)
        genome.fitness_value = 1.1
        self.config.fitness_threshold = 1.0
        tp = MixedGenerationPopulation(self.config, self.fitness_function, self.evaluation_threshold, self.initial_state)
        self.assertTrue(tp.should_terminate(genome))
        genome.fitness_value = 0.9
        self.assertFalse(tp.should_terminate(genome))

    @patch('neuroevolution.evolution.mixed_generation_population.MixedGenerationPopulation._generate_offspring', return_value={3: DefaultGenome(key=3)})
    @patch('neuroevolution.evolution.mixed_generation_population.MixedGenerationPopulation._handle_possible_extinction')
    def test_reproduce_evaluated(self, mock_generate_offspring, mock_handle_possible_extinction):
        """Test the reproduction process after evaluation."""
        tp = MixedGenerationPopulation(self.config, self.fitness_function, self.evaluation_threshold, self.initial_state)
        
        # Mock the evaluated genomes to simulate the state before reproduction
        tp.evaluated_genomes = tp.population.copy()

        # Perform reproduction
        tp.reproduce_evaluated()
        mock_generate_offspring.assert_called_once()
        mock_handle_possible_extinction.assert_called_once()

        # Verify that the offspring were added to the population
        self.assertIn(3, tp.population)
        self.assertEqual(tp.population[3].key, 3)

        # Verify that the evaluated genomes were removed
        self.assertNotIn(1, tp.population)
        self.assertNotIn(2, tp.population)

        # Check that the available genomes list is updated correctly
        self.assertIn(3, tp.available_genomes)
        self.assertNotIn(1, tp.available_genomes)
        self.assertNotIn(2, tp.available_genomes)

if __name__ == '__main__':
    unittest.main()
