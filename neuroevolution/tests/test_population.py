# /path/to/test_tournament_population.py

import unittest
from unittest.mock import Mock, patch, create_autospec
from neat.config import Config
from neat.genome import DefaultGenome
from neat.reporting import ReporterSet
from neuroevolution.evolution.tournament_reproduction import TournamentReproduction
from neuroevolution.evolution.tournament_species import TournamentSpeciesSet
from neuroevolution.evolution.tournament_stagnation import TournamentStagnation
from neuroevolution.evolution.tournament_population import TournamentPopulation

class TestTournamentPopulation(unittest.TestCase):
    def setUp(self):
        # Set up the config mock
        self.config = Mock(spec=Config)
        self.config.pop_size = 10
        self.config.genome_type = DefaultGenome
        self.config.genome_config = Mock()
        self.config.species_set_type = create_autospec(TournamentSpeciesSet)
        self.config.species_set_config = Mock()
        self.config.reproduction_type = create_autospec(TournamentReproduction)
        self.config.reproduction_config = Mock()
        self.config.stagnation_type = create_autospec(TournamentStagnation)
        self.config.stagnation_config = Mock()
        self.config.fitness_criterion = "max"
        self.config.fitness_threshold = 100.0
        self.config.no_fitness_termination = False

        # Create a mock for the species set with a species attribute
        self.mock_species_set = self.config.species_set_type.return_value
        self.mock_species_set.species = {}
        self.mock_species_set.set_new_population = Mock()
        self.mock_species_set.speciate = Mock()

        # Create a mock fitness function
        self.fitness_function = Mock()

        # Mock population
        self.population = {
            i: DefaultGenome(i) for i in range(self.config.pop_size)
        }

    @patch('neat.reporting.ReporterSet')
    @patch('neuroevolution.evolution.tournament_reproduction.TournamentReproduction.create_new_genomes')
    def test_initial_state(self, mock_create_new_genomes, mock_reporter_set):
        mock_create_new_genomes.return_value = self.population

        tp = TournamentPopulation(
            config=self.config, 
            fitness_function=self.fitness_function, 
            reporter_set=mock_reporter_set, 
            reproduction=mock_create_new_genomes, 
            species_set=self.mock_species_set
        )

        self.assertEqual(tp.generation, 0)
        self.assertEqual(len(tp.population), self.config.pop_size)
        self.assertEqual(tp.best_genome, None)
        self.assertTrue(self.mock_species_set.set_new_population.called)
        self.assertTrue(self.mock_species_set.speciate.called)

if __name__ == '__main__':
    unittest.main()
