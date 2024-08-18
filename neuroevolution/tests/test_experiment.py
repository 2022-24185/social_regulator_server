import unittest
from unittest.mock import MagicMock
from neuroevolution.server.models import UserData
from neuroevolution.run_experiments.async_experiment import AsyncExperiment

class TestableSimulatedUserEvalExperiment(AsyncExperiment):
    def create_evolution(self):
        return MagicMock()

    def create_phenotype_creator(self):
        return MagicMock()

    def create_fitness_function(self):
        return MagicMock()


class TestSimulatedUserEvalExperiment(unittest.TestCase):
    def setUp(self):
        self.experiment = TestableSimulatedUserEvalExperiment('neuroevolution/evolution/config_cppn_social_brain')
        self.experiment.evolution.create_new_population = MagicMock()
        self.experiment.evolution.receive_evaluation = MagicMock()
        self.experiment.evolution.return_random_individual = MagicMock()
        self.experiment.phenotype_creator.create_network_from_genome = MagicMock()

    def test_start_with_num_generations(self):
        self.experiment.instanciate = MagicMock()
        self.experiment.run_simulation = MagicMock()
        self.experiment.evolution.get_current_generation = MagicMock(side_effect=[0, 1, 2, 3])

        self.experiment.start(num_generations=3)

        self.experiment.instanciate.assert_called_once()
        self.assertEqual(self.experiment.run_simulation.call_count, 3)

    def test_start_without_num_generations(self):
        self.experiment.instanciate = MagicMock()
        self.experiment.run_simulation = MagicMock()
        self.experiment.stop_experiment = True

        self.experiment.start()

        self.experiment.instanciate.assert_called_once()
        self.experiment.run_simulation.assert_not_called()

    def test_stop(self):
        self.experiment.evolution.terminate_evolution = MagicMock()

        self.experiment.stop()

        self.assertTrue(self.experiment.stop_experiment)
        self.experiment.evolution.terminate_evolution.assert_called_once()

    def test_reset(self):
        self.experiment.stop = MagicMock()
        self.experiment.instanciate = MagicMock()
        self.experiment.start = MagicMock()

        self.experiment.reset()

    def test_instanciate_creates_new_population_and_simulates_user_requests(self):
        self.experiment.instanciate()
        self.experiment.evolution.create_new_population.assert_called_once()
        self.assertEqual(len(self.experiment.evaluation_pool), 10)

    def test_run_simulation_simulates_user_request_and_evaluation(self):
        # Setup
        self.experiment.simulate_user_request = MagicMock()
        self.experiment.simulate_user_evaluation = MagicMock()
        self.experiment.receive_evaluation = MagicMock()

        # Action
        self.experiment.run_simulation()

        # Assert
        self.experiment.simulate_user_request.assert_called_once()
        self.experiment.simulate_user_evaluation.assert_called_once()
        self.experiment.receive_evaluation.assert_called_once()

    def test_receive_evaluation_processes_user_data(self):
        test_data = UserData(
            genome_id=1,
            time_since_startup=100,
            user_rating=5,
            last_message=None,
            last_message_time=None,
            last_response=None,
            last_response_time=None
        )
        self.experiment.receive_evaluation(test_data)
        self.experiment.evolution.receive_evaluation.assert_called_once_with(test_data)

    def test_get_random_individual_returns_tuple(self):
        # Setup
        random_individual_mock = MagicMock()
        random_individual_mock.key = 1
        self.experiment.evolution.return_random_individual.return_value = random_individual_mock

        # Action
        gid, network = self.experiment.get_random_individual()

        # Assert
        self.assertEqual(gid, 1)
        self.experiment.phenotype_creator.create_network_from_genome.assert_called_once_with(random_individual_mock)

    def test_simulate_user_evaluation_removes_individual_from_evaluation_list(self):
        # Setup
        self.experiment.evaluation_pool = [1, 2, 3]

        # Action
        user_data = self.experiment.simulate_user_evaluation()

        # Assert
        self.assertNotIn(user_data.genome_id, self.experiment.evaluation_pool)
        self.assertTrue(isinstance(user_data, UserData))

    def test_simulate_user_request_adds_individual_to_evaluation_list(self):
        # Setup
        self.experiment.get_random_individual = MagicMock(return_value=(1, None))
        self.experiment.evaluation_pool = []

        # Action
        self.experiment.simulate_user_request()

        # Assert
        self.assertIn(1, self.experiment.evaluation_pool)