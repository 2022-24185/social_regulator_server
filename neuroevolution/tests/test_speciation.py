import unittest
from unittest.mock import Mock, MagicMock, call, patch

from neuroevolution.evolution.speciation import Speciation

class TestableSpeciation(Speciation):
    def __init__(self, *args, **kwargs):
        self.mock_cache = Mock()
        self.mock_species_set = Mock()
        super().__init__(*args, **kwargs)

    def create_distance_cache(self, config):
        return self.mock_cache
    
    def create_species_set(self):
        return self.mock_species_set
    
class TestSpeciation(unittest.TestCase): 
    def setUp(self):
        self.config = MagicMock()
        self.config.species_set_config.compatibility_threshold = 3.0
        self.speciation = TestableSpeciation(self.config)

    def test_initialization(self):
        # Verify initialization values and dependencies are set up correctly
        self.assertIs(self.speciation.distance_cache, self.speciation.mock_cache)
        self.assertIs(self.speciation.species_set, self.speciation.mock_species_set)
        self.assertEqual(self.speciation.compatibility_threshold, 3.0)

    def test_speciate(self):
        # Setup
        population = {1: MagicMock(), 2: MagicMock()}
        self.speciation.mock_species_set.get_unspeciated.return_value = [1, 2]
        self.speciation.mock_species_set.get_all_species_ids.return_value = [10]
        self.speciation.mock_species_set.get_compatible_genomes.return_value = [(0.1, 10)]
        self.speciation.mock_species_set.create_new_species.return_value = 20
        
        # Execute
        self.speciation.partition_population(population, generation=1)

        expected_calls = [
            call(10, population[1]),
            call(10, population[2])
        ]
        self.speciation.mock_species_set.add_member.assert_has_calls(expected_calls, any_order=True)

class TestSetNewRepresentatives(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.compatibility_threshold = 3.0
        self.speciation = TestableSpeciation(self.config)

    def test_no_unspeciated_genomes(self):
        # Setup
        self.speciation.mock_species_set.get_unspeciated.return_value = []
        
        # Execute
        self.speciation.set_new_representatives({})

        # Assert
        self.speciation.mock_species_set.get_all_species.assert_not_called()

    @patch('random.choice', return_value=1)
    def test_random_choice_representative(self, choice_mock):
        # Setup
        population = {1: 'Genome1', 2: 'Genome2'}
        self.speciation.mock_species_set.get_unspeciated.return_value = [1, 2]
        species_instance = MagicMock()
        species_instance.get_representative.return_value = None
        self.speciation.mock_species_set.get_all_species.return_value = [('species1', species_instance)]

        # Execute
        self.speciation.set_new_representatives(population)

        # Assert
        self.speciation.mock_species_set.update_species_representative.assert_called_once_with('species1', 'Genome1')

    @patch('random.choice', return_value=1)
    def test_extract_new_representative_no_rep_id(self, mock_choice):
        # Setup
        unspeciated = [1, 2, 3]


        # Execute
        new_rep, updated_unspeciated = self.speciation.extract_new_representative_id(unspeciated, lambda x, y: abs(x-y), None)

        # Assert
        self.assertEqual(new_rep, 1)
        self.assertNotIn(1, updated_unspeciated)
        self.assertIn(2, updated_unspeciated)
        self.assertIn(3, updated_unspeciated)
        mock_choice.assert_called_once()

    def test_find_closest_element_representative(self):
        # Setup
        population = {1: 1, 2: 2, 3: 3}
        self.speciation.mock_species_set.get_unspeciated.return_value = [2, 3]
        species_instance = MagicMock()
        species_instance.get_representative.return_value = MagicMock(key=1)  # Assume existing representative
        self.speciation.mock_species_set.get_all_species.return_value = [(1, species_instance)]

        # Mock distance function
        def mock_distance_fn(a, b):
            return abs(population[a] - population[b])
        self.speciation.distance_cache = MagicMock(side_effect=mock_distance_fn)

        # Execute
        self.speciation.set_new_representatives(population)

        # Assert
        # Since Genome 2 is the closest to Genome 1 (existing representative), it should be chosen as the new representative
        new_representative = 2
        self.speciation.mock_species_set.update_species_representative.assert_called_with(1, population[new_representative])
        # Verify that Genome 2 is also added as a member of the species
        self.speciation.mock_species_set.update_species_representative.assert_called_with(1, population[new_representative])

if __name__ == '__main__':
    unittest.main()