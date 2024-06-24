import unittest
from unittest.mock import MagicMock, Mock, patch
from neat.genome import DefaultGenome
from neuroevolution.evolution.mixed_generation_species import (
    MixedGenerationSpecies,
    GenomeDistanceCache,
    MixedGenerationSpeciesSet,
    species_factory,
    distance_cache_factory,
)
from neat.config import Config


class TestMixedGenerationSpeciesSet(unittest.TestCase):
    def setUp(self):
        self.species_set = MixedGenerationSpeciesSet(None, None, None, None)
        self.species_set.species = {i: MagicMock() for i in range(1, 4)}
        self.species_set.population = {i: MagicMock() for i in range(1, 10)}
        self.species_set.unspeciated = set(range(1, 10))
        self.species_set.distance_cache = MagicMock()
        self.species_set.compatibility_threshold = 1.0
        self.species_set.distance_cache = GenomeDistanceCache(config=MagicMock())
        self.species_set.distance_cache.__call__ = Mock(return_value=0.1)

    def test_initialize_with_empty_population(self):
        self.species_set.set_new_population({})
        self.assertEqual(len(self.species_set.population), 0)
        self.assertEqual(len(self.species_set.unspeciated), 0)
        self.assertEqual(len(self.species_set.species), 0)
        self.assertEqual(len(self.species_set.genome_to_species), 0)

    def test_set_new_population(self):
        new_population = {i: MagicMock(spec=DefaultGenome) for i in range(10, 20)}
        self.species_set.set_new_population(new_population)
        self.assertEqual(self.species_set.population, new_population)
        self.assertEqual(len(self.species_set.unspeciated), len(new_population))
        self.assertEqual(len(self.species_set.species), 0)
        self.assertEqual(len(self.species_set.genome_to_species), 0)

    def test_batch_update_population(self):
        new_population = {10: MagicMock(spec=DefaultGenome), 11: MagicMock(spec=DefaultGenome)}
        self.species_set.batch_update_population(new_population)
        self.assertIn(10, self.species_set.population)
        self.assertIn(11, self.species_set.population)

    def test_batch_update_with_existing_population(self):
        new_population = {1: MagicMock(spec=DefaultGenome), 2: MagicMock(spec=DefaultGenome)}
        self.species_set.batch_update_population(new_population)
        self.assertEqual(self.species_set.population[1], new_population[1])
        self.assertEqual(self.species_set.population[2], new_population[2])

    def test_remove_dead_genomes(self):
        genome1 = Mock(spec=DefaultGenome)
        genome1.key = 1
        genome2 = Mock(spec=DefaultGenome)
        genome2.key = 2

        self.species_set.set_new_population({1: genome1, 2: genome2})
        self.species_set.remove_dead_genomes({1})

        self.assertEqual(len(self.species_set.population), 1)
        self.assertIn(2, self.species_set.population)
        self.assertNotIn(1, self.species_set.population)
        
    @patch('neuroevolution.evolution.mixed_generation_species.MixedGenerationSpeciesSet.find_new_representatives', return_value=({}, {}))
    @patch('neuroevolution.evolution.mixed_generation_species.MixedGenerationSpeciesSet.partition_population', return_value=(set(), {}))
    @patch('neuroevolution.evolution.mixed_generation_species.MixedGenerationSpeciesSet.update_species_collection')
    @patch('neuroevolution.evolution.mixed_generation_species.MixedGenerationSpeciesSet.log_species_statistics')
    def test_speciate(self, mock_log, mock_update, mock_partition, mock_find_new_representatives):
        generation = 1
        config = MagicMock(spec=Config)
        self.species_set.speciate(generation, config)
        mock_find_new_representatives.assert_called_once()
        mock_partition.assert_called_once()
        mock_update.assert_called_once()
        mock_log.assert_called_once()

    @patch('neuroevolution.evolution.mixed_generation_species.MixedGenerationSpeciesSet.find_closest_unspeciated_genome', return_value=1)
    def test_find_new_representatives(self, mock_find_closest):
        self.species_set.unspeciated = {1, 2, 3}
        self.species_set.species = {1: MagicMock(representative=MagicMock(key=1))}
        self.species_set.population = {1: MagicMock(DefaultGenome), 2: MagicMock(DefaultGenome), 3: MagicMock(DefaultGenome)}
        new_representatives, new_members = self.species_set.find_new_representatives(self.species_set.species, self.species_set.unspeciated, self.species_set.population, self.species_set.distance_cache)
        self.assertIn(1, new_representatives)
        self.assertIn(1, new_members[1])

    @patch('neuroevolution.evolution.mixed_generation_species.MixedGenerationSpeciesSet.find_closest_unspeciated_genome', return_value=None)
    def test_find_new_representatives_no_unspeciated(self, mock_find_closest):
        self.species_set.unspeciated = set()
        new_representatives, new_members = self.species_set.find_new_representatives(self.species_set.species, self.species_set.unspeciated, self.species_set.population, self.species_set.distance_cache)
        self.assertEqual(new_representatives, {})
        self.assertEqual(new_members, {})

    @patch('neuroevolution.evolution.mixed_generation_species.MixedGenerationSpeciesSet.find_closest_unspeciated_genome', return_value=1)
    @patch('neuroevolution.evolution.mixed_generation_species.GenomeDistanceCache.__call__', return_value=0.1)
    def test_direct_distance_call(self, mock_distance, mock_find_closest):
        # Attempt to directly use the mock to confirm its behavior
        # Explicitly creating a new instance and applying the mock directly to this instance
        distance_cache = GenomeDistanceCache(config=MagicMock())
        distance_cache.__call__ = Mock(return_value=0.1)
        # Using the newly created instance with directly set mock
        distance = distance_cache(1, 2)
        print("Direct distance call result:", distance)
        self.assertIsInstance(distance, float, "The mocked distance should be a float.")

    @patch('neuroevolution.evolution.mixed_generation_species.MixedGenerationSpeciesSet.find_closest_unspeciated_genome')
    @patch('neuroevolution.evolution.mixed_generation_species.GenomeDistanceCache.__call__')
    def test_partition_population(self, mock_distance, mock_find_closest):
        mock_distance.side_effect = lambda a, b: 0.1
        mock_find_closest.side_effect = lambda x: x
        population = {
            1: MagicMock(spec=DefaultGenome, key=1),
            2: MagicMock(spec=DefaultGenome, key=2),
            3: MagicMock(spec=DefaultGenome, key=3),
            4: MagicMock(spec=DefaultGenome, key=4)
        }
        unspeciated = {2, 4}
        new_representatives = {1: 1, 2: 3}
        new_members = {1: [1], 2: [3]}
        new_groups, new_memberships = self.species_set.partition_population(unspeciated, population, new_representatives, new_members, self.species_set.distance_cache, self.species_set.compatibility_threshold)
        self.assertIn(1, new_memberships[1])
        self.assertIn(3, new_memberships[2])
        self.assertIn(2, new_memberships[1])
        self.assertNotIn(4, new_memberships[2])
        self.assertNotIn(3, new_groups)


    # CORNER CASES


class TestTournamentSpecies(unittest.TestCase):

    def setUp(self):
        self.species = MixedGenerationSpecies(key=1, generation=0)
        self.mock_genome1 = Mock(fitness=1.0)
        self.mock_genome2 = Mock(fitness=2.0)
        self.mock_genome3 = Mock(fitness=3.0)
        self.species.update(representative=self.mock_genome1, members={1: self.mock_genome1, 2: self.mock_genome2})

    def test_update(self):
        new_representative = self.mock_genome2
        new_members = {2: self.mock_genome2, 3: self.mock_genome3}
        self.species.update(new_representative, new_members)
        self.assertEqual(self.species.representative, new_representative)
        self.assertEqual(self.species.members, new_members)

    def test_get_fitnesses(self):
        fitnesses = self.species.get_fitnesses()
        self.assertEqual(fitnesses, [1.0, 2.0])

    def test_get_subset_of_fitnesses(self):
        subset_fitnesses = self.species.get_subset_of_fitnesses([1])
        self.assertEqual(subset_fitnesses, [1.0])

    def test_remove_members(self):
        self.species.remove_members({1})
        self.assertNotIn(1, self.species.members)
        self.assertIn(2, self.species.members)

    ## EDGE CASES

    def test_update_empty_members(self):
        new_representative = self.mock_genome2
        new_members = {}
        self.species.update(new_representative, new_members)
        self.assertEqual(self.species.representative, new_representative)
        self.assertEqual(self.species.members, new_members)

    def test_get_fitnesses_empty_members(self):
        self.species.update(representative=self.mock_genome1, members={})
        fitnesses = self.species.get_fitnesses()
        self.assertEqual(fitnesses, [])

    def test_get_subset_of_fitnesses_non_existent_ids(self):
        subset_fitnesses = self.species.get_subset_of_fitnesses([3, 4])
        self.assertEqual(subset_fitnesses, [])

    def test_remove_members_empty_set(self):
        self.species.remove_members(set())
        self.assertEqual(self.species.members, {1: self.mock_genome1, 2: self.mock_genome2})

    def test_remove_members_non_existent(self):
        self.species.remove_members({3})
        self.assertEqual(self.species.members, {1: self.mock_genome1, 2: self.mock_genome2})

    def test_remove_all_members(self):
        self.species.remove_members({1, 2})
        self.assertEqual(self.species.members, {})

    def test_update_with_partial_data(self):
        new_representative = None
        new_members = {2: self.mock_genome2}
        self.species.update(new_representative, new_members)
        self.assertEqual(self.species.representative, new_representative)
        self.assertEqual(self.species.members, new_members)


class TestGenomeDistanceCache(unittest.TestCase):

    def setUp(self):
        # Mock configuration for the GenomeDistanceCache
        self.mock_config = Mock()

        # Initialize the distance cache with the mock configuration
        self.distance_cache = GenomeDistanceCache(config=self.mock_config)

        # Create mock genomes
        self.genome1 = Mock(spec=DefaultGenome)
        self.genome2 = Mock(spec=DefaultGenome)
        self.genome3 = Mock(spec=DefaultGenome)

        # Assign keys to mock genomes
        self.genome1.key = 1
        self.genome2.key = 2
        self.genome3.key = 3

    def test_distance_calculation_and_caching(self):
        # Mock the distance calculation between genome1 and genome2
        self.genome1.distance.return_value = 5.0
        distance = self.distance_cache(self.genome1, self.genome2)

        self.genome1.distance.assert_called_once_with(self.genome2, self.mock_config)
        self.assertEqual(distance, 5.0)
        self.assertEqual(self.distance_cache.misses, 1)
        self.assertEqual(self.distance_cache.hits, 0)
        self.assertIn((1, 2), self.distance_cache.distances)
        self.assertIn((2, 1), self.distance_cache.distances)

    def test_distance_retrieval_from_cache(self):
        self.genome1.distance.return_value = 5.0
        self.distance_cache(self.genome1, self.genome2)
        self.genome1.distance.reset_mock()
        distance = self.distance_cache(self.genome1, self.genome2)

        self.genome1.distance.assert_not_called()
        self.assertEqual(distance, 5.0)
        self.assertEqual(self.distance_cache.misses, 1)
        self.assertEqual(self.distance_cache.hits, 1)

    def test_distance_different_genomes(self):
        self.genome1.distance.return_value = 5.0
        self.genome2.distance.return_value = 6.0

        distance1 = self.distance_cache(self.genome1, self.genome3)
        distance2 = self.distance_cache(self.genome2, self.genome3)

        self.genome1.distance.assert_called_once_with(self.genome3, self.mock_config)
        self.genome2.distance.assert_called_once_with(self.genome3, self.mock_config)
        self.assertEqual(distance1, 5.0)
        self.assertEqual(distance2, 6.0)
        self.assertEqual(self.distance_cache.misses, 2)
        self.assertEqual(self.distance_cache.hits, 0)

    def test_distance_nonexistent_cache(self):
        non_existent_genome = Mock(spec=DefaultGenome)
        non_existent_genome.key = 4
        self.genome1.distance.return_value = 7.0

        distance = self.distance_cache(self.genome1, non_existent_genome)
        self.genome1.distance.assert_called_once_with(non_existent_genome, self.mock_config)
        self.assertEqual(distance, 7.0)
        self.assertEqual(self.distance_cache.misses, 1)
        self.assertEqual(self.distance_cache.hits, 0)
        self.assertIn((1, 4), self.distance_cache.distances)
        self.assertIn((4, 1), self.distance_cache.distances)

    def test_symmetric_distances(self):
        self.genome1.distance.return_value = 5.0

        distance1 = self.distance_cache(self.genome1, self.genome2)
        distance2 = self.distance_cache(self.genome2, self.genome1)

        self.genome1.distance.assert_called_once_with(self.genome2, self.mock_config)
        self.assertEqual(distance1, 5.0)
        self.assertEqual(distance2, 5.0)
        self.assertEqual(self.distance_cache.misses, 1)
        self.assertEqual(self.distance_cache.hits, 1)

    def test_complex_genome_distances(self):
        complex_genome1 = Mock(spec=DefaultGenome)
        complex_genome2 = Mock(spec=DefaultGenome)
        complex_genome1.key = 5
        complex_genome2.key = 6
        complex_genome1.distance.return_value = 8.0

        distance = self.distance_cache(complex_genome1, complex_genome2)
        complex_genome1.distance.assert_called_once_with(complex_genome2, self.mock_config)
        self.assertEqual(distance, 8.0)
        self.assertEqual(self.distance_cache.misses, 1)
        self.assertEqual(self.distance_cache.hits, 0)

    ## EDGE CASES

    def test_empty_genomes(self):
        # Create mock empty genomes (no nodes or connections)
        self.genome1.nodes = {}
        self.genome2.nodes = {}
        self.genome1.connections = {}
        self.genome2.connections = {}
        
        # Mock the distance calculation to return 0.0 for empty genomes
        self.genome1.distance.return_value = 0.0

        distance = self.distance_cache(self.genome1, self.genome2)
        self.genome1.distance.assert_called_once_with(self.genome2, self.mock_config)
        self.assertEqual(distance, 0.0)
        self.assertEqual(self.distance_cache.misses, 1)
        self.assertEqual(self.distance_cache.hits, 0)

    def test_repeated_calls(self):
        self.genome1.distance.return_value = 5.0

        distance1 = self.distance_cache(self.genome1, self.genome2)
        distance2 = self.distance_cache(self.genome1, self.genome2)
        distance3 = self.distance_cache(self.genome1, self.genome2)

        self.genome1.distance.assert_called_once_with(self.genome2, self.mock_config)
        self.assertEqual(distance1, 5.0)
        self.assertEqual(distance2, 5.0)
        self.assertEqual(distance3, 5.0)
        self.assertEqual(self.distance_cache.misses, 1)
        self.assertEqual(self.distance_cache.hits, 2)

    def test_self_distance(self):
        self.genome1.distance.return_value = 0.0

        distance = self.distance_cache(self.genome1, self.genome1)
        self.genome1.distance.assert_called_once_with(self.genome1, self.mock_config)
        self.assertEqual(distance, 0.0)
        self.assertEqual(self.distance_cache.misses, 1)
        self.assertEqual(self.distance_cache.hits, 0)
        self.assertIn((1, 1), self.distance_cache.distances)

    def test_large_number_of_genomes(self):
        num_genomes = 1000
        genomes = [Mock(spec=DefaultGenome) for i in range(num_genomes)]
        for i, genome in enumerate(genomes):
            genome.key = i
            genome.distance.return_value = i * 0.1
        
        for i in range(num_genomes):
            for j in range(i+1, num_genomes):
                distance = self.distance_cache(genomes[i], genomes[j])
                expected_distance = i * 0.1
                self.assertEqual(distance, expected_distance)

        self.assertEqual(self.distance_cache.misses, num_genomes * (num_genomes - 1) // 2)
        self.assertEqual(self.distance_cache.hits, 0)

    def test_invalid_inputs(self):
        with self.assertRaises(AttributeError):
            self.distance_cache(None, self.genome1)

        with self.assertRaises(AttributeError):
            self.distance_cache(self.genome1, None)

        with self.assertRaises(AttributeError):
            self.distance_cache(None, None)


if __name__ == "__main__":
    unittest.main()