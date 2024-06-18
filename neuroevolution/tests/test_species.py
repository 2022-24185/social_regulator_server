import unittest
from unittest.mock import MagicMock, patch
from neat.genome import DefaultGenome
from neuroevolution.evolution.tournament_species import (
    TournamentSpecies,
    GenomeDistanceCache,
    TournamentSpeciesSet,
    species_factory,
    distance_cache_factory,
)


class TestTournamentSpeciesSet(unittest.TestCase):
    def setUp(self):
        self.species_set = TournamentSpeciesSet(MagicMock(), MagicMock(), MagicMock(), distance_cache_factory)
        self.species_set.species = {1: MagicMock(), 2: MagicMock()}
        self.species_set.population = {1: MagicMock(spec=DefaultGenome), 2: MagicMock(spec=DefaultGenome)}
        self.species_set.genome_to_species = {1: 2}
        self.species_set.distance_cache = GenomeDistanceCache(MagicMock())

    def test_speciate(self):
        self.species_set._find_new_representatives = MagicMock(return_value=({}, {}))
        self.species_set._partition_population = MagicMock()
        self.species_set._update_species_collection = MagicMock()
        self.species_set._log_species_statistics = MagicMock()

        self.species_set.speciate(1, self.species_set.species_set_config)

        self.species_set._find_new_representatives.assert_called_once()
        self.species_set._partition_population.assert_called_once()
        self.species_set._update_species_collection.assert_called_once()
        self.species_set._log_species_statistics.assert_called_once()

    def test_initialize_parameters(self):
        self.species_set.species_set_config.compatibility_threshold = 3.0
        self.species_set.initialize_parameters(self.species_set.species_set_config)
        self.assertEqual(self.species_set.compatibility_threshold, 3.0)
        self.assertIsInstance(self.species_set.distance_cache, GenomeDistanceCache)

    def test_representatives_and_members_consistency(self):
        self.species_set._find_new_representatives = MagicMock(return_value=({1: 1, 2: 2}, {1: [1], 2: [2]}))
        self.species_set._partition_population = MagicMock()
        self.species_set._update_species_collection = MagicMock()
        self.species_set._log_species_statistics = MagicMock()

        self.species_set.speciate(1, self.species_set.species_set_config)

        new_representatives, new_members = self.species_set._find_new_representatives()
        self.assertEqual(len(new_representatives), len(new_members), "Number of representatives and members should be equal after _find_new_representatives")

        self.species_set._partition_population(new_representatives, new_members)
        self.assertEqual(len(new_representatives), len(new_members), "Number of representatives and members should be equal after _partition_population")

    def test_set_new_population(self):
        population = {1: MagicMock(spec=DefaultGenome), 2: MagicMock(spec=DefaultGenome)}
        self.species_set.set_new_population(population)
        self.assertEqual(self.species_set.population, population)
        self.assertEqual(self.species_set.unspeciated, set(population.keys()))

    @patch.object(GenomeDistanceCache, '__call__', return_value=2.0)
    def test_partition_population(self, mock_distance_cache):
        self.species_set.distance_cache = GenomeDistanceCache(MagicMock())
        self.species_set.species = {1: TournamentSpecies(1, 0), 2: TournamentSpecies(2, 0)}
        self.species_set.unspeciated = {4, 5}
        self.species_set.population = {1: MagicMock(), 2: MagicMock(), 3: MagicMock(), 4: MagicMock(), 5: MagicMock()}
        self.species_set.compatibility_threshold = 3.0

        new_representatives = {1: 2, 2: 3}
        new_members = {1: [2], 2: [3]}
        self.species_set._partition_population(new_representatives, new_members)

        self.assertEqual(new_members, {1: [2, 4, 5], 2: [3]})

    def test_update_species_collection(self):
        self.species_set.population = {1: MagicMock(spec=DefaultGenome), 2: MagicMock(spec=DefaultGenome)}
        self.species_set.species = {1: MagicMock(), 2: MagicMock()}
        representatives = {1: 1, 2: 2}
        species_members = {1: [1], 2: [2]}
        self.species_set._update_species_collection(representatives, species_members, 1)
        self.assertEqual(len(self.species_set.species), 2)

    def test_get_species_id(self):
        self.species_set.genome_to_species = {1: 2}
        self.assertEqual(self.species_set.get_species_id(1), 2)

    def test_get_species(self):
        species = MagicMock()
        self.species_set.species = {2: species}
        self.species_set.genome_to_species = {1: 2}
        self.assertEqual(self.species_set.get_species(1), species)

    # CORNER CASES

    def test_get_species_id_nonexistent(self):
        self.species_set.genome_to_species = {}
        with self.assertRaises(KeyError):
            self.species_set.get_species_id(1)

    def test_get_species_nonexistent(self):
        self.species_set.species = {}
        self.species_set.genome_to_species = {1: 2}
        with self.assertRaises(KeyError):
            self.species_set.get_species(1)

    def test_partition_population_empty(self):
        self.species_set.species = {1: TournamentSpecies(1, 0), 2: TournamentSpecies(2, 0)}
        self.species_set.unspeciated = set()
        self.species_set.population = {}
        self.species_set.compatibility_threshold = 3.0

        new_representatives = {1: 2, 2: 3}
        new_members = {1: [2], 2: [3]}
        self.species_set._partition_population(new_representatives, new_members)

        self.assertEqual(new_members, {1: [2], 2: [3]})


class TestTournamentSpecies(unittest.TestCase):
    def setUp(self):
        self.tournament_species = TournamentSpecies(1, 0)

    def test_update(self):
        representative = MagicMock(spec=DefaultGenome)
        members = {1: MagicMock(spec=DefaultGenome), 2: MagicMock(spec=DefaultGenome)}
        self.tournament_species.update(representative, members)
        self.assertEqual(self.tournament_species.representative, representative)
        self.assertEqual(self.tournament_species.members, members)

    def test_get_fitnesses(self):
        member1 = MagicMock(spec=DefaultGenome)
        member1.fitness = 1.0
        member2 = MagicMock(spec=DefaultGenome)
        member2.fitness = 2.0
        self.tournament_species.members = {1: member1, 2: member2}
        self.assertEqual(self.tournament_species.get_fitnesses(), [1.0, 2.0])

    def test_get_subset_of_fitnesses(self):
        member1 = MagicMock(spec=DefaultGenome)
        member1.fitness = 1.0
        member2 = MagicMock(spec=DefaultGenome)
        member2.fitness = 2.0
        self.tournament_species.members = {1: member1, 2: member2}
        self.assertEqual(self.tournament_species.get_subset_of_fitnesses([1]), [1.0])

    def test_get_fitnesses_empty(self):
        self.tournament_species.members = {}
        self.assertEqual(self.tournament_species.get_fitnesses(), [])

    def test_get_subset_of_fitnesses_empty(self):
        self.tournament_species.members = {}
        self.assertEqual(self.tournament_species.get_subset_of_fitnesses([1]), [])

    def test_get_subset_of_fitnesses_nonexistent_member(self):
        member1 = MagicMock(spec=DefaultGenome)
        member1.fitness = 1.0
        self.tournament_species.members = {1: member1}
        self.assertEqual(self.tournament_species.get_subset_of_fitnesses([2]), [])


class TestGenomeDistanceCache(unittest.TestCase):
    def setUp(self):
        self.genome_distance_cache = GenomeDistanceCache(MagicMock())

    def test_call(self):
        genome0 = MagicMock(spec=DefaultGenome)
        genome0.key = 1
        genome0.distance = MagicMock(return_value=2.0)
        genome1 = MagicMock(spec=DefaultGenome)
        genome1.key = 2
        self.assertEqual(self.genome_distance_cache(genome0, genome1), 2.0)
        self.assertEqual(self.genome_distance_cache.misses, 1)
        self.assertEqual(self.genome_distance_cache(genome0, genome1), 2.0)
        self.assertEqual(self.genome_distance_cache.hits, 1)


if __name__ == "__main__":
    unittest.main()