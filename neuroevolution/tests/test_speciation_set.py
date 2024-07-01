import unittest
from unittest.mock import Mock, MagicMock
from itertools import count
from neuroevolution.evolution.species_set import MixedGenerationSpeciesSet
from neuroevolution.evolution.species import MixedGenerationSpecies

class TestableMixedGenerationSpeciesSet(MixedGenerationSpeciesSet):
    def __init__(self, *args, **kwargs):
        self.mock_species_iterator = count(1)
        super().__init__(*args, **kwargs)

    def create_new_species(self, generation):
        species_id = next(self.mock_species_iterator)
        mock_species = MagicMock(spec=MixedGenerationSpecies)
        self.species[species_id] = mock_species
        return species_id

class TestMixedGenerationSpeciesSet(unittest.TestCase):
    def setUp(self):
        self.species_set = TestableMixedGenerationSpeciesSet()

    def test_init(self):
        self.assertEqual(self.species_set.species, {})
        self.assertEqual(self.species_set.genome_to_species, {})

    def test_reset(self):
        self.species_set.species = {'dummy': 'data'}
        self.species_set.reset()
        self.assertEqual(list(self.species_set.species.keys()), [1])

    def test_create_new_species(self):
        mock_generation = Mock()
        new_species_id = self.species_set.create_new_species(mock_generation)
        self.assertIn(new_species_id, self.species_set.species)
        self.assertEqual(new_species_id, 1)

    def test_create_new_species_and_add_member(self):
        species_id = self.species_set.create_new_species(0)
        self.assertIn(species_id, self.species_set.species)
        mock_genome = MagicMock(key=Mock())
        self.species_set.add_member(species_id, mock_genome)
        self.species_set.species[species_id].add_member.assert_called_once_with(mock_genome)

    def test_add_member(self):
        species_id = self.species_set.create_new_species(0)
        mock_genome = MagicMock(key=42)
        self.species_set.add_member(species_id,mock_genome)
        self.assertEqual(self.species_set.genome_to_species[42], species_id)

    def test_species_id_increment(self):
        species_id1 = self.species_set.create_new_species(0)
        species_id2 = self.species_set.create_new_species(0)
        self.assertNotEqual(species_id1, species_id2)
        self.assertEqual(species_id2, species_id1 + 1)

    def test_set_and_get_representative(self):
        species_id = self.species_set.create_new_species(0)
        mock_genome = MagicMock()
        self.species_set.set_representative(species_id, (42, mock_genome))
        self.species_set.species[species_id].set_representative.assert_called_with((42, mock_genome))

    def test_get_species(self):
        species_id = self.species_set.create_new_species(0)
        mock_genome = MagicMock()
        self.species_set.set_representative(species_id, (42, mock_genome))
        self.assertEqual(self.species_set.get_species(species_id), self.species_set.species[species_id])

    def test_species_deletion_and_reset(self):
        species_id1 = self.species_set.create_new_species(0)
        species_id2 = self.species_set.create_new_species(0)
        self.species_set.remove_species(species_id1)
        self.assertNotIn(species_id1, self.species_set.species)
        self.assertIn(species_id2, self.species_set.species)

        # Reset and check if all species are removed
        self.species_set.reset()
        self.assertEqual(len(self.species_set.species), 1)

    def test_get_all_species_ids(self):
        species_id1 = self.species_set.create_new_species(0)
        species_id2 = self.species_set.create_new_species(0)
        self.assertEqual(self.species_set.get_all_species_ids(), [species_id1, species_id2])

    def test_get_all_species_objects(self):
        species_id1 = self.species_set.create_new_species(0)
        species_id2 = self.species_set.create_new_species(0)
        self.assertEqual(self.species_set.get_all_species_objects(), [self.species_set.species[species_id1], self.species_set.species[species_id2]])

    def test_get_all_species(self):
        species_id1 = self.species_set.create_new_species(0)
        species_id2 = self.species_set.create_new_species(0)
        self.assertEqual(self.species_set.get_all_species(), [(species_id1, self.species_set.species[species_id1]), (species_id2, self.species_set.species[species_id2])])

    def test_get_active_species(self):
        species_id1 = self.species_set.create_new_species(0)
        species_id2 = self.species_set.create_new_species(0)
        self.species_set.get_species(species_id1).is_active.return_value = False
        self.species_set.get_species(species_id2).is_active.return_value = True
        self.assertEqual(self.species_set.get_active_species(), [self.species_set.species[species_id2]])

    def test_get_unspeciated(self):
        # Setup: Mock a population and genome_to_species mapping
        species_id1 = self.species_set.create_new_species(0)
        species_id2 = self.species_set.create_new_species(0)
        self.species_set.genome_to_species = {
            1: species_id1,
            2: species_id2,
            3: species_id1,
            4: species_id2,
        }
        mock_population = {
            1: MagicMock(),
            2: MagicMock(),
            3: MagicMock(),
            4: MagicMock(),
            5: MagicMock(),
            6: MagicMock(),
            7: MagicMock(),
        }
        
        # Test: get_unspeciated() should return all genome ids that are not in a species
        self.assertEqual(self.species_set.get_unspeciated(mock_population), {5, 6, 7})

    def test_mark_stagnant(self):
        species_id = self.species_set.create_new_species(0)
        self.species_set.species[species_id] = MixedGenerationSpecies(0, 1)
        self.species_set.species[species_id].active = True
        self.species_set.mark_stagnant(species_id)
        self.assertFalse(self.species_set.species[species_id].is_active())

    def test_remove_stagnant_species(self):
        species_id1 = self.species_set.create_new_species(0)
        species_id2 = self.species_set.create_new_species(0)
        self.species_set.species[species_id1] = MixedGenerationSpecies(0, 1)
        self.species_set.species[species_id2] = MixedGenerationSpecies(1, 1)
        self.species_set.species[species_id1].active = False
        self.species_set.species[species_id2].active = True
        self.species_set.remove_stagnant_species()
        self.assertNotIn(species_id1, self.species_set.species)
        self.assertIn(species_id2, self.species_set.species)

    def test_get_representative_ids(self):
        species_id1 = self.species_set.create_new_species(0)
        species_id2 = self.species_set.create_new_species(0)
        self.species_set.species[species_id1].get_representative.return_value = 42
        self.species_set.species[species_id2].get_representative.return_value = 43
        reps = self.species_set.get_representatives()
        expected_reps = {species_id1: 42, species_id2: 43}
        self.assertEqual(reps, expected_reps)

    def test_get_species_id_for_genome(self):
        species_id1 = self.species_set.create_new_species(0)
        self.species_set.species[species_id1].is_member.return_value = True
        found_species_id = self.species_set.get_species_id_for_genome(42)
        self.assertEqual(found_species_id, species_id1)
        self.species_set.species[species_id1].is_member.assert_called_with(42)

    def test_remove_species(self):
        species_id1 = self.species_set.create_new_species(0)
        self.species_set.remove_species(species_id1)
        self.assertNotIn(species_id1, self.species_set.species)
        self.assertNotIn(species_id1, self.species_set.genome_to_species.values())

class TestGetCompatibleGenomes(unittest.TestCase):
    def setUp(self):
        self.species_set = MixedGenerationSpeciesSet()
        self.mock_population = {'genome1': MagicMock(), 'genome2': MagicMock()}
        self.mock_compatibility_fn = MagicMock()

    def test_empty_species_set(self):
        self.assertEqual(self.species_set.get_compatible_genomes('genome1', self.mock_population, self.mock_compatibility_fn), [])

    def test_no_compatible_genomes(self):
        self.species_set.species = {1: MagicMock(get_representative=MagicMock(return_value=MagicMock(key='genome1')))}
        self.mock_compatibility_fn.return_value = 0
        self.assertEqual(self.species_set.get_compatible_genomes('genome1', self.mock_population, self.mock_compatibility_fn), [])

    def test_with_compatible_genomes(self):
        self.species_set.species = {1: MagicMock(key=1, get_representative=MagicMock(return_value=MagicMock(key='genome1')))}
        self.mock_compatibility_fn.return_value = 0.5
        expected = [(0.5, 1)]
        self.assertEqual(self.species_set.get_compatible_genomes('genome1', self.mock_population, self.mock_compatibility_fn), expected)


    def test_no_representative_in_species(self):
        species_mock = MagicMock(key=1, get_representative=MagicMock(return_value=None), get_random_member=MagicMock(return_value=Mock(key='genome2')))
        self.species_set.species = {1: species_mock}
        self.mock_compatibility_fn.return_value = 0.7
        expected = [(0.7, 1)]
        self.assertEqual(self.species_set.get_compatible_genomes('genome1', self.mock_population, self.mock_compatibility_fn), expected)

if __name__ == '__main__':
    unittest.main()

