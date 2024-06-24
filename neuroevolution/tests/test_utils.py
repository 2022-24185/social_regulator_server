import unittest
from neuroevolution.evolution.shared.speciation_utils import find_closest_element, get_compatible_elements, partition_elements

class TestFindClosestElement(unittest.TestCase):

    def test_normal_case(self):
        target = 5
        candidates = {1, 2, 3, 8, 9}
        distance_fn = lambda a, b: abs(a - b)
        result = find_closest_element(target, candidates, distance_fn)
        self.assertEqual(result, 3)

    def test_single_candidate(self):
        target = 5
        candidates = {10}
        distance_fn = lambda a, b: abs(a - b)
        result = find_closest_element(target, candidates, distance_fn)
        self.assertEqual(result, 10)

    def test_empty_candidates(self):
        target = 5
        candidates = set()
        distance_fn = lambda a, b: abs(a - b)
        with self.assertRaises(ValueError):
            find_closest_element(target, candidates, distance_fn)

    def test_multiple_closest_candidates(self):
        target = 5
        candidates = {4, 6}
        distance_fn = lambda a, b: abs(a - b)
        result = find_closest_element(target, candidates, distance_fn)
        self.assertIn(result, {4, 6})


class TestGetCompatibleElements(unittest.TestCase):

    def test_normal_case(self):
        groups = {1: 'a', 2: 'b'}
        element = 'c'
        elements = {1: 'a', 2: 'b', 3: 'c'}
        compatibility_fn = lambda x, y: len(x) == len(y)
        result = get_compatible_elements(groups, element, elements, compatibility_fn)
        self.assertEqual(result, [(True, 1), (True, 2)])

    def test_no_compatible_elements(self):
        groups = {1: 'aa', 2: 'bb'}
        element = 'c'
        elements = {1: 'aa', 2: 'bb', 3: 'c'}
        compatibility_fn = lambda x, y: len(x) == len(y)
        result = get_compatible_elements(groups, element, elements, compatibility_fn)
        self.assertEqual(result, [])

    def test_empty_groups(self):
        groups = {}
        element = 'c'
        elements = {1: 'aa', 2: 'bb', 3: 'c'}
        compatibility_fn = lambda x, y: len(x) == len(y)
        result = get_compatible_elements(groups, element, elements, compatibility_fn)
        self.assertEqual(result, [])

    def test_multiple_compatible_elements(self):
        groups = {1: 'a', 2: 'c'}
        element = 'c'
        elements = {1: 'a', 2: 'c', 3: 'c'}
        compatibility_fn = lambda x, y: len(x) == len(y)
        result = get_compatible_elements(groups, element, elements, compatibility_fn)
        self.assertEqual(result, [(True, 1), (True, 2)])


class TestPartitionElements(unittest.TestCase):

    def test_normal_case(self):
        ungrouped = {3}
        elements = {1: 'a', 2: 'b', 3: 'c'}
        groups = {1: 1, 2: 2}
        memberships = {1: [1], 2: [2]}
        compatibility_fn = lambda x, y: len(x) == len(y)
        
        # Perform partitioning
        original_groups = set(groups.keys())
        new_groups, result = partition_elements(ungrouped, elements, groups, memberships, compatibility_fn)
        new_group_ids = set(new_groups.keys()).difference(original_groups)

        self.assertEqual(len(new_group_ids), 0, "Expected no new groups to be created")
        self.assertIn(3, result[1] + result[2], "Expected element 3 to be in one of the existing groups")
        expected_memberships = {1: [1, 3], 2: [2]}
        self.assertEqual(result, expected_memberships)

    def test_no_ungrouped_elements(self):
        ungrouped = set()
        elements = {1: 'a', 2: 'b', 3: 'c'}
        groups = {1: 1, 2: 2}
        memberships = {1: [1], 2: [2]}
        compatibility_fn = lambda x, y: len(x) == len(y)
        new_groups, result = partition_elements(ungrouped, elements, groups, memberships, compatibility_fn)
        self.assertEqual(result, memberships)

    def test_all_elements_compatible(self):
        ungrouped = {3, 4}
        elements = {1: 'a', 2: 'b', 3: 'c', 4: 'd'}
        groups = {1: 1, 2: 2}
        memberships = {1: [1], 2: [2]}
        compatibility_fn = lambda x, y: True  # All elements are compatible
        new_groups, result = partition_elements(ungrouped, elements, groups, memberships, compatibility_fn)
        self.assertEqual(len(result), 2)  # No new groups created

    def test_no_compatible_elements(self):
        ungrouped = {3, 4}
        elements = {1: 'a', 2: 'b', 3: 'c', 4: 'd'}
        groups = {1: 1, 2: 2}
        memberships = {1: [1], 2: [2]}
        compatibility_fn = lambda x, y: False  # No elements are compatible
        new_groups, result = partition_elements(ungrouped, elements, groups, memberships, compatibility_fn)
        new_group_ids = set(new_groups.keys()).difference(groups.keys())
        self.assertEqual(len(new_group_ids), 2)

    def test_multiple_elements(self):
        ungrouped = {3, 4, 5}
        elements = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e'}
        groups = {1: 1, 2: 2}
        memberships = {1: [1], 2: [2]}
        compatibility_fn = lambda x, y: len(x) == len(y)
        new_groups, result = partition_elements(ungrouped, elements, groups, memberships, compatibility_fn)
        new_group_ids = set(new_groups.keys()).difference(groups.keys())
        for new_group_id in new_group_ids:
            for member in result[new_group_id]:
                self.assertIn(member, {3, 4, 5})

if __name__ == "__main__":
    unittest.main()

