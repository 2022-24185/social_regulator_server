import unittest
from neuroevolution.evolution.shared.speciation_utils import find_closest_element

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

if __name__ == '__main__':
    unittest.main()