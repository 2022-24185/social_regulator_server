from typing import Dict, List, Set, Tuple, Callable, TypeVar
from neuroevolution.evolution.species_set import MixedGenerationSpeciesSet

T = TypeVar('T')

def find_closest_element(
    target: T, 
    candidates: Set[T], 
    distance_fn: Callable[[T, T], float]
) -> T:
    """Returns the closest element in candidates to target."""
    return min(candidates, key=lambda candidate: distance_fn(target, candidate))

