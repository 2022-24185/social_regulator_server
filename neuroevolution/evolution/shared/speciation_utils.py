from typing import Dict, List, Set, Tuple, Callable, TypeVar

T = TypeVar('T')

def find_closest_element(
    target: T, 
    candidates: Set[T], 
    distance_fn: Callable[[T, T], float]
) -> T:
    """Returns the closest element in candidates to target."""
    return min(candidates, key=lambda candidate: distance_fn(target, candidate))

def get_compatible_elements(
    groups: Dict[int, T], 
    element: T, 
    elements: Dict[int, T], 
    compatibility_fn: Callable[[T, T], bool]
) -> List[Tuple[float, int]]:
    """Returns a list of (compatibility, group_id) tuples for each group in groups."""
    return [(compatibility_fn(elements[group_id], element), group_id)
            for group_id, _ in groups.items()
            if compatibility_fn(elements[group_id], element)]

def partition_elements(
    ungrouped: Set[int], 
    elements: Dict[int, T], 
    groups: Dict[int, int], 
    memberships: Dict[int, List[int]], 
    compatibility_fn: Callable[[T, T], bool]
) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    """Partitions ungrouped elements into groups based on compatibility."""
    new_groups = groups.copy()
    new_memberships = {k: v[:] for k, v in memberships.items()}

    ungrouped_copy = ungrouped.copy()  # Create a copy to avoid modifying the original set

    for element_id in ungrouped_copy:
        element = elements[element_id]
        candidates = get_compatible_elements(new_groups, element, elements, compatibility_fn)
        
        print(f"Processing element {element_id} ({element}), candidates: {candidates}")
        
        if candidates:
            _, best_group_id = min(candidates, key=lambda x: x[0])
            print(f"Element {element_id} ({element}) is compatible with group {best_group_id}")
            new_memberships[best_group_id].append(element_id)
        else:
            new_group_id = max(new_groups.keys(), default=0) + 1
            print(f"Element {element_id} ({element}) is not compatible with any existing group. Creating new group {new_group_id}")
            new_groups[new_group_id] = element_id
            new_memberships[new_group_id] = [element_id]
    
    print(f"Final memberships: {new_memberships}")
    return new_groups, new_memberships
