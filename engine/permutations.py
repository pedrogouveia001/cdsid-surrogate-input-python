import itertools
import numpy as np

def gerar_cases(numcrit: int) -> np.ndarray:
    """
    Generates all permutations for the criteria, sorted lexicographically,
    plus an extra dummy row for the equal weights case.
    Equivalent to the Delphi `GerarCases` procedure.
    
    Args:
        numcrit: Number of criteria.
        
    Returns:
        np.ndarray of shape (numcrit! + 1, numcrit) containing the permutations and dummy row.
    """
    # Create the initial array [1, 2, ..., numcrit]
    base_array = list(range(1, numcrit + 1))
    
    # itertools.permutations generates permutations in lexicographic order
    # if the input iterable is sorted.
    perms = list(itertools.permutations(base_array))
    
    # Append extra dummy row for equal weights case (all zeros)
    perms.append(tuple([0] * numcrit))
    
    return np.array(perms, dtype=int)

