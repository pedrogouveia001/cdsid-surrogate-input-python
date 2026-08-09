# pyrefly: ignore [missing-import]
import numpy as np
from .surrogate import extract_unique_solutions
from .stats import compute_statistics

def promethee_results(matriz_poa_promethee: np.ndarray, matriz_dif_vg_promethee: np.ndarray):
    """
    Port of the aggregation part of the `Promethee` Delphi procedure.
    Groups PROMETHEE II solutions and computes statistics.
    
    Args:
        matriz_poa_promethee: np.ndarray of shape (num_cases, num_alt)
        matriz_dif_vg_promethee: np.ndarray of shape (num_cases, num_alt)
        
    Returns:
        Dictionary with unique PROMETHEE solutions, frequencies, and stats.
    """
    matriz_sol_promethee, result_sol_promethee, case_sol_promethee = extract_unique_solutions(matriz_poa_promethee)
    
    stats_promethee = compute_statistics(
        matriz_sol_promethee, 
        result_sol_promethee, 
        case_sol_promethee, 
        matriz_dif_vg_promethee
    )
    
    return {
        "matriz_sol": matriz_sol_promethee,
        "result_sol": result_sol_promethee,
        "case_sol": case_sol_promethee,
        "stats": stats_promethee
    }
