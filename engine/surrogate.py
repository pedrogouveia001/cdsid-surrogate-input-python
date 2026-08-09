# pyrefly: ignore [missing-import]
import numpy as np
from .promethee import comparacao_par_a_par, calculo_fluxos

def calcular_pesos_roc(num_crit: int) -> np.ndarray:
    """
    Precomputes the ROC (Rank-Order Centroid) weights for each rank position.
    weights[0] is for the 1st rank, weights[num_crit-1] is for the last rank.
    Formula: W_i = (1/num_crit) * sum_{j=i}^{num_crit} (1/j)
    """
    weights = np.zeros(num_crit)
    for i in range(num_crit):
        soma = sum(1.0 / j for j in range(i + 1, num_crit + 1))
        weights[i] = (1.0 / num_crit) * soma
    return weights

def calcular_surrogate(
    cases_ordem_crit: np.ndarray, 
    matriz_conseq_norm: np.ndarray, 
    tipocrit: list[int]
):
    """
    Port of `Surrogate` Delphi procedure.
    Evaluates all permutations using ROC weights and PROMETHEE II.
    
    Args:
        cases_ordem_crit: np.ndarray of shape (N!+1, num_crit)
        matriz_conseq_norm: np.ndarray of shape (num_alt, num_crit)
        tipocrit: list of ints for criteria types
        
    Returns:
        Dictionary with all calculated matrices and statistics.
    """
    num_cases = cases_ordem_crit.shape[0]
    num_alt, num_crit = matriz_conseq_norm.shape
    
    roc_weights = calcular_pesos_roc(num_crit)
    
    # Initialize ROC matrices
    resultado_roc = np.zeros((num_cases, num_alt), dtype=float)
    matriz_dif_vg = np.zeros((num_cases, num_alt), dtype=float)
    matriz_poa = np.zeros((num_cases, num_alt), dtype=int)
    
    # Initialize PROMETHEE matrices
    resultado_promethee = np.zeros((num_cases, num_alt), dtype=float)
    matriz_dif_vg_promethee = np.zeros((num_cases, num_alt), dtype=float)
    matriz_poa_promethee = np.zeros((num_cases, num_alt), dtype=int)
    
    # Pre-calculate pairwise comparison matrix for PROMETHEE (independent of weights)
    matriz_para_par = comparacao_par_a_par(matriz_conseq_norm, tipocrit)
    
    peso_crit_case = np.zeros(num_crit, dtype=float)
    
    # Process all normal cases (permutations)
    for k in range(num_cases - 1):
        # Delphi logic: CasesOrdemcrit[k, i] is the rank (1-based) of criterion i.
        # So we assign the weight for that rank to criterion i.
        for i in range(num_crit):
            rank = cases_ordem_crit[k, i] - 1
            peso_crit_case[i] = roc_weights[rank]
            
        # ROC Calculation
        for j in range(num_alt):
            resultado_roc[k, j] = np.sum(matriz_conseq_norm[j, :] * peso_crit_case)
            
        aux_resultado = np.max(resultado_roc[k, :])
        for j in range(num_alt):
            matriz_dif_vg[k, j] = aux_resultado - resultado_roc[k, j]
            # Floating point comparison might be risky, but we'll use a small tolerance or exact match
            if np.isclose(aux_resultado, resultado_roc[k, j], atol=1e-9):
                matriz_poa[k, j] = 1
                
        # PROMETHEE Calculation
        _, _, net_flow = calculo_fluxos(matriz_para_par, peso_crit_case)
        for j in range(num_alt):
            resultado_promethee[k, j] = net_flow[j]
            
        aux_resultado_prom = np.max(resultado_promethee[k, :])
        for j in range(num_alt):
            matriz_dif_vg_promethee[k, j] = aux_resultado_prom - resultado_promethee[k, j]
            if np.isclose(aux_resultado_prom, resultado_promethee[k, j], atol=1e-9):
                matriz_poa_promethee[k, j] = 1
                
    # Process the last case (equal weights)
    k = num_cases - 1
    peso_equal = np.ones(num_crit) / num_crit
    
    # ROC Calculation
    for j in range(num_alt):
        resultado_roc[k, j] = np.sum(matriz_conseq_norm[j, :] * peso_equal)
        
    aux_resultado = np.max(resultado_roc[k, :])
    for j in range(num_alt):
        matriz_dif_vg[k, j] = aux_resultado - resultado_roc[k, j]
        if np.isclose(aux_resultado, resultado_roc[k, j], atol=1e-9):
            matriz_poa[k, j] = 1
            
    # PROMETHEE Calculation
    _, _, net_flow = calculo_fluxos(matriz_para_par, peso_equal)
    for j in range(num_alt):
        resultado_promethee[k, j] = net_flow[j]
        
    aux_resultado_prom = np.max(resultado_promethee[k, :])
    for j in range(num_alt):
        matriz_dif_vg_promethee[k, j] = aux_resultado_prom - resultado_promethee[k, j]
        if np.isclose(aux_resultado_prom, resultado_promethee[k, j], atol=1e-9):
            matriz_poa_promethee[k, j] = 1
            
    # Group unique solutions (ROC)
    matriz_sol, result_sol, case_sol = extract_unique_solutions(matriz_poa)
    
    # Note: Further statistics (MediaDifSol, etc.) are computed on unique solutions.
    # We return the basic matrices to avoid doing all logic here if it can be modularized.
    
    return {
        "resultado_roc": resultado_roc,
        "matriz_dif_vg": matriz_dif_vg,
        "matriz_poa": matriz_poa,
        "resultado_promethee": resultado_promethee,
        "matriz_dif_vg_promethee": matriz_dif_vg_promethee,
        "matriz_poa_promethee": matriz_poa_promethee,
        "matriz_sol": matriz_sol,
        "result_sol": result_sol,
        "case_sol": case_sol
    }


def extract_unique_solutions(matriz_poa: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds unique solution profiles in matriz_poa (excluding the last equal weights row)
    and counts their occurrences.
    """
    num_cases = matriz_poa.shape[0] - 1  # Exclude the last case for grouping
    num_alt = matriz_poa.shape[1]
    
    # Find unique rows
    matriz_sol = []
    result_sol = []
    case_sol = np.zeros(num_cases + 1, dtype=int)
    
    for i in range(num_cases):
        row = matriz_poa[i, :]
        found = False
        for k, sol in enumerate(matriz_sol):
            if np.array_equal(row, sol):
                result_sol[k] += 1
                case_sol[i] = k + 1  # 1-based indexing as in Delphi
                found = True
                break
        
        if not found:
            matriz_sol.append(row)
            result_sol.append(1)
            case_sol[i] = len(matriz_sol)
            
    return np.array(matriz_sol), np.array(result_sol), case_sol
