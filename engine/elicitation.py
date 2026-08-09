# pyrefly: ignore [missing-import]
import numpy as np

def analise_para_elicitacao(
    cases_ordem_crit: np.ndarray, 
    matriz_poa: np.ndarray, 
    result_sol: np.ndarray, 
    matriz_sol: np.ndarray
) -> dict:
    """
    Port of `AnaliseParaElicitacao` from Delphi.
    Analyzes the impact of criteria preferences on the top 2 alternatives (X and Z).
    
    Args:
        cases_ordem_crit: Permutations of criteria (num_cases, num_crit)
        matriz_poa: POA matrix (num_cases, num_alt)
        result_sol: Frequencies of unique solutions
        matriz_sol: Unique solutions matrix
        
    Returns:
        Dictionary with probability matrices for X, Z, and Others.
    """
    num_cases, num_crit = cases_ordem_crit.shape
    num_alt = matriz_poa.shape[1]
    
    # Identify top 2 alternatives (X and Z) from the most frequent solutions
    # We find the alternatives associated with the most frequent solutions
    ordem_sol = np.argsort(-result_sol)
    
    # Getting the alternatives that represent Sol 1 and Sol 2
    # In Delphi, altx and altz are determined differently (Ordemalt[0], Ordemalt[1]), 
    # which is the frequency of each alternative individually.
    # Let's calculate alternative frequencies first.
    alt_freq = np.zeros(num_alt)
    for j in range(num_alt):
        # Count cases where alternative j is part of the POA
        alt_freq[j] = np.sum(matriz_poa[:-1, j])
        
    ordem_alt = np.argsort(-alt_freq)
    
    if len(ordem_alt) >= 2:
        alt_x = ordem_alt[0]
        alt_z = ordem_alt[1]
    elif len(ordem_alt) == 1:
        alt_x = ordem_alt[0]
        alt_z = -1
    else:
        alt_x = -1
        alt_z = -1
        
    matriz_prob_x = np.zeros((num_crit, num_crit), dtype=float)
    matriz_prob_z = np.zeros((num_crit, num_crit), dtype=float)
    matriz_prob_outros = np.zeros((num_crit, num_crit), dtype=float)
    
    # We ignore the last dummy case (surrogate equal weights)
    valid_cases = num_cases - 1
    
    # We want to find the probability that Alt X wins GIVEN that Crit C1 > Crit C2
    for c1 in range(num_crit):
        for c2 in range(num_crit):
            if c1 == c2: continue
            
            # Count cases where C1 > C2
            # Delphi uses rank: CasesOrdemcrit[k, c1] < CasesOrdemcrit[k, c2] means C1 is more important than C2
            # (Rank 1 is better than Rank 2)
            cases_c1_gt_c2 = cases_ordem_crit[:-1, c1] < cases_ordem_crit[:-1, c2]
            total_c1_gt_c2 = np.sum(cases_c1_gt_c2)
            
            if total_c1_gt_c2 > 0:
                # Cases where C1 > C2 AND Alt X is the solution
                if alt_x != -1:
                    x_wins = np.sum(cases_c1_gt_c2 & (matriz_poa[:-1, alt_x] == 1))
                    matriz_prob_x[c1, c2] = x_wins / total_c1_gt_c2
                    
                # Cases where C1 > C2 AND Alt Z is the solution
                if alt_z != -1:
                    z_wins = np.sum(cases_c1_gt_c2 & (matriz_poa[:-1, alt_z] == 1))
                    matriz_prob_z[c1, c2] = z_wins / total_c1_gt_c2
                    
                # Other alternatives
                matriz_prob_outros[c1, c2] = 1.0 - matriz_prob_x[c1, c2] - matriz_prob_z[c1, c2]
                if matriz_prob_outros[c1, c2] < 0:
                    matriz_prob_outros[c1, c2] = 0.0

    return {
        "alt_x": int(alt_x),
        "alt_z": int(alt_z),
        "matriz_prob_x": matriz_prob_x,
        "matriz_prob_z": matriz_prob_z,
        "matriz_prob_outros": matriz_prob_outros
    }
