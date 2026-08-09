# pyrefly: ignore [missing-import]
import numpy as np

def apply_decision_rules(
    result_sol: np.ndarray, 
    matriz_sol: np.ndarray, 
    stats: dict, 
    total_cases: int
) -> dict:
    """
    Port of decision rules from `btnshowresultsClick` and `ResultadosPromethee`.
    Evaluates probability, max loss, and avg loss against predefined thresholds.
    
    Args:
        result_sol: Frequencies of each unique solution
        matriz_sol: The unique solutions matrix
        stats: Dictionary of computed statistics
        total_cases: Total number of valid permutations/cases
        
    Returns:
        Dictionary with recommendation text and details.
    """
    # Sort solutions by descending probability (frequency)
    ordem_sol = np.argsort(-result_sol)
    
    faixa_prob = [0.80, 0.70, 0.60, 0.50]
    epsilon = [0.5, 0.4, 0.3, 0.2]   # Max Loss Thresholds
    omega = [0.25, 0.2, 0.15, 0.10]  # Average Loss Thresholds
    
    media_naosol = stats['media_geral_naosol']
    max_naosol = stats['maximo_geral_naosol']
    
    def get_alts_for_sol(sol_idx):
        return np.where(matriz_sol[sol_idx] == 1)[0].tolist()
        
    # Check 1-Alternative
    p1 = result_sol[ordem_sol[0]] / total_cases
    for i in range(4):
        if i == 0 and p1 >= faixa_prob[i]: pass
        elif i > 0 and faixa_prob[i-1] > p1 >= faixa_prob[i]: pass
        else: continue
        
        if max_naosol[ordem_sol[0]] < epsilon[i] and media_naosol[ordem_sol[0]] < omega[i]:
            alts = get_alts_for_sol(ordem_sol[0])
            return {
                "status": "Best Alternative",
                "recommended_alts": alts,
                "probability": p1 * 100,
                "rule_level": i
            }
            
    # Check 2-Alternatives
    if len(result_sol) >= 2:
        p12 = (result_sol[ordem_sol[0]] + result_sol[ordem_sol[1]]) / total_cases
        for i in range(4):
            if i == 0 and p12 >= faixa_prob[i]: pass
            elif i > 0 and faixa_prob[i-1] > p12 >= faixa_prob[i]: pass
            else: continue
            
            if (max_naosol[ordem_sol[0]] < epsilon[i] and media_naosol[ordem_sol[0]] < omega[i] and
                max_naosol[ordem_sol[1]] < epsilon[i] and media_naosol[ordem_sol[1]] < omega[i]):
                alts1 = get_alts_for_sol(ordem_sol[0])
                alts2 = get_alts_for_sol(ordem_sol[1])
                return {
                    "status": "Two Alternatives are Competitive",
                    "recommended_alts": list(set(alts1 + alts2)),
                    "probability": p12 * 100,
                    "rule_level": i
                }

    # Check 3-Alternatives
    if len(result_sol) >= 3:
        p123 = (result_sol[ordem_sol[0]] + result_sol[ordem_sol[1]] + result_sol[ordem_sol[2]]) / total_cases
        for i in range(4):
            if i == 0 and p123 >= faixa_prob[i]: pass
            elif i > 0 and faixa_prob[i-1] > p123 >= faixa_prob[i]: pass
            else: continue
            
            if (max_naosol[ordem_sol[0]] < epsilon[i] and media_naosol[ordem_sol[0]] < omega[i] and
                max_naosol[ordem_sol[1]] < epsilon[i] and media_naosol[ordem_sol[1]] < omega[i] and
                max_naosol[ordem_sol[2]] < epsilon[i] and media_naosol[ordem_sol[2]] < omega[i]):
                alts1 = get_alts_for_sol(ordem_sol[0])
                alts2 = get_alts_for_sol(ordem_sol[1])
                alts3 = get_alts_for_sol(ordem_sol[2])
                return {
                    "status": "Three Alternatives are Competitive",
                    "recommended_alts": list(set(alts1 + alts2 + alts3)),
                    "probability": p123 * 100,
                    "rule_level": i
                }

    return {
        "status": "Unable to make it",
        "recommended_alts": [],
        "probability": p1 * 100,
        "rule_level": -1
    }
