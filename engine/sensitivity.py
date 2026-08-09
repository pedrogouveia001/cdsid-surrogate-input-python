# pyrefly: ignore [missing-import]
import numpy as np
from .permutations import gerar_cases
from .normalization import escala_razao
from .promethee import comparacao_par_a_par

def calcular_pesos_roc(num_crit: int) -> np.ndarray:
    """
    Computes the ROC (Rank-Order Centroid) weights for each rank position.
    """
    weights = np.zeros(num_crit)
    for i in range(num_crit):
        soma = sum(1.0 / j for j in range(i + 1, num_crit + 1))
        weights[i] = (1.0 / num_crit) * soma
    return weights

def run_sensitivity_analysis(
    matriz_conseq: np.ndarray,
    tipocrit: list[int],
    niveis: list[int],
    rationality: str,
    rank_filters: list,
    holistic_evals: list,
    nomes_alt: list[str],
    variations_pct: list[float], # percentage e.g. [0.1, 0.1, ...]
    num_simulations: int = 10000,
    nomes_crit: list[str] = None,
    decomposition_prefs: list = None
):
    num_alt, num_crit = matriz_conseq.shape
    
    # 1. Precompute/get the base cases (permutations)
    cases_ordem_crit = gerar_cases(num_crit)
    
    # Apply rank position filters if present (same as solve API)
    if rank_filters:
        perms = cases_ordem_crit[:-1]
        valid_mask = np.ones(perms.shape[0], dtype=bool)
        for p, crit_idx in enumerate(rank_filters):
            if crit_idx is not None:
                crit_idx = int(crit_idx)
                valid_mask &= (perms[:, crit_idx] == (p + 1))
        filtered_perms = perms[valid_mask]
        cases_ordem_crit = np.vstack([filtered_perms, cases_ordem_crit[-1]])
        
    # Apply holistic filters under the ORIGINAL matrix to select the valid perms
    # Same as solve API
    matriz_conseq_norm_orig, max_crit_orig, min_crit_orig = escala_razao(matriz_conseq, tipocrit, niveis)
    
    # Compute surrogate results for original matrix to filter perms
    from .surrogate import calcular_surrogate
    surrogate_data = calcular_surrogate(cases_ordem_crit, matriz_conseq_norm_orig, tipocrit)
    
    is_compensatory = (rationality == 'compensatory')
    num_cases = cases_ordem_crit.shape[0]
    active_valid_mask = np.ones(num_cases, dtype=bool)
    
    nomes_alt_clean = [str(x).strip() for x in nomes_alt]
    for ev in holistic_evals:
        alt1_name = str(ev.get('alt1', '')).strip()
        alt2_name = str(ev.get('alt2', '')).strip()
        relation   = ev.get('relation', '')
        if not alt1_name or not alt2_name or relation not in ['>=', '<=']:
            continue
        idx1 = next((i for i, n in enumerate(nomes_alt_clean) if n == alt1_name), None)
        if alt2_name == 'fictitious':
            idx2 = 'fictitious'
        else:
            idx2 = next((i for i, n in enumerate(nomes_alt_clean) if n == alt2_name), None)
            
        if idx1 is None or idx2 is None:
            continue
            
        if is_compensatory:
            scores = surrogate_data['resultado_roc']
        else:
            scores = surrogate_data['resultado_promethee']
            
        if idx2 == 'fictitious':
            fict_val = ev.get('fictitiousValue')
            if fict_val is None:
                fict_val = float(np.min(np.max(scores, axis=1)))
            else:
                fict_val = float(fict_val)
                
            if relation == '>=':
                active_valid_mask &= (scores[:, idx1] >= fict_val - 1e-9)
            else:
                active_valid_mask &= (scores[:, idx1] <= fict_val + 1e-9)
        else:
            if relation == '>=':
                active_valid_mask &= (scores[:, idx1] >= scores[:, idx2] - 1e-9)
            else:
                active_valid_mask &= (scores[:, idx1] <= scores[:, idx2] + 1e-9)
            
    # Filter decomposition preferences
    if decomposition_prefs and nomes_crit:
        roc_w = calcular_pesos_roc(num_crit)
        weights_matrix_filter = np.zeros((num_cases, num_crit))
        for k in range(num_cases - 1):
            for i in range(num_crit):
                weights_matrix_filter[k, i] = roc_w[cases_ordem_crit[k, i] - 1]
        weights_matrix_filter[-1, :] = 1.0 / num_crit

        for dp in decomposition_prefs:
            crit_a_name = dp.get('critA')
            crit_b_name = dp.get('critB')
            relation = dp.get('relation')
            ratio = float(dp.get('ratio', 0.0))
            
            idx_a = next((i for i, n in enumerate(nomes_crit) if n == crit_a_name), None)
            idx_b = next((i for i, n in enumerate(nomes_crit) if n == crit_b_name), None)
            
            if idx_a is not None and idx_b is not None and relation in ['>=', '<=']:
                w_a = weights_matrix_filter[:, idx_a]
                w_b = weights_matrix_filter[:, idx_b]
                if relation == '>=':
                    active_valid_mask &= (w_b >= ratio * w_a - 1e-9)
                else:
                    active_valid_mask &= (w_b <= ratio * w_a + 1e-9)

    # Filter the weight permutations
    cases_filtered = cases_ordem_crit[active_valid_mask]
    P = cases_filtered.shape[0]
    
    if P == 0:
        raise ValueError("Inconsistency in filters: 0 valid weight permutations.")
        
    # Prepare the weights matrix W for valid cases
    roc_weights = calcular_pesos_roc(num_crit)
    W = np.zeros((P, num_crit))
    for k in range(P):
        case_row = cases_filtered[k]
        is_equal = np.all(case_row == 0)
        if is_equal:
            W[k, :] = 1.0 / num_crit
        else:
            for i in range(num_crit):
                rank = case_row[i] - 1
                W[k, i] = roc_weights[rank]
                
    # 2. Setup bounds for perturbation
    # For each criterion, find max and min from original matrix or levels
    max_crit = np.zeros(num_crit)
    min_crit = np.zeros(num_crit)
    for i in range(num_crit):
        if tipocrit[i] not in [2, 3]:
            max_crit[i] = np.max(matriz_conseq[:, i])
            min_crit[i] = np.min(matriz_conseq[:, i])
        else:
            if niveis[i] == 2:
                max_crit[i] = 1.0
                min_crit[i] = 0.0
            elif niveis[i] > 2:
                max_crit[i] = float(niveis[i])
                min_crit[i] = 1.0
                
    ranges = max_crit - min_crit
    # Delta for each criterion
    deltas = np.array(variations_pct) * ranges
    
    # 3. Monte Carlo Simulation
    winner_counts = np.zeros(num_alt, dtype=int)
    total_trials = num_simulations * P
    
    for m in range(num_simulations):
        # Perturb the consequence matrix
        # Generate random perturbation uniform in [-delta, delta]
        perturbation = np.random.uniform(-deltas, deltas, size=(num_alt, num_crit))
        matriz_perturbed = matriz_conseq + perturbation
        
        # Clip perturbed consequences within original bounds [min_crit, max_crit]
        for i in range(num_crit):
            matriz_perturbed[:, i] = np.clip(matriz_perturbed[:, i], min_crit[i], max_crit[i])
            # For discrete and integer criteria, round to nearest integer
            if tipocrit[i] in [2, 3, 4, 5]:
                matriz_perturbed[:, i] = np.round(matriz_perturbed[:, i])
                
        # Normalize
        matriz_conseq_norm, _, _ = escala_razao(matriz_perturbed, tipocrit, niveis)
        
        if is_compensatory:
            # Additive ROC model: scores = M_norm * W.T (shape: num_alt x P)
            S = np.dot(matriz_conseq_norm, W.T)
        else:
            # PROMETHEE II model: scores = net_flows_per_crit * W.T
            # Calculate pairwise matrix
            matriz_para_par = comparacao_par_a_par(matriz_conseq_norm, tipocrit)
            net_flows_per_crit = np.zeros((num_alt, num_crit))
            for c in range(num_crit):
                pos_flow = np.sum(matriz_para_par[c, :, :], axis=1) / (num_alt - 1)
                neg_flow = np.sum(matriz_para_par[c, :, :], axis=0) / (num_alt - 1)
                net_flows_per_crit[:, c] = pos_flow - neg_flow
            S = np.dot(net_flows_per_crit, W.T)
            
        # Find winners for each permutation
        max_scores = np.max(S, axis=0)
        is_winner = np.isclose(S, max_scores, atol=1e-9)
        winner_counts += np.sum(is_winner, axis=1)
        
    # Calculate probabilities
    probabilities = winner_counts / total_trials
    
    return {
        "alternatives": nomes_alt,
        "probabilities": probabilities.tolist(),
        "deltas": deltas.tolist(),
        "min_crit": min_crit.tolist(),
        "max_crit": max_crit.tolist()
    }
