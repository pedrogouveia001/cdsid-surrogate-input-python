# pyrefly: ignore [missing-import]
import numpy as np

def compute_statistics(
    matriz_sol: np.ndarray, 
    result_sol: np.ndarray, 
    case_sol: np.ndarray, 
    matriz_dif_vg: np.ndarray
):
    """
    Computes aggregated statistical results for either ROC or PROMETHEE.
    Based on Delphi `Surrogate` / `Promethee` procedures.
    
    Args:
        matriz_sol: np.ndarray of shape (num_sols, num_alt)
            Each row is a binary vector indicating which alternatives belong to that solution.
        result_sol: np.ndarray of shape (num_sols,) (frequency of each solution)
        case_sol: np.ndarray of shape (num_cases,) mapping each case to its solution index (1-based)
        matriz_dif_vg: np.ndarray of shape (num_cases, num_alt)
            For each case and each alternative, the difference from the best value (loss from best).
            
    Returns:
        dict containing the computed statistics.
    
    Key concepts:
        - media_dif_sol[i, j]: avg loss of sol j's alternatives when sol i WINS
        - max_dif_sol[i, j]:   max loss of sol j's alternatives when sol i WINS
        - media_geral[i]:      avg loss across all competitors when sol i WINS
        - maximo_geral[i]:     max loss across all competitors when sol i WINS
        - media_geral_naosol[i]: avg loss of sol i's alternatives when sol i DOES NOT WIN
        - maximo_geral_naosol[i]: max loss of sol i's alternatives when sol i DOES NOT WIN
    """
    num_sols, num_alt = matriz_sol.shape
    num_cases = matriz_dif_vg.shape[0]
    
    # Exclude the surrogate (equal weights, last case) in denominator logic
    num_valid_cases = num_cases - 1

    # --- Per-solution-pair statistics (sol i wins vs sol j) ---
    media_dif_sol = np.zeros((num_sols, num_sols), dtype=float)
    max_dif_sol = np.zeros((num_sols, num_sols), dtype=float)
    min_dif_sol = np.full((num_sols, num_sols), np.inf, dtype=float)

    # --- Per-solution global win statistics ---
    media_geral = np.zeros(num_sols, dtype=float)
    maximo_geral = np.zeros(num_sols, dtype=float)
    minimo_geral = np.full(num_sols, np.inf, dtype=float)

    # --- Per-solution NON-WIN (naosol) statistics ---
    # These measure: when solution i is NOT the winner, how much does it lose?
    media_geral_naosol = np.zeros(num_sols, dtype=float)
    maximo_geral_naosol = np.zeros(num_sols, dtype=float)
    minimo_geral_naosol = np.full(num_sols, np.inf, dtype=float)

    # Precompute which cases belong to each solution (1-based indexing in case_sol)
    # case_sol[k] == i+1 means case k belongs to solution i
    sol_cases = []
    nonsol_cases = []
    for i in range(num_sols):
        wins = np.where(case_sol == i + 1)[0]
        losses = np.where(case_sol != i + 1)[0]
        # Exclude the last equal-weights case from both
        wins_valid = wins[wins < num_cases - 1]
        losses_valid = losses[losses < num_cases - 1]
        sol_cases.append(wins_valid)
        nonsol_cases.append(losses_valid)

    # ---- Compute pairwise stats: solution i wins, solution j loses ----
    for i in range(num_sols):
        win_cases = sol_cases[i]
        num_win = len(win_cases)
        
        for j in range(num_sols):
            if i == j:
                continue

            # Alternatives that belong to solution j
            j_alts = np.where(matriz_sol[j, :] == 1)[0]
            num_j_alts = len(j_alts)
            if num_j_alts == 0 or num_win == 0:
                continue

            # For each winning case, compute the loss of sol j's alternatives
            # matriz_dif_vg[k, t] = (best_value - value_of_alt_t) for case k
            # So loss of sol j's alternatives when sol i wins:
            losses_j_in_i_wins = matriz_dif_vg[np.ix_(win_cases, j_alts)]  # shape (num_win, num_j_alts)

            # Sum over alternatives for each case
            losses_per_case = np.sum(losses_j_in_i_wins, axis=1)  # shape (num_win,)

            # media_dif_sol[i, j]: mean loss per alternative per case
            media_dif_sol[i, j] = np.sum(losses_j_in_i_wins) / (num_j_alts * num_win)

            # max_dif_sol[i, j]: maximum single-alternative loss across all winning cases
            max_dif_sol[i, j] = np.max(losses_j_in_i_wins)

            # min_dif_sol[i, j]: minimum single-alternative loss
            min_val = np.min(losses_j_in_i_wins)
            min_dif_sol[i, j] = min_val

        # media_geral[i]: avg loss of all non-i alternatives when sol i wins
        # = average of media_dif_sol[i, j] across all j != i
        valid_js = [j for j in range(num_sols) if j != i and np.sum(matriz_sol[j, :]) > 0 and num_win > 0]
        if valid_js:
            media_geral[i] = np.mean([media_dif_sol[i, j] for j in valid_js])
            maximo_geral[i] = max(max_dif_sol[i, j] for j in valid_js)
            min_cands = [min_dif_sol[i, j] for j in valid_js if min_dif_sol[i, j] < np.inf]
            minimo_geral[i] = min(min_cands) if min_cands else 0.0

    # ---- Compute naosol stats: when solution i does NOT win, how much does it lose? ----
    for i in range(num_sols):
        loss_cases = nonsol_cases[i]
        num_loss = len(loss_cases)

        # Alternatives that belong to solution i
        i_alts = np.where(matriz_sol[i, :] == 1)[0]
        num_i_alts = len(i_alts)

        if num_i_alts == 0 or num_loss == 0:
            continue

        # For each losing case, the loss of sol i's alternatives
        # matriz_dif_vg[k, t] = (best_value - value_of_alt_t) for case k
        losses_i_in_losses = matriz_dif_vg[np.ix_(loss_cases, i_alts)]  # shape (num_loss, num_i_alts)

        media_geral_naosol[i] = np.sum(losses_i_in_losses) / (num_i_alts * num_loss)
        maximo_geral_naosol[i] = np.max(losses_i_in_losses)
        min_val = np.min(losses_i_in_losses)
        minimo_geral_naosol[i] = min_val if min_val < np.inf else 0.0

    # Replace remaining inf values with 0
    min_dif_sol[np.isinf(min_dif_sol)] = 0.0
    minimo_geral[np.isinf(minimo_geral)] = 0.0
    minimo_geral_naosol[np.isinf(minimo_geral_naosol)] = 0.0

    # Calculate Standard Deviations (pairwise, for sol i winning)
    desvio_padrao_dif_sol = np.zeros((num_sols, num_sols), dtype=float)
    desvio_padrao_geral = np.zeros(num_sols, dtype=float)

    for i in range(num_sols):
        win_cases = sol_cases[i]
        num_win = len(win_cases)
        if num_win == 0:
            continue

        for j in range(num_sols):
            if i == j:
                continue

            j_alts = np.where(matriz_sol[j, :] == 1)[0]
            num_j_alts = len(j_alts)
            if num_j_alts == 0:
                continue

            losses_j = matriz_dif_vg[np.ix_(win_cases, j_alts)]  # (num_win, num_j_alts)
            mean_j = media_dif_sol[i, j]
            variance = np.sum((losses_j - mean_j) ** 2) / (num_j_alts * num_win)
            desvio_padrao_dif_sol[i, j] = np.sqrt(variance)

        # Global std for sol i winning (across all j != i)
        valid_js = [j for j in range(num_sols) if j != i and np.sum(matriz_sol[j, :]) > 0]
        if valid_js:
            all_losses = []
            mean_g = media_geral[i]
            for j in valid_js:
                j_alts = np.where(matriz_sol[j, :] == 1)[0]
                if len(j_alts) > 0:
                    losses_j = matriz_dif_vg[np.ix_(win_cases, j_alts)]
                    all_losses.append(losses_j.flatten())
            if all_losses:
                all_losses_arr = np.concatenate(all_losses)
                desvio_padrao_geral[i] = np.std(all_losses_arr)

    return {
        "media_dif_sol": media_dif_sol,
        "max_dif_sol": max_dif_sol,
        "min_dif_sol": min_dif_sol,
        "media_geral": media_geral,
        "maximo_geral": maximo_geral,
        "minimo_geral": minimo_geral,
        "media_geral_naosol": media_geral_naosol,
        "maximo_geral_naosol": maximo_geral_naosol,
        "minimo_geral_naosol": minimo_geral_naosol,
        "desvio_padrao_dif_sol": desvio_padrao_dif_sol,
        "desvio_padrao_geral": desvio_padrao_geral
    }
