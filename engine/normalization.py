# pyrefly: ignore [missing-import]
import numpy as np

def escala_razao(matriz_conseq: np.ndarray, tipocrit: list[int], niveis: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Port of `EscalaRazao` Delphi procedure.
    Normalizes the consequence matrix based on criteria types (maximization vs minimization)
    and discrete levels.

    Args:
        matriz_conseq: np.ndarray of shape (num_alt, num_crit)
        tipocrit: list of ints representing the type of each criterion
                  (0: Cont Min, 1: Cont Max, 2: Disc Min, 3: Disc Max, 4: Int Min, 5: Int Max)
        niveis: list of ints representing the number of levels for discrete criteria (ignored for others)

    Returns:
        tuple containing:
            - matriz_conseq_norm: Normalized consequence matrix of shape (num_alt, num_crit)
            - max_crit: Maximum values for each criterion
            - min_crit: Minimum values for each criterion
    """
    num_alt, num_crit = matriz_conseq.shape
    matriz_conseq_norm = np.zeros_like(matriz_conseq, dtype=float)
    
    max_crit = np.zeros(num_crit)
    min_crit = np.zeros(num_crit)

    for i in range(num_crit):
        if tipocrit[i] not in [2, 3]:
            # Continuous or Integer criteria: empirical max/min from the data
            max_crit[i] = np.max(matriz_conseq[:, i])
            min_crit[i] = np.min(matriz_conseq[:, i])
        else:
            # Discrete criteria: max/min based on 'niveis' configuration
            if niveis[i] == 2:
                max_crit[i] = 1.0
                min_crit[i] = 0.0
            elif niveis[i] > 2:
                max_crit[i] = float(niveis[i])
                min_crit[i] = 1.0

    for i in range(num_crit):
        denominator = max_crit[i] - min_crit[i]
        # Avoid division by zero if all values are the same
        if denominator == 0:
            denominator = 1.0
            
        for j in range(num_alt):
            if tipocrit[i] in [1, 3, 5]:
                # Maximization criteria
                matriz_conseq_norm[j, i] = (matriz_conseq[j, i] - min_crit[i]) / denominator
            elif tipocrit[i] in [0, 2, 4]:
                # Minimization criteria
                matriz_conseq_norm[j, i] = (matriz_conseq[j, i] - max_crit[i]) / (-denominator)

    return matriz_conseq_norm, max_crit, min_crit
