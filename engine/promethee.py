# pyrefly: ignore [missing-import]
import numpy as np

def comparacao_par_a_par(matriz_conseq: np.ndarray, tipocrit: list[int]) -> np.ndarray:
    """
    Port of `ComparacaoParAPar` Delphi procedure.
    Constructs the pairwise comparison matrix for each criterion.
    Strict preference is applied: 1 if alternative i is strictly better than j, 0 otherwise.

    Args:
        matriz_conseq: np.ndarray of shape (num_alt, num_crit)
        tipocrit: list of ints representing the type of each criterion
                  (1,3,5 are maximization; 0,2,4 are minimization)

    Returns:
        np.ndarray of shape (num_crit, num_alt, num_alt) representing preferences
    """
    num_alt, num_crit = matriz_conseq.shape
    matriz_para_par = np.zeros((num_crit, num_alt, num_alt), dtype=float)

    for xcrit in range(num_crit):
        if tipocrit[xcrit] in [1, 3, 5]:
            # Maximization
            for l in range(num_alt):
                for c in range(num_alt):
                    if l != c:
                        # If a(l) > a(c) strictly
                        if matriz_conseq[l, xcrit] > matriz_conseq[c, xcrit]:
                            matriz_para_par[xcrit, l, c] = 1.0
                        else:
                            matriz_para_par[xcrit, l, c] = 0.0
        elif tipocrit[xcrit] in [0, 2, 4]:
            # Minimization
            for l in range(num_alt):
                for c in range(num_alt):
                    if l != c:
                        # If a(l) < a(c) strictly
                        if matriz_conseq[l, xcrit] < matriz_conseq[c, xcrit]:
                            matriz_para_par[xcrit, l, c] = 1.0
                        else:
                            matriz_para_par[xcrit, l, c] = 0.0

    return matriz_para_par


def calculo_fluxos(matriz_para_par: np.ndarray, peso_crit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Port of `CalculoFluxos` Delphi procedure.
    Calculates outranking matrix, positive flow, negative flow, and net flow (PROMETHEE II).

    Args:
        matriz_para_par: np.ndarray of shape (num_crit, num_alt, num_alt)
        peso_crit: np.ndarray of shape (num_crit,) representing criterion weights

    Returns:
        tuple containing:
            - positive_flow: np.ndarray of shape (num_alt,)
            - negative_flow: np.ndarray of shape (num_alt,)
            - net_flow: np.ndarray of shape (num_alt,)
    """
    num_crit, num_alt, _ = matriz_para_par.shape
    sob_class_matrix = np.zeros((num_alt, num_alt), dtype=float)

    # Calculate Outranking Matrix (SobClassMatrix)
    for i in range(num_alt):
        for j in range(num_alt):
            if i != j:
                for xcrit in range(num_crit):
                    sob_class_matrix[i, j] += peso_crit[xcrit] * matriz_para_par[xcrit, i, j]

    # Positive Flow
    positive_flow = np.zeros(num_alt, dtype=float)
    for i in range(num_alt):
        soma = np.sum(sob_class_matrix[i, :])
        positive_flow[i] = soma / (num_alt - 1)

    # Negative Flow
    negative_flow = np.zeros(num_alt, dtype=float)
    for j in range(num_alt):
        soma = np.sum(sob_class_matrix[:, j])
        negative_flow[j] = soma / (num_alt - 1)

    # Net Flow
    net_flow = positive_flow - negative_flow

    return positive_flow, negative_flow, net_flow
