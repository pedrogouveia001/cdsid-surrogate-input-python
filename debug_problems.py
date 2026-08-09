import sqlite3
import numpy as np
from app import search_next_decomposition_question, calcular_pesos_roc, gerar_cases, escala_razao, calcular_surrogate, extract_unique_solutions

def debug_all():
    conn = sqlite3.connect('spear.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, nome_problema, racionalidade FROM problema")
    problems = cursor.fetchall()
    
    for prob in problems:
        pid = prob['id']
        rationality = prob['racionalidade']
        
        cursor.execute("SELECT * FROM criterio WHERE ID_problema = ? ORDER BY id", (pid,))
        crits = cursor.fetchall()
        if not crits:
            continue
        nomes_crit = [c['nome_criterio'] for c in crits]
        tipocrit = [c['tipo_criterio'] for c in crits]
        niveis = [c['niveis'] for c in crits]
        
        cursor.execute("SELECT * FROM alternativa WHERE ID_problema = ? ORDER BY id", (pid,))
        alts = cursor.fetchall()
        nomes_alt = [a['nome_alternativa'] for a in alts]
        
        num_crit = len(crits)
        num_alt = len(alts)
        matriz_conseq = np.zeros((num_alt, num_crit))
        for i, alt in enumerate(alts):
            for j, crit in enumerate(crits):
                cursor.execute(
                    """SELECT valor_performance FROM matrizconsequencia 
                       WHERE ID_problema = ? AND ID_alternativa = ? AND ID_criterio = ?""",
                    (pid, alt['id'], crit['id'])
                )
                val = cursor.fetchone()
                if val:
                    matriz_conseq[i, j] = val['valor_performance']
                    
        cases_ordem_crit = gerar_cases(num_crit)
        matriz_conseq_norm, max_crit, min_crit = escala_razao(matriz_conseq, tipocrit, niveis)
        
        # Calculate surrogate data based on rationality
        is_compensatory = (rationality == 'compensatory')
        surrogate_data = calcular_surrogate(cases_ordem_crit, matriz_conseq_norm, tipocrit)
        poa = surrogate_data['matriz_poa'] if is_compensatory else surrogate_data['matriz_poa_promethee']
        
        num_cases = cases_ordem_crit.shape[0]
        roc_w = calcular_pesos_roc(num_crit)
        weights_matrix = np.zeros((num_cases, num_crit))
        for k in range(num_cases - 1):
            for i in range(num_crit):
                weights_matrix[k, i] = roc_w[cases_ordem_crit[k, i] - 1]
        weights_matrix[-1, :] = 1.0 / num_crit
        
        active_valid_mask = np.ones(num_cases, dtype=bool)
        
        matriz_sol, result_sol, case_sol = extract_unique_solutions(poa)
        converged = (len(matriz_sol) == 1)
        
        next_q = search_next_decomposition_question(
            weights_matrix,
            active_valid_mask,
            num_crit,
            nomes_crit,
            poa,
            set()
        )
        print(f"Problem {pid}: '{prob['nome_problema']}' ({rationality}) | unique sols: {len(matriz_sol)} | converged: {converged} | question: {next_q}")
        
    conn.close()

if __name__ == '__main__':
    debug_all()
