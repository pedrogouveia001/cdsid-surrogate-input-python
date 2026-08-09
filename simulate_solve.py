import requests
import sqlite3
import numpy as np

def simulate():
    conn = sqlite3.connect('spear.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get last problem
    cursor.execute("SELECT id, nome_problema, racionalidade FROM problema ORDER BY id DESC LIMIT 1")
    prob = cursor.fetchone()
    if not prob:
        print("No problems found")
        return
        
    pid = prob['id']
    rationality = prob['racionalidade']
    print(f"SIMULATING FOR PROBLEM ID {pid}: {prob['nome_problema']} ({rationality})")
    
    cursor.execute("SELECT * FROM criterio WHERE ID_problema = ? ORDER BY id", (pid,))
    crits = cursor.fetchall()
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
                
    conn.close()
    
    payload = {
        "problemName": prob['nome_problema'],
        "rationality": rationality,
        "numCrit": num_crit,
        "numAlt": num_alt,
        "nomeCrit": nomes_crit,
        "tipoCrit": tipocrit,
        "niveisCrit": niveis,
        "nomeAlt": nomes_alt,
        "matrizConseq": matriz_conseq.tolist(),
        "rankFilters": [None] * num_crit,
        "holisticEvaluations": [],
        "decompositionPreferences": [],
        "excludedPairs": []
    }
    
    res = requests.post("http://127.0.0.1:5000/api/solve", json=payload)
    data = res.json()
    print("Success:", data.get("success"))
    if not data.get("success"):
        print("Error:", data.get("error"))
    else:
        print("Decomposition Question in response:", data.get("decompositionQuestion"))
        print("Converged (unique solutions length):", len(data[rationality if rationality == 'compensatory' else 'promethee']['matrizSol']))
        print("Total cases:", data[rationality if rationality == 'compensatory' else 'promethee']['totalCases'])

if __name__ == '__main__':
    simulate()
