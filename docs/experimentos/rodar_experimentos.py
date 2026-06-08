import subprocess
import csv
from configs import CONFIGS

RESULTADOS = []

for i, cfg in enumerate(CONFIGS):
    print(f"\nRodando experimento {i+1}: {cfg}\n")

    comando = [
         "python", "./solucao.py",
        "--K", str(cfg["K"]),
        "--alpha", str(cfg["alpha"]),
        "--gamma", str(cfg["gamma"]),
        "--recarregar"
    ]

    subprocess.run(comando)

    # Ler resultados das pistas 17 e 18
    for pista in ["17", "18"]:
        arquivo = f"q_learning_pista_{pista}.txt"

        with open(arquivo) as f:
            texto = f.read()

        sucesso = "SIM" in texto

        # Extrair reward (simples)
        reward = float(
            texto.split("Recompensa total: ")[1].split("\n")[0]
        )

        RESULTADOS.append({
            "config": str(cfg),
            "pista": pista,
            "sucesso": sucesso,
            "reward": reward
        })

# salvar csv
with open("./resultados.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=RESULTADOS[0].keys())
    writer.writeheader()
    writer.writerows(RESULTADOS)

print("\nExperimentos finalizados!")