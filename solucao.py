"""
Esqueleto da sua solução para o EP do carrinho (versão tabular).

Você deve implementar:
    - AgenteQLearning  (tabular)

E preencher main() para orquestrar:
    1. Treinamento round-robin nas pistas 01-16 → salva treinamento/qlearning.pkl.
    2. Avaliação gulosa (ε = 0) nas pistas de holdout 17 e 18 → gera
       q_learning_pista_17.txt e q_learning_pista_18.txt (formato do README §4.3).

Uso:
    python solucao.py                         # treina (se necessário) + avalia em 17 e 18
    python solucao.py --recarregar            # força re-treino (ignora pickle existente)
    python solucao.py --avaliar pistas/X.txt  # apenas avalia o modelo salvo em X

Termos como `step`, `reset`, `obs`, `action`, `reward` são mantidos em inglês
por serem o vocabulário canônico de Aprendizado por Reforço (Sutton & Barto).
"""

import sys
import random
import argparse
import pickle
from pathlib import Path

import numpy as np

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from env import AmbienteCarro  # noqa: E402
# from visualize import renderizar_episodio  # use isto para animar seu agente no terminal


# === Configuração ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Diretório onde o modelo treinado será salvo via pickle (ver enunciado/anexo_b_pickle.md)
DIR_TREINAMENTO = Path("treinamento")
DIR_TREINAMENTO.mkdir(exist_ok=True)

# Conjuntos de pistas
PISTAS_TREINO = [f"pistas/pista_{i:02d}.txt" for i in range(1, 17)]   # 01..16
PISTAS_HOLDOUT = [f"pistas/pista_{i:02d}.txt" for i in range(17, 19)] # 17, 18


# ============================================================================
# Q-LEARNING TABULAR
# ============================================================================

class AgenteQLearning:
    """
    TODO: implementar Q-Learning tabular com discretização do estado.

    Dicas:
    - O estado é um vetor de 6 floats em [0, 1]. Discretize cada componente
      em K baldes (sugerido K=5 para baseline) para chegar a uma chave discreta.
    - Use um dict {chave_discreta: np.array(n_actions)} ou um np.ndarray
      multidimensional para a tabela Q.
    - Atualização: Q(s,a) ← Q(s,a) + α [r + γ max_{a'} Q(s', a') − Q(s,a)]
    """
 
    def __init__(self, obs_dim, n_actions, K=5, alpha=0.1, gamma=0.99,
             eps_inicial=1.0, eps_final=0.05):

        self.n_actions = n_actions
        self.K = K

        self.alpha = alpha
        self.gamma = gamma

        self.eps = eps_inicial
        self.eps_inicial = eps_inicial
        self.eps_final = eps_final

        self.Q = {}

    def discretizar(self, obs):
        return tuple(
            min(int(v * self.K), self.K - 1)
            for v in obs
        )

    def obter_q(self, estado):
        if estado not in self.Q:
            self.Q[estado] = np.random.uniform(
            -0.01,
            0.01,
            self.n_actions
        )

        return self.Q[estado]    


    def escolher_acao(self, obs):

        estado = self.discretizar(obs)

        if random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)

        return int(np.argmax(self.obter_q(estado)))

    def atualizar(self, s, a, r, s_prox, terminou):

        estado = self.discretizar(s)
        estado_prox = self.discretizar(s_prox)

        q_atual = self.obter_q(estado)[a]

        if terminou:
            alvo = r
        else:
            alvo = r + self.gamma * np.max(
                self.obter_q(estado_prox)
            )

        self.Q[estado][a] += self.alpha * (
            alvo - q_atual
        )

    @classmethod
    def from_modelo(cls, modelo):

        agente = cls(
            obs_dim=6,
            n_actions=5,
            K=modelo["discretization_K"]
        )

        agente.Q = modelo["q_table"]
        agente.eps = 0.0
        return agente



# ============================================================================
# LOOP DE TREINAMENTO (round-robin nas 16 pistas de treino)
# ============================================================================

def treinar_round_robin(pistas_treino, agente, n_episodios_por_pista,
                       max_passos, decaimento_eps_episodios, verbose=True):
    """
    Loop de treinamento em round-robin: a cada episódio, sorteia uma pista
    do conjunto de treino e roda UM episódio nela.

    Total de episódios = n_episodios_por_pista * len(pistas_treino).
    Com 16 pistas e 30k por pista, são 480k episódios no total.
    """
    historico_recompensas = []          # uma entrada por episódio (todas as pistas)
    historico_sucessos = []
    rewards_por_pista = {p: [] for p in pistas_treino}

    n_total = n_episodios_por_pista * len(pistas_treino)

    # Cache de ambientes — recriar AmbienteCarro a cada episódio é caro porque
    # o BFS do campo de progresso é recalculado. Mantenha um dict pista→env.
    envs = {p: AmbienteCarro(p, max_steps=max_passos, seed=SEED) for p in pistas_treino}

    for ep in range(n_total):

        if ep < decaimento_eps_episodios:
            frac = ep / decaimento_eps_episodios
            agente.eps = (
                agente.eps_inicial
                - frac *
                (agente.eps_inicial - agente.eps_final)
            )
        else:
            agente.eps = agente.eps_final
        pista = random.choice(pistas_treino)
        env = envs[pista]
        obs = env.reset()
        recompensa_total = 0
        sucesso = False

        while True:
            action = agente.escolher_acao(obs)
            obs_prox, reward, term, trunc, info = env.step(action)
            agente.atualizar(
            obs,
            action,
            reward,
            obs_prox,
            term or trunc
            )
            recompensa_total += reward
            obs = obs_prox
            if term or trunc:
                sucesso = info.get("chegada", False)
                break

        historico_recompensas.append(recompensa_total)
        historico_sucessos.append(int(sucesso))
        rewards_por_pista[pista].append(recompensa_total)

        if verbose and ep % 5000 == 0 and ep > 0:
            taxa = np.mean(historico_sucessos[-5000:])
            print(
                f"Episódio {ep}/{n_total} "
                f"| eps={agente.eps:.3f} "
                f"| sucesso={taxa:.2%}"
            )

    return historico_recompensas, historico_sucessos, rewards_por_pista


# ============================================================================
# AVALIAÇÃO (com ε = 0)
# ============================================================================

def avaliar(env, agente, n_episodios=10):

    resultados = []
    eps_original = agente.eps
    agente.eps = 0.0

    for _ in range(n_episodios):
        obs = env.reset()
        recompensa_total = 0
        velocidades = []
        sucesso = False
        while True:
            action = agente.escolher_acao(obs)
            obs, reward, term, trunc, info = env.step(action)
            recompensa_total += reward
            velocidades.append(env.carro.v)
            if term or trunc:
                sucesso = info.get("chegada", False)
                resultados.append({
                    "n_passos": env.passos,
                    "recompensa_total": recompensa_total,
                    "sucesso": sucesso,
                    "velocidade_media": np.mean(velocidades) if velocidades else 0,
                    "velocidade_maxima": max(velocidades) if velocidades else 0
                })
                break
    agente.eps = eps_original

    melhor = max(
        resultados,
        key=lambda r: r["recompensa_total"]
    )

    return melhor


# ============================================================================
# SALVAR / CARREGAR MODELO (ver enunciado/anexo_b_pickle.md)
# ============================================================================

def treinar_ou_carregar(nome, fn_treinar, recarregar=False):
    """
    Se 'treinamento/{nome}.pkl' existe e recarregar=False, carrega.
    Caso contrário, chama fn_treinar() e salva o resultado.
    """
    arquivo = DIR_TREINAMENTO / f"{nome}.pkl"
    if arquivo.exists() and not recarregar:
        print(f"Carregando {arquivo} ...")
        with open(arquivo, "rb") as f:
            return pickle.load(f)
    else:
        print(f"Treinando {nome} ...")
        resultado = fn_treinar()
        with open(arquivo, "wb") as f:
            pickle.dump(resultado, f)
        print(f"Salvo em {arquivo}")
        return resultado


# ============================================================================
# GERAÇÃO DOS ARQUIVOS DE SAÍDA
# ============================================================================

def escrever_saida(
    caminho,
    nome_algoritmo,
    pista,
    resultado_avaliacao,
    n_episodios_treinados
):

    with open(caminho, "w", encoding="utf-8") as f:
        f.write(
            f"=== Pista: {Path(pista).name} ===\n"
        )
        f.write(
            f"Algoritmo: {nome_algoritmo}\n"
        )
        f.write(
            f"Episódios totais de treinamento: "
            f"{n_episodios_treinados}\n"
        )
        f.write(
            f"Tempo de chegada (passos): "
            f"{resultado_avaliacao['n_passos']}\n"
        )
        f.write(
            f"Velocidade média: "
            f"{resultado_avaliacao['velocidade_media']:.2f}\n"
        )
        f.write(
            f"Velocidade máxima atingida: "
            f"{resultado_avaliacao['velocidade_maxima']:.2f}\n"
        )
        f.write(
            f"Recompensa total: "
            f"{resultado_avaliacao['recompensa_total']:.2f}\n"
        )
        f.write(
            f"Sucesso: "
            f"{'SIM' if resultado_avaliacao['sucesso'] else 'NAO'}\n"
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodios-por-pista", type=int, default=30_000,
                        help="Episódios de treino por pista no round-robin (default: 30000)")
    parser.add_argument("--max-passos", type=int, default=500)
    parser.add_argument("--K", type=int, default=5,
                        help="Baldes da discretização (default: 5; ver README §3.2)")
    parser.add_argument("--recarregar", action="store_true",
                        help="Força re-treino mesmo se o pickle existir")
    parser.add_argument("--avaliar", type=str, default=None,
                        help="Apenas avalia o modelo salvo na pista especificada (pula treino)")
    args = parser.parse_args()

    def fn_treinar():
        agente = AgenteQLearning(
            obs_dim=6,
            n_actions=5,
            K=args.K
        )
        n_total = (
            args.episodios_por_pista
            * len(PISTAS_TREINO)
        )
        rewards, sucessos, rewards_por_pista = (
            treinar_round_robin(
                PISTAS_TREINO,
                agente,
                args.episodios_por_pista,
                args.max_passos,
                int(0.8 * n_total)
            )
        )
        return {
            "q_table": agente.Q,
            "discretization_K": args.K,
            "n_episodes_trained": n_total,
            "rewards_history": rewards,
            "rewards_por_pista": rewards_por_pista,
            "config": {
                "alpha": agente.alpha,
                "gamma": agente.gamma
            }
        }

    modelo = treinar_ou_carregar(
        "qlearning",
        fn_treinar,
        args.recarregar
    )

    agente_avaliacao = (
        AgenteQLearning.from_modelo(modelo)
    )

    pistas_avaliar = (
        [args.avaliar]
        if args.avaliar
        else PISTAS_HOLDOUT
    )
    print(" ")

    print("Modelo carregado")
    print("Estados aprendidos:", len(modelo["q_table"]))
    print(" ")

    for pista in pistas_avaliar:
        print(f"Avaliando {pista}")

        env = AmbienteCarro(
            pista,
            max_steps=args.max_passos,
            seed=SEED
        )

        resultado = avaliar(
            env,
            agente_avaliacao
        )

        print(
        f"{Path(pista).stem:10} | "
        f"Sucesso={resultado['sucesso']} | "
        f"Passos={resultado['n_passos']} | "
        f"Reward={resultado['recompensa_total']:.1f}"
        )
        print(" ")

        nome_pista = Path(pista).stem

        escrever_saida(
            f"q_learning_{nome_pista}.txt",
            "Q-Learning",
            pista,
            resultado,
            modelo["n_episodes_trained"]
        )


if __name__ == "__main__":
    main()
