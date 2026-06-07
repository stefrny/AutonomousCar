# Q-Learning | Carro Autônomo

---

## 1. Visão geral
  Este projeto implementa um agente de Q-Learning tabular para controlar um carro autônomo em uma pista 2D baseada em grid e tem como principal objetivo aprender, por interação com o ambiente, uma política para conduzir da largada até a chegada evitando colisões.

  Abordagem (RL):
  - Estado contínuo: vetor LIDAR de 6 dimensões
  - Discretização para tabela Q
  - Política ε-greedy
  - Treinamento em round-robin em múltiplas pistas
  - Avaliação em pistas de holdout (17 e 18)

---

## 2. Estrutura de solucao.py

  1. Configuração geral e paths
  2. Implementação do agente (Q-Learning)
  3. Loop de treinamento (round-robin)
  4. Avaliação e geração de relatórios
  5. Serialização do modelo (pickle)
  6. Função main()

---

## 3. Agente Q-Learning (AgenteQLearning)

### 3.1 Representação do estado
  Estado observado pelo agente: obs = [d_0, d_+30, d_-30, d_+60, d_-60, v_norm]
  
  Cada componente está em [0, 1].

- Discretização para Q-learning tabular:
  
  Definição da função:
    ```
    def discretizar(obs):
        return tuple(min(int(v * K), K - 1) for v in obs)
    ```
  Cada variável é dividida em K bins (default: K=3).
  Estado final: tupla de 6 inteiros.


### 3.2 Tabela Q

  - Estrutura da Q-table:
    ```
    self.Q = {
        estado_discreto: np.array([Q(a0), Q(a1), ..., Q(a4)])
    }
    ```
- Inicialização:
  - Estados criados sob demanda
  - Inicialização com valores pequenos aleatórios:
    `np.random.uniform(-0.01, 0.01, n_actions)`

  OBS: n_actions = 5 (assumindo 5 ações possíveis)

### 3.3 Política ε-greedy
- Algoritmo:
  - Se um valor aleatório < ε: ação aleatória
  - Caso contrário: ação = argmax Q(s)
- Propriedades:
  - ε começa alto (exploração)
  - ε decai ao longo do treinamento
  - Converge para exploração quase nula

### 3.4 Atualização Q-Learning
- Equação (atualização padrão):
  - Q(s, a) ← Q(s, a) + α [ r + γ max_{a'} Q(s', a') − Q(s, a) ]
- Casos:
  - Se o episódio termina: alvo = reward
  - Caso contrário: alvo = reward + γ max Q(s', a')
  
---

## 4. Treinamento (Round-Robin)
- Estratégia:
  - O agente treina alternando entre as 16 pistas de treino:
    - pista = random.choice(pistas_treino)
- Motivo:
  - Evita overfitting em uma única pista
  - Evita esquecimento catastrófico
  - Evita viés de sequência
- Estrutura do treinamento:
  - Cada episódio usa uma pista aleatória
  - Ambiente reutilizado via cache (envs)
  - Atualização ocorre a cada step
  - Decaimento de ε:
    - eps = linear_decay(eps_inicial → eps_final)
  - Exploração alta no início, exploração baixa no final

---

## 5. Função de recompensa
- Recompensa vem diretamente do ambiente:
  - Progresso na pista: positivo incremental
  - Penalidade por tempo: -0.1 por passo
  - Colisão: -100 (termina episódio)
  - Chegada: +500

---

## 6. Avaliação
- Configuração:
  - ep = 0.0 (política gulosa)
- Métricas coletadas:
  - número de passos
  - recompensa total
  - sucesso (chegou ao fim)
  - velocidade média
  - velocidade máxima

---

## 7. Serialização (Pickle)
- O modelo salvo contém:
  - {
      "q_table": agente.Q,
      "discretization_K": K,
      "n_episodes_trained": total,
      "rewards_history": ...,
      "config": {
          "alpha": α,
          "gamma": γ
      }
    }
- Arquivo gerado:
  - treinamento/qlearning.pkl

---

## 8. Função main()
- Fluxo geral:
  - Parse de argumentos CLI
  - Treinamento ou carregamento do modelo
  - Construção do agente de avaliação
  - Avaliação nas pistas: 17, 18
  - Geração de arquivos .txt

---

## 9. Resultados obtidos
- Desempenho geral:
  - Alta taxa de sucesso em pistas 01–09
  - Sucesso parcial em pistas médias/difíceis
  - Algumas falhas pontuais (ex.: pista 10, 14, 16)
- Holdout (generalização):
  - pista 17: ✔ sucesso (147 passos, reward 624.3)
  - pista 18: ✔ sucesso (157 passos, reward 629.3)
  
Indica boa generalização do LIDAR local + Q-learning tabular.

---

## 10. Análise crítica do comportamento
- Pontos fortes:
  - Generalização razoável com estado reduzido (LIDAR)
  - Aprendizado eficiente com discretização K=3
  - Round-robin ajuda na estabilidade do aprendizado
- Limitações observadas:
  - Falhas em pistas específicas (10, 14, 16)
  - Sensível à geometria de curvas fechadas
  - Tabela Q ainda relativamente pequena (594 estados aprendidos)
- Interpretação do número de estados:
  - Estados aprendidos: 594
  - Espaço total possível: 5^6 = 15.625
  - visitado: ~3.8% do espaço

---

## 11. Conclusão
  A implementação demonstra que a Q-Learning tabular é suficiente para navegação em ambiente estruturado, LIDAR local permite generalização entre pistas, Round-robin é essencial para estabilidade do aprendizado e Discretização K=3 fornece bom trade-off entre granularidade e generalização


---