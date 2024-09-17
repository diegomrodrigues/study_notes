## REFT: Reasoning with Reinforced Fine-Tuning

<imagem: Um diagrama mostrando o fluxo do processo REFT, com um modelo de linguagem sendo inicialmente treinado por SFT (Supervised Fine-Tuning), seguido por uma etapa de exploração usando PPO (Proximal Policy Optimization) para gerar múltiplos caminhos de raciocínio (CoT) e, finalmente, convergindo para uma política otimizada.>

### Introdução

O artigo apresenta uma nova abordagem chamada Reinforced Fine-Tuning (REFT) para aprimorar a capacidade de raciocínio de Grandes Modelos de Linguagem (LLMs) na resolução de problemas matemáticos [1]. REFT surge como uma alternativa ao Supervised Fine-Tuning (SFT) tradicional, que utiliza anotações de Chain-of-Thought (CoT) para treinar modelos. A principal motivação para o desenvolvimento do REFT é superar as limitações do SFT em termos de generalização, especialmente em cenários onde apenas um caminho de raciocínio anotado está disponível para cada questão no conjunto de treinamento [2].

> 💡 **Conceito-chave**: REFT combina SFT com aprendizado por reforço online para explorar múltiplos caminhos de raciocínio, melhorando a generalização do modelo.

### Revisão da Literatura

O artigo se posiciona no contexto de pesquisas recentes sobre resolução de problemas matemáticos usando LLMs. Ele reconhece a importância de abordagens anteriores que utilizam SFT com anotações CoT [3], mas identifica uma lacuna crucial: a limitação de usar apenas um caminho de raciocínio anotado por questão no conjunto de treinamento.

**Contribuições principais da literatura anterior:**

1. Uso de CoT para melhorar o raciocínio em LLMs [4]
2. Aplicação de SFT para treinar modelos em tarefas de raciocínio [5]
3. Desenvolvimento de técnicas de prompting e engenharia de dados para melhorar o desempenho [6]

O REFT se distingue ao introduzir uma abordagem de aprendizado por reforço que permite a exploração de múltiplos caminhos de raciocínio durante o treinamento, potencialmente levando a uma melhor generalização.

### Metodologia

A metodologia do REFT consiste em duas etapas principais:

1. **Warm-up com SFT**: 
   - Inicialização do modelo usando SFT por 1-2 épocas [7]
   - Objetivo: Equipar o modelo com capacidade básica de gerar respostas corretas

2. **Refinamento com Aprendizado por Reforço**:
   - Utilização do algoritmo PPO (Proximal Policy Optimization) [8]
   - Exploração de múltiplos caminhos de raciocínio (CoT) para cada questão
   - Recompensas derivadas naturalmente das respostas corretas

**Modelos Teóricos e Conceituais:**

| Conceito                               | Explicação                                                   |
| -------------------------------------- | ------------------------------------------------------------ |
| **Chain-of-Thought (CoT)**             | Sequência de etapas de raciocínio intermediárias para resolver um problema [9] |
| **Proximal Policy Optimization (PPO)** | Algoritmo de aprendizado por reforço que otimiza a política de geração de respostas [10] |

**Procedimentos Experimentais:**

1. Warm-up do modelo usando SFT por 1-2 épocas
2. Amostragem de múltiplos caminhos CoT usando a política atual
3. Cálculo de recompensas baseadas na correção das respostas
4. Atualização da política usando PPO
5. Repetição dos passos 2-4 até convergência

> ⚠️ **Detalhe Importante**: O REFT não requer um modelo de recompensa separado, diferentemente do RLHF (Reinforcement Learning from Human Feedback) [11].

**Equações e Fórmulas Principais:**

1. Função de perda para SFT:

   $$
   \mathcal{L}_{SFT}(\theta) = -\mathbb{E}_{e\sim\mathcal{D}}\left[\sum_{t=1}^L \log(\pi_\theta(a_t|s_t))\right]
   $$

   Onde:
   - $\theta$: parâmetros do modelo
   - $e$: sequência de tokens da CoT
   - $\mathcal{D}$: conjunto de dados de treinamento
   - $L$: comprimento máximo da sequência
   - $\pi_\theta(a_t|s_t)$: probabilidade de ação $a_t$ dado o estado $s_t$ [12]

2. Função de recompensa para PPO:

   $$
   r(s_t, a_t, s_{t+1}) = \begin{cases}
   1, & \text{EXTRACT}(s_{t+1}) = y \\
   0.1, & \text{EXTRACT}(s_{t+1}) \neq \text{null}, \neq y \\
   0, & \text{EXTRACT}(s_{t+1}) = \text{null}
   \end{cases}
   $$

   Onde:
   - $s_t$: estado atual
   - $a_t$: ação tomada
   - $s_{t+1}$: próximo estado
   - $y$: resposta correta
   - EXTRACT(): função para extrair a resposta do estado [13]

3. Objetivo do PPO:

   $$
   \mathcal{L}_{RL}(\theta, \phi) = \mathcal{L}_{policy} + \alpha\mathcal{L}_{value}
   $$

   Onde:
   - $\mathcal{L}_{policy}$: função de perda da política
   - $\mathcal{L}_{value}$: função de perda do valor
   - $\alpha$: coeficiente de balanceamento [14]

### Resultados

Os experimentos foram conduzidos em três conjuntos de dados: GSM8K, SVAMP e MathQA, utilizando dois modelos base: Galactica-6.7B e CodeLLAMA-7B [15].

**Tabela de Resultados Principais:**

| Método           | GSM8K (N-CoT / P-CoT) | SVAMP (N-CoT / P-CoT) | MathQAMCQ (N-CoT / P-CoT) |
| ---------------- | --------------------- | --------------------- | ------------------------- |
| Galactica + SFT  | 42.68 / 58.83         | 54.50 / 70.09         | 58.07 / 64.61             |
| Galactica + REFT | 48.14 / 68.91         | 61.40 / 74.09         | 58.13 / 70.47             |
| CodeLLAMA + SFT  | 43.59 / 63.68         | 58.09 / 75.40         | 56.01 / 64.79             |
| CodeLLAMA + REFT | 53.30 / 75.28         | 64.50 / 79.19         | 60.13 / 71.83             |

[16]

> ✔️ **Achado Significativo**: REFT consistentemente supera o SFT em quase todas as configurações, com melhorias particularmente notáveis no GSM8K.

**Análises e Interpretações:**

1. REFT demonstra ganhos significativos em relação ao SFT, especialmente em tarefas mais complexas como GSM8K [17].
2. A abordagem baseada em programação (P-CoT) geralmente supera a abordagem baseada em linguagem natural (N-CoT) [18].
3. O desempenho do REFT é ainda mais aprimorado quando combinado com técnicas de inferência como votação por maioria e reordenação por modelo de recompensa [19].

### Proposições, Teoremas e Provas

Embora o artigo não apresente teoremas formais, ele propõe várias hipóteses e princípios importantes:

**Proposição 1: Generalização Aprimorada**

*Enunciado*: O REFT melhora a capacidade de generalização dos modelos de linguagem em tarefas de raciocínio matemático em comparação com o SFT [20].

*Evidência*:
1. Desempenho consistentemente superior em múltiplos conjuntos de dados [21].
2. Capacidade de explorar múltiplos caminhos de raciocínio durante o treinamento [22].

**Proposição 2: Eficiência de Dados**

*Enunciado*: O REFT obtém melhorias de desempenho utilizando o mesmo conjunto de questões de treinamento que o SFT, sem necessidade de dados adicionais ou aumentados [23].

*Evidência*:
1. Resultados comparáveis ou superiores a métodos que utilizam dados aumentados ou gerados por modelos como GPT-3.5 [24].
2. Capacidade de extrair mais informações úteis do mesmo conjunto de dados através da exploração de múltiplos caminhos de raciocínio [25].

> ❗ **Ponto de Atenção**: A eficiência do REFT em termos de dados sugere que há potencial inexplorado nos conjuntos de dados existentes que pode ser aproveitado através de técnicas de aprendizado por reforço.

### Discussão

O REFT apresenta uma abordagem inovadora para melhorar o raciocínio em LLMs, com várias vantagens e algumas limitações:

**Comparações com Trabalhos Anteriores:**

| Aspecto                    | REFT [26] | SFT Tradicional [27] |
| -------------------------- | --------- | -------------------- |
| Exploração de Caminhos     | Múltiplos | Único                |
| Generalização              | Superior  | Limitada             |
| Eficiência de Dados        | Alta      | Moderada             |
| Complexidade Computacional | Maior     | Menor                |

**Limitações e Perspectivas Futuras:**

1. **Eficiência de Treinamento**: O REFT requer mais épocas de treinamento para convergir em comparação com o SFT [28].
   - *Perspectiva*: Investigar técnicas de otimização para acelerar a convergência.

2. **Reward Hacking**: Potencial para exploração de recompensas em cenários com espaço de respostas limitado (e.g., múltipla escolha) [29].
   - *Perspectiva*: Desenvolver funções de recompensa mais sofisticadas ou baseadas em processos.

3. **Escalabilidade**: O artigo foca em modelos de tamanho moderado (6.7B-7B parâmetros) [30].
   - *Perspectiva*: Explorar a aplicabilidade do REFT em modelos maiores e mais complexos.

### Conclusão

O REFT representa um avanço significativo na melhoria da capacidade de raciocínio de LLMs para resolução de problemas matemáticos [31]. Ao combinar SFT com aprendizado por reforço online, o REFT demonstra uma capacidade superior de generalização e eficiência de dados em comparação com abordagens tradicionais [32].

As principais contribuições do REFT incluem:
1. Exploração de múltiplos caminhos de raciocínio durante o treinamento [33].
2. Melhoria significativa no desempenho em benchmarks padrão de resolução de problemas matemáticos [34].
3. Demonstração do potencial do aprendizado por reforço para aprimorar LLMs em tarefas de raciocínio [35].

Futuros desenvolvimentos do REFT podem focar na otimização da eficiência de treinamento, no refinamento das funções de recompensa e na aplicação da técnica a uma gama mais ampla de tarefas de raciocínio além da matemática [36].

### Perguntas Teóricas

1. Como o REFT poderia ser modificado para incorporar feedback humano durante o processo de treinamento, semelhante ao RLHF, mantendo sua eficiência em termos de dados?

2. Considerando a arquitetura do REFT, como você proporia uma extensão do método para lidar com problemas que requerem raciocínio em múltiplas etapas, onde a recompensa intermediária não é facilmente definida?

3. Analise teoricamente como a escolha do coeficiente de divergência KL ($\beta$) no REFT afeta o equilíbrio entre exploração e estabilidade do modelo. Como isso se compara com os trade-offs em outros algoritmos de aprendizado por reforço?

### Perguntas Teóricas Avançadas

1. Desenvolva uma prova teórica para demonstrar sob quais condições o REFT garantidamente converge para uma política ótima, considerando o espaço de busca de caminhos de raciocínio e a função de recompensa proposta no artigo.

2. Proponha uma extensão teórica do REFT que incorpore incerteza epistêmica na geração de caminhos de raciocínio. Como isso afetaria a convergência e a generalização do modelo em problemas com múltiplas soluções válidas?

3. Analise matematicamente o impacto da função de recompensa esparsa do REFT na eficiência do aprendizado. Como você modificaria a função de recompensa para criar um gradiente de aprendizado mais suave, mantendo a fidelidade ao objetivo original?

4. Formule um teorema que relacione a complexidade do espaço de busca de caminhos de raciocínio no REFT com a taxa de convergência do algoritmo. Como isso se compara com os limites teóricos de outros algoritmos de aprendizado por reforço em espaços de ação discretos e de alta dimensionalidade?

5. Desenvolva um framework teórico para analisar a transferência de conhecimento no REFT entre diferentes domínios de problemas matemáticos. Como você quantificaria e maximizaria a transferência de habilidades de raciocínio aprendidas?

### Referências

[1] "We propose a simple yet effective approach called Reinforced Fine-Tuning (ReFT) to enhance the generalizability of learning LLMs for reasoning, with math problem-solving as an example." *(Abstract)*

[2] "Intuitively, it would be better for the algorithm to learn from multiple annotated reasoning paths given a question." *(Abstract)*

[3] "The state-of-the-art approaches to solving math problems (Luo et al., 2023; Wang et al., 2023a) employ Supervised Fine-Tuning (SFT) to train the models using Chain-of-Thought (CoT) annotations (Wei et al., 2022)." *(Introduction)*

[4] "As shown in Figure 1, a CoT annotation outlines the intermediate reasoning steps toward solving a math problem." *(Introduction)*

[5] "Usually there is only CoT annotation for each question in the training data, i.e., one correct reasoning path, which is utilized in SFT." *(Introduction)*

[6] "Recent research efforts focus on CoT prompt design and data engineering." *(Related Work)*

[7] "ReFT commences with a warm-up stage involving Supervised Fine-