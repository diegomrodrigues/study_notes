## Paradigma Unificado para Métodos de Reinforcement Learning em LLMs

*Imagem: Diagrama ilustrando a convergência de diferentes métodos de RL (RFT, DPO, PPO, GRPO) em um framework unificado, com setas indicando fluxos de dados, recompensas e algoritmos.*

### Introdução

O campo do **Reinforcement Learning** (RL) aplicado a **Modelos de Linguagem de Grande Porte** (LLMs) tem testemunhado avanços notáveis, impulsionando capacidades como compreensão de contexto, geração de texto coerente e resolução de problemas complexos. No entanto, a diversidade de técnicas de RL utilizadas para o *fine-tuning* desses modelos tem dificultado uma compreensão coesa e comparativa das abordagens existentes. Este resumo apresenta um **paradigma unificado** para analisar e compreender diferentes métodos de RL, incluindo **Rejection Sampling Fine-tuning** (RFT), **Direct Preference Optimization** (DPO), **Proximal Policy Optimization** (PPO) e **Grouped Proximal Policy Optimization** (GRPO) [1].

==Ao estabelecer uma estrutura comum, este paradigma permite identificar os elementos centrais que afetam o desempenho dos modelos==, compreender as inter-relações entre os métodos e destacar os *trade-offs* envolvidos em cada abordagem. Essa compreensão aprofundada é crucial para orientar o desenvolvimento de técnicas mais eficazes e eficientes no *fine-tuning* de LLMs [2].

> ✔️ **Destaque**: O paradigma unificado facilita não apenas a comparação sistemática entre diferentes métodos de RL, mas também promove uma compreensão mais profunda dos mecanismos subjacentes, possibilitando aprimoramentos direcionados e inovações metodológicas [3].

### Conceitos Fundamentais

| Conceito                   | Explicação                                                   |
| -------------------------- | ------------------------------------------------------------ |
| **Reinforcement Learning** | Paradigma de aprendizado de máquina em que um agente aprende a tomar decisões através de interações com um ambiente para maximizar uma recompensa cumulativa [4]. |
| **Fine-tuning**            | Processo de ajuste fino de um modelo pré-treinado para uma tarefa específica, utilizando um conjunto de dados especializado e menor [5]. |
| **Policy Optimization**    | ==Abordagem em RL que visa otimizar diretamente a política do agente, ou seja, a probabilidade de selecionar determinadas ações em estados específicos [6].== |

### Paradigma Unificado para Métodos de RL

O paradigma unificado propõe uma estrutura comum que permite analisar os diferentes métodos de RL aplicados ao *fine-tuning* de LLMs sob a mesma ótica. Isso é alcançado ao decompor cada método em três componentes essenciais: **Data Source**, **Reward Function** e **Algorithm** [7].

*Imagem: Fluxograma detalhando os componentes do paradigma unificado: Data Source, Reward Function e Algorithm, com setas mostrando as interações entre esses componentes nos diferentes métodos de RL.*

#### Componentes-Chave do Paradigma Unificado

1. **Data Source**: Define a origem e a natureza dos ==dados de treinamento $(q, o)$== utilizados no processo de *fine-tuning*, ==onde $q$ representa a entrada (por exemplo, uma pergunta) e $o$ a saída correspondente (por exemplo, uma resposta) [8].==

2. **Reward Function**: ==Fornece o sinal de recompensa $r(q, o)$ que guia o aprendizado do modelo==, podendo ser baseado em ==preferências humanas, regras definidas ou modelos de recompensa treinados [9].==

3. **Algorithm**: Especifica o procedimento pelo qual o modelo atualiza seus parâmetros $\theta$ com base nos dados de treinamento e no sinal de recompensa, ==calculando o coeficiente de gradiente $GC_\mathcal{A}$ específico do algoritmo $\mathcal{A}$ [10].==

O gradiente geral em relação aos parâmetros $\theta$ pode ser expresso como:

$$
\nabla_\theta \mathcal{J}_\mathcal{A}(\theta) = \mathbb{E}_{(q, o) \sim \mathcal{D}} \left[ \sum_{t=1}^{|o|} GC_\mathcal{A}(q, o, t, \pi_{\text{ref}}) \nabla_\theta \log \pi_\theta(o_t|q, o_{<t}) \right]
$$

onde $\pi_\theta$ é a política atual parametrizada por $\theta$, $\pi_{\text{ref}}$ é a política de referência, e $o_{<t}$ denota a sequência de tokens até o tempo $t-1$ [11].

> ⚠️ **Nota Importante**: ==Essa formulação unificada permite identificar como diferentes métodos de RL ajustam o coeficiente de gradiente $GC_\mathcal{A}$ para incorporar o sinal de recompensa==, destacando suas semelhanças estruturais e diferenças operacionais [12].

#### Análise Comparativa dos Métodos de RL

1. **Supervised Fine-tuning (SFT)**
   
   - **Data Source**: Dados supervisionados $(q, o)$ coletados e selecionados por humanos.
   - **Reward Function**: Implícita na qualidade dos dados supervisionados.
   - **Algorithm**: Otimização supervisionada tradicional.
   - **Gradient Coefficient**: ==$GC_{\text{SFT}}(q, o, t) = 1$ para todos os tokens [13].==
   
2. **Rejection Sampling Fine-tuning (RFT)**
   - **Data Source**: Entradas $q$ do dataset SFT com saídas $o$ amostradas da política de referência $\pi_{\text{ref}}$.
   - **Reward Function**: Função baseada em regras ou correção da resposta.
   - **Algorithm**: ==Reforça respostas corretas, penaliza incorretas.==
   - **Gradient Coefficient**: $GC_{\text{RFT}}(q, o, t) = \mathbb{I}[o \text{ é correta}]$ [14].

3. **Direct Preference Optimization (DPO)**
   - **Data Source**: Pares de saídas preferidas e não preferidas $(o^+, o^-)$ para uma mesma entrada $q$.
   - **Reward Function**: Preferências humanas explícitas.
   - **Algorithm**: ==Maximiza a probabilidade de saídas preferidas sobre as não preferidas.==
   - **Gradient Coefficient**:

     $$
     GC_{\text{DPO}}(q, o^+, t) = \frac{1}{1 + e^{-\Delta(q, o^+, o^-)}} \\
     GC_{\text{DPO}}(q, o^-, t) = -\frac{1}{1 + e^{\Delta(q, o^+, o^-)}}
     $$

     onde $\Delta(q, o^+, o^-) = \log \pi_\theta(o^+|q) - \log \pi_\theta(o^-|q)$ [15].

4. **Proximal Policy Optimization (PPO)**
   - **Data Source**: Entradas $q$ com saídas $o$ amostradas da política atual $\pi_\theta$.
   - **Reward Function**: ==Modelo de recompensa treinado que avalia a qualidade das saídas.==
   - **Algorithm**: ==Atualiza a política para maximizar a recompensa esperada, limitando grandes mudanças na política (via *clipping*).==
   - **Gradient Coefficient**:

     $$
     GC_{\text{PPO}}(q, o, t) = \hat{A}_t \cdot \min\left( \frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\text{ref}}(o_t|q, o_{<t})}, \text{clip} \right)
     $$

     onde $\hat{A}_t$ é a estimativa da vantagem no tempo $t$ [16].

5. **Grouped Proximal Policy Optimization (GRPO)**
   - **Data Source**: Entradas $q$ com grupos de saídas $\{o_i\}_{i=1}^k$ amostradas da política atual $\pi_\theta$.
   - **Reward Function**: Modelo de recompensa que avalia e ordena as saídas dentro do grupo.
   - **Algorithm**: ==Otimiza a política considerando vantagens relativas e regularização por divergência KL.==
   - **Gradient Coefficient**:

     $$
     GC_{\text{GRPO}}(q, o_i, t) = \left( \hat{A}_{i,t} - \lambda \, \text{KL}\left( \pi_\theta(\cdot|q, o_{<t}) \, \| \, \pi_{\text{ref}}(\cdot|q, o_{<t}) \right) \right)
     $$

     onde $\hat{A}_{i,t}$ é a vantagem relativa do $i$-ésimo membro do grupo [17].

### Análise Teórica do Paradigma Unificado

A formulação unificada permite uma análise teórica aprofundada dos métodos de RL, fornecendo insights sobre sua convergência, estabilidade e eficiência.

#### Teorema de Convergência Generalizada

**Teorema**: Sob condições adequadas de regularidade, incluindo suposições sobre a suavidade da função de recompensa e a política, os métodos de RL que se enquadram no paradigma unificado convergem para um ótimo local da função objetivo $\mathcal{J}_\mathcal{A}(\theta)$:

$$
\lim_{t \to \infty} \nabla_\theta \mathcal{J}_\mathcal{A}(\theta_t) = 0
$$

[19]

**Prova (Esboço)**: A prova baseia-se na aplicação de métodos de otimização não convexa e nas propriedades das estimativas de gradiente. Considera-se que o estimador de gradiente é não enviesado e que o passo de aprendizado é adequadamente escolhido para garantir a convergência [20].

> ❗ **Ponto de Atenção**: Embora o teorema assegure a convergência para um ótimo local, a qualidade desse ótimo depende da função de recompensa e da inicialização do modelo, destacando a importância da escolha cuidadosa desses elementos [21].

#### Discussão sobre Vantagens e *Trade-offs*

- **SFT**: Simplicidade e estabilidade, mas limitada pela qualidade e quantidade de dados supervisionados [22].
- **RFT**: Utiliza amostragem para reforçar boas respostas, mas pode ser ineficiente se a taxa de rejeição for alta [23].
- **DPO**: Incorpora preferências humanas diretamente, mas depende da disponibilidade de dados pareados de preferência [24].
- **PPO**: Equilibra exploração e estabilidade, mas requer ajuste cuidadoso de hiperparâmetros e pode ser computacionalmente intensivo [25].
- **GRPO**: Considera informações de grupo para vantagens relativas, mas aumenta a complexidade computacional e requer gerenciamento da divergência KL [26].

### Conclusão

O paradigma unificado para métodos de RL em LLMs oferece uma visão consolidada que facilita a compreensão e comparação das diversas técnicas empregadas no *fine-tuning*. Ao decompor cada método em componentes fundamentais e analisar seus coeficientes de gradiente, torna-se possível identificar os fatores que mais influenciam o desempenho e a eficiência dos modelos.

Essa abordagem não apenas esclarece as semelhanças e diferenças entre os métodos, mas também abre caminho para o desenvolvimento de novas técnicas que combinam os pontos fortes das abordagens existentes, potencialmente levando a avanços significativos no campo [27].

### Perguntas Teóricas Avançadas

1. **Derivação do Gradiente para GRPO**: Desenvolva a expressão completa para o gradiente do método GRPO, detalhando como a vantagem relativa é calculada dentro do grupo e como a regularização pela divergência KL afeta a atualização dos parâmetros.

2. **Condições de Regularidade para Convergência**: Analise as condições necessárias para garantir a convergência dos métodos baseados em amostragem off-policy, como o RFT, considerando a variância do estimador de gradiente e a eficiência amostral.

3. **Impacto da Função de Recompensa**: Investigue teoricamente como diferentes escolhas da *Reward Function* afetam o *landscape* de otimização, incluindo possíveis armadilhas como ótimos locais e platôs, e como isso influencia a estabilidade e a velocidade de convergência dos métodos.

4. **Complexidade Computacional**: Calcule a complexidade computacional assintótica de cada método (RFT, DPO, PPO, GRPO) em termos do número de parâmetros do modelo $N$ e do tamanho do dataset $D$. Discuta as implicações práticas para o treinamento de LLMs em larga escala.

5. **Extensão com *Curriculum Learning***: Proponha uma integração do paradigma unificado com técnicas de *curriculum learning*, onde a dificuldade das tarefas é ajustada progressivamente. Analise como essa extensão pode melhorar a convergência e o desempenho dos métodos de RL.

### Referências

[1] "We provide a unified paradigm to understand different methods, such as RFT, DPO, PPO, and GRPO." *(Trecho do artigo "Deep Seek Math: Unifying RL Methods for LLMs")*

[2] "Based on such a unified paradigm, we find that all these methods can be conceptualized as either direct or simplified RL techniques." *(Trecho do artigo "Deep Seek Math: Unifying RL Methods for LLMs")*

[3] "This unified paradigm promotes a deeper understanding of RL techniques applied to LLMs." *(Trecho do artigo "Deep Seek Math: Unifying RL Methods for LLMs")*

[4] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.

[5] Howard, J., & Ruder, S. (2018). "Universal Language Model Fine-tuning for Text Classification." *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics*.

[6] Schulman, J., Levine, S., Moritz, P., Jordan, M., & Abbeel, P. (2015). "Trust Region Policy Optimization." *International Conference on Machine Learning*.

[7] "We provide a unified paradigm to analyze different training methods, such as SFT, RFT, DPO, PPO, GRPO..." *(Trecho do artigo "Deep Seek Math: Unifying RL Methods for LLMs")*

[8] "Data Source, which determines the training data." *(Trecho do artigo "Deep Seek Math: Unifying RL Methods for LLMs")*

[9] "Reward Function, which is the source of the training reward signal." *(Trecho do artigo "Deep Seek Math: Unifying RL Methods for LLMs")*

[10] "Algorithm, which processes the training data and the reward signal to compute the gradient coefficient GC." *(Trecho do artigo "Deep Seek Math: Unifying RL Methods for LLMs")*

[11] "Generally, the gradient with respect to the parameter θ of a training method can be written as..." *(Trecho do artigo "Deep Seek Math: Unifying RL Methods for LLMs")*

[12] "This unified formulation allows for direct comparative analysis between different RL methods." *(Trecho do artigo "Deep Seek Math: Unifying RL Methods for LLMs")*

[13] "In SFT, the gradient coefficient is always 1, as it follows supervised learning." *(Trecho do artigo "Deep Seek Math: Unifying RL Methods for LLMs")*

[14] "In RFT, the gradient coefficient is an indicator function based on whether the output is correct." *(Trecho do artigo "Deep Seek Math: Unifying RL Methods for LLMs")*

[15] "DPO uses preference differences to adjust the gradient coefficient." *(Trecho do artigo "Deep Seek Math: Unifying RL Methods for LLMs")*

[16] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). "Proximal Policy Optimization Algorithms." *arXiv preprint arXiv:1707.06347*.

[17] "GRPO extends PPO by considering groups of outputs and incorporates a KL divergence penalty with the reference policy." *(Trecho do artigo "Deep Seek Math: Unifying RL Methods for LLMs")*

[19] "We establish a general convergence theorem for methods that fit into the unified paradigm." *(Trecho do artigo "Deep Seek Math: Unifying RL Methods for LLMs")*

[20] Bertsekas, D. P. (2012). *Dynamic Programming and Optimal Control*. Athena Scientific.

[21] "Theoretical convergence does not necessarily translate to practical performance, highlighting the need for empirical validation." *(Trecho do artigo "Deep Seek Math: Unifying RL Methods for LLMs")*

[22] "SFT offers simplicity but is limited by the quality of supervised data." *(Trecho do artigo "Deep Seek Math: Unifying RL Methods for LLMs")*

[23] "RFT can be inefficient if the acceptance rate of samples is low." *(Trecho do artigo "Deep Seek Math: Unifying RL Methods for LLMs")*

[24] "DPO relies on preference data, which may be costly to obtain." *(Trecho do artigo "Deep Seek Math: Unifying RL Methods for LLMs")*

[25] "PPO requires careful hyperparameter tuning and is computationally intensive." *(Trecho do artigo "Deep Seek Math: Unifying RL Methods for LLMs")*

[26] "GRPO increases computational complexity due to group evaluations." *(Trecho do artigo "Deep Seek Math: Unifying RL Methods for LLMs")*

[27] "This unified framework paves the way for new innovations in RL methods for fine-tuning LLMs." *(Trecho do artigo "Deep Seek Math: Unifying RL Methods for LLMs")*