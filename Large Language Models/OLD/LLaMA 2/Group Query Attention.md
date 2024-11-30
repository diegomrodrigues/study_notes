# GQA: Treinamento de Modelos Transformer Multi-Query Generalizados a partir de Checkpoints Multi-Head

````mermaid
graph LR
    %% Diagrama mostrando a transição de MHA para GQA

    subgraph MHA [Atenção Multi-Head]
        direction LR
        InputMHA[Entrada]
        InputMHA --> Q1[Consulta 1]
        InputMHA --> K1[Chave 1]
        InputMHA --> V1[Valor 1]
        InputMHA --> Q2[Consulta 2]
        InputMHA --> K2[Chave 2]
        InputMHA --> V2[Valor 2]
        InputMHA --> Q3[Consulta 3]
        InputMHA --> K3[Chave 3]
        InputMHA --> V3[Valor 3]
        InputMHA --> Q4[Consulta 4]
        InputMHA --> K4[Chave 4]
        InputMHA --> V4[Valor 4]

        Q1 --> Attn1[Atendimento 1]
        K1 --> Attn1
        V1 --> Attn1

        Q2 --> Attn2[Atendimento 2]
        K2 --> Attn2
        V2 --> Attn2

        Q3 --> Attn3[Atendimento 3]
        K3 --> Attn3
        V3 --> Attn3

        Q4 --> Attn4[Atendimento 4]
        K4 --> Attn4
        V4 --> Attn4

        Attn1 --> SaidaMHA[Saída MHA]
        Attn2 --> SaidaMHA
        Attn3 --> SaidaMHA
        Attn4 --> SaidaMHA
    end

    MHA -->|Transição| GQA

    subgraph GQA [Atenção Grouped-Query]
        direction LR
        InputGQA[Entrada]
        InputGQA --> Q1g[Consulta 1]
        InputGQA --> Q2g[Consulta 2]
        InputGQA --> Q3g[Consulta 3]
        InputGQA --> Q4g[Consulta 4]
        InputGQA --> Kg[Chave Agrupada]
        InputGQA --> Vg[Valor Agrupado]

        Q1g --> AttnG1[Atendimento 1]
        Kg --> AttnG1
        Vg --> AttnG1

        Q2g --> AttnG2[Atendimento 2]
        Kg --> AttnG2
        Vg --> AttnG2

        Q3g --> AttnG3[Atendimento 3]
        Kg --> AttnG3
        Vg --> AttnG3

        Q4g --> AttnG4[Atendimento 4]
        Kg --> AttnG4
        Vg --> AttnG4

        AttnG1 --> SaidaGQA[Saída GQA]
        AttnG2 --> SaidaGQA
        AttnG3 --> SaidaGQA
        AttnG4 --> SaidaGQA
    end
````



### Introdução

O artigo "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" apresenta uma inovação significativa no campo do processamento de linguagem natural, focando na otimização de modelos Transformer para inferência rápida [1]. Os autores abordam o ==desafio de equilibrar a qualidade do modelo com a velocidade de inferência==, propondo uma solução que combina as vantagens da ==atenção multi-query (MQA) com a robustez da atenção multi-head (MHA) [2].==

A relevância deste trabalho é destacada pelo crescente uso de modelos de linguagem em larga escala e a necessidade de torná-los mais eficientes para aplicações práticas. Os objetivos principais do artigo são:

1. ==Propor um método para converter modelos MHA existentes em modelos MQA com apenas 5% do custo computacional original de pré-treinamento [3].==
2. ==Introduzir a atenção grouped-query (GQA), uma generalização da MQA que oferece um equilíbrio entre qualidade e velocidade [4].==

> 💡 **Contribuição Chave**: O artigo apresenta uma abordagem inovadora para melhorar a eficiência de inferência dos modelos Transformer sem comprometer significativamente a qualidade.

### Revisão da Literatura

O artigo se posiciona no contexto de pesquisas anteriores sobre otimização de modelos Transformer, particularmente no que diz respeito à redução do overhead de largura de banda de memória durante a inferência. ==Os autores reconhecem o trabalho seminal de Shazeer (2019) na proposição da atenção multi-query [5], que reduziu significativamente o overhead de memória ao usar apenas uma cabeça de chave e valor.==

Trabalhos subsequentes, como Pope et al. (2022) e de Jong et al. (2022), demonstraram a eficácia da MQA, especialmente para entradas longas [6]. O artigo também menciona outras abordagens para reduzir o overhead de largura de banda de memória, incluindo:

- ==Flash attention (Dao et al., 2022)==
- Quantização (Dettmers et al., 2022; Frantar et al., 2022)
- Destilação de modelo (Hinton et al., 2015; Gou et al., 2021)
- Atenção cruzada esparsa em camadas (de Jong et al., 2022)
- Amostragem especulativa (Chen et al., 2023; Leviathan et al., 2022) [7]

A contribuição única deste artigo está na proposta de uma abordagem intermediária entre MHA e MQA, bem como um método eficiente para converter modelos existentes.

### Metodologia

#### Modelos Teóricos e Conceituais:

| Conceito                        | Explicação                                                   |
| ------------------------------- | ------------------------------------------------------------ |
| **Atenção Multi-Head (MHA)**    | Método padrão em modelos Transformer, onde múltiplas cabeças de atenção processam consultas, chaves e valores independentemente [8]. |
| **Atenção Multi-Query (MQA)**   | Variante que usa ==múltiplas cabeças de consulta, mas apenas uma cabeça de chave e valor, reduzindo o overhead de memória [9].== |
| **Atenção Grouped-Query (GQA)** | Generalização proposta que ==divide as cabeças de consulta em grupos, cada um compartilhando uma única cabeça de chave e valor [10].== |

#### Procedimentos Experimentais:

1. **Conversão de Checkpoint:**

   - **Agrupamento das Projeções de Chave e Valor:**

     ==As matrizes de projeção de chave e valor das cabeças são agrupadas usando média (mean pooling) para criar matrizes de projeção compartilhadas [16]:==
     $$
     W_g^K = \frac{1}{|C_g|} \sum_{i \in C_g} W_i^K
     $$

     $$
     W_g^V = \frac{1}{|C_g|} \sum_{i \in C_g} W_i^V
     $$

     Onde $C_g$ é o conjunto de cabeças no grupo $g$.

2. **Uptraining:**

   - **Treinamento Adicional:**

     ==Após a conversão, o modelo é treinado adicionalmente por uma fração $\alpha$ dos passos de pré-treinamento original [17].==

     Isso permite que o modelo ==ajuste os pesos para a nova estrutura de atenção, recuperando o desempenho.==

3. **Configurações Experimentais:**

   - **Arquitetura Base:**

     Todos os modelos são baseados na arquitetura T5.1.1 [18].

   - **Hiperparâmetros:**

     O otimizador Adafactor é utilizado com os mesmos hiperparâmetros e agendamento de taxa de aprendizado do T5 original [19].

   - **Aplicação de GQA:**

     MQA e GQA são aplicados apenas à auto-atenção do decodificador e à atenção cruzada, não à auto-atenção do codificador [20].

> ⚠️ **Detalhe Importante**: MQA e GQA são aplicados apenas à auto-atenção do decodificador e à atenção cruzada, não à auto-atenção do codificador [18].

#### Equações e Fórmulas Principais:

![image-20240917133412325](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240917133412325.png)

A atenção grouped-query pode ser representada matematicamente como:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Onde:
- $Q$ representa as matrizes de consulta (múltiplas cabeças)
- $K$ e $V$ representam as matrizes de chave e valor (número reduzido de cabeças)
- $d_k$ é a dimensão das chaves

Para GQA com $G$ grupos, temos:

$$
Q \in \mathbb{R}^{n \times (G \times h_q) \times d_k}, K, V \in \mathbb{R}^{n \times G \times d_k}
$$

Onde:
- $n$ é o tamanho da sequência
- $h_q$ é o número de cabeças de consulta por grupo
- $G$ é o número de grupos [19]

##### Atenção Grouped-Query:

Para $G$ grupos, cada com $h_q$ cabeças de consulta, temos:

- Número total de cabeças de consulta: $H = G \times h_q$

- Projeções:

  $$
  Q_{g,i} = QW_{g,i}^Q, \quad K_g = KW_g^K, \quad V_g = VW_g^V
  $$

  Onde $i = 1, \dots, h_q$, e $g = 1, \dots, G$.

- Atenção para cada cabeça no grupo $g$:

  $$
  \text{head}_{g,i} = \text{Attention}(Q_{g,i}, K_g, V_g)
  $$

- Saída GQA:

  $$
  \text{GQA}(Q, K, V) = \text{Concat}(\text{head}_{1,1}, \dots, \text{head}_{G,h_q})W^O
  $$

### Resultados

Os autores apresentam resultados comparativos entre modelos MHA, MQA e GQA em várias tarefas de processamento de linguagem natural:

#### Tabela de Desempenho:

| Modelo    | T<sub>infer</sub> (s) | Média | CNN/DM R<sub>1</sub> | arXiv R<sub>1</sub> | PubMed R<sub>1</sub> | MediaSum R<sub>1</sub> | MultiNews R<sub>1</sub> | WMT BLEU | TriviaQA F1 |
| --------- | --------------------- | ----- | -------------------- | ------------------- | -------------------- | ---------------------- | ----------------------- | -------- | ----------- |
| MHA-Large | 0.37                  | 46.0  | 42.9                 | 44.6                | 46.2                 | 35.5                   | 46.6                    | 27.7     | 78.2        |
| MHA-XXL   | 1.51                  | 47.2  | 43.8                 | 45.6                | 47.5                 | 36.4                   | 46.9                    | 28.4     | 81.9        |
| MQA-XXL   | 0.24                  | 46.6  | 43.0                 | 45.0                | 46.9                 | 36.1                   | 46.5                    | 28.5     | 81.3        |
| GQA-8-XXL | 0.28                  | 47.1  | 43.5                 | 45.4                | 47.7                 | 36.3                   | 47.2                    | 28.4     | 81.6        |

[20]

#### Análises e Interpretações:

> ✔️ **Achado Significativo**: GQA-8-XXL alcança desempenho próximo ao MHA-XXL com tempo de inferência significativamente menor, oferecendo um equilíbrio ótimo entre qualidade e velocidade [21].

1. MQA-XXL apresenta uma redução substancial no tempo de inferência (0.24s) em comparação com MHA-XXL (1.51s), mantendo um desempenho competitivo [22].
2. GQA-8-XXL oferece um compromisso intermediário, com tempo de inferência ligeiramente superior ao MQA-XXL (0.28s), mas com desempenho mais próximo ao MHA-XXL [23].
3. Em tarefas específicas como PubMed e MultiNews, GQA-8-XXL supera até mesmo o MHA-XXL, demonstrando sua eficácia em certos domínios [24].

### Proposições, Teoremas e Provas

Embora o artigo não apresente teoremas formais, ele propõe conceitos importantes que podem ser formulados como proposições:

**Proposição 1: Eficácia do Uptraining**

*Enunciado:* O uptraining de modelos MHA para MQA ou GQA com apenas 5% dos passos originais de pré-treinamento é suficiente para recuperar a maior parte do desempenho do modelo original [25].

*Prova (Empírica):*
1. Os autores converteram checkpoints MHA para MQA e GQA.
2. Realizaram uptraining por α=0.05 (5%) dos passos originais de pré-treinamento.
3. Avaliaram o desempenho em várias tarefas de NLP.
4. Os resultados mostram que o desempenho dos modelos uptrainados se aproxima significativamente dos modelos MHA originais [26].

> ❗ **Ponto de Atenção:** A eficácia do uptraining pode variar dependendo da tarefa e do tamanho do modelo, sendo necessária uma análise cuidadosa para cada aplicação específica.

**Proposição 2: Vantagem da Atenção Grouped-Query**

*Enunciado:* A atenção grouped-query (GQA) oferece um compromisso superior entre qualidade do modelo e velocidade de inferência em comparação com MHA e MQA puras [27].

*Prova (Empírica e Teórica):*
1. Teoricamente, GQA reduz o overhead de memória em comparação com MHA, mas mantém mais capacidade que MQA.
2. Empiricamente, GQA-8-XXL alcança desempenho médio de 47.1, próximo ao MHA-XXL (47.2), com tempo de inferência de 0.28s, significativamente menor que MHA-XXL (1.51s).
3. GQA supera MQA em qualidade (47.1 vs 46.6) com um aumento mínimo no tempo de inferência (0.28s vs 0.24s) [28].

### Discussão

#### Comparações com Trabalhos Anteriores:

| Aspecto       | Este Artigo (GQA) [29]                           | MQA (Shazeer, 2019) [30]               |
| ------------- | ------------------------------------------------ | -------------------------------------- |
| Método        | Grupos intermediários de cabeças de chave/valor  | Uma única cabeça de chave/valor        |
| Qualidade     | Próxima à MHA                                    | Degradação em relação à MHA            |
| Velocidade    | Ligeiramente menor que MQA pura                  | Máxima redução de overhead             |
| Flexibilidade | Permite ajuste fino entre qualidade e velocidade | Fixo na configuração de menor overhead |

#### Limitações e Perspectivas Futuras:

1. **Limitação de Avaliação:** O artigo reconhece que as métricas utilizadas (como ROUGE) podem não capturar completamente a qualidade das saídas, especialmente para sequências longas [31].

2. **Generalização:** Os experimentos focam principalmente em modelos encoder-decoder. É necessário investigar o impacto de GQA em arquiteturas decoder-only, que são cada vez mais populares [32].

3. **Otimização de Hiperparâmetros:** O número ótimo de grupos GQA pode variar dependendo do tamanho do modelo e da tarefa. Pesquisas futuras poderiam explorar métodos para determinar automaticamente a configuração ideal [33].

### Conclusão

O artigo apresenta uma contribuição significativa para a otimização de modelos Transformer, introduzindo a atenção grouped-query (GQA) e um método eficiente de uptraining. As principais conclusões são:

1. GQA oferece um equilíbrio superior entre qualidade e velocidade de inferência em comparação com MHA e MQA puras [34].
2. O método de uptraining proposto permite a conversão eficiente de modelos MHA existentes para GQA com apenas 5% do custo computacional original [35].
3. A abordagem proposta é particularmente promissora para modelos de grande escala, onde o overhead de memória é um gargalo significativo [36].

Futuros caminhos de pesquisa incluem a aplicação de GQA a arquiteturas decoder-only, otimização automática do número de grupos, e investigação de seus benefícios em tarefas além do processamento de linguagem natural [37].

### Perguntas Teóricas

1. Derive a complexidade computacional e de memória da atenção grouped-query em função do número de grupos G, número de cabeças de consulta H, e tamanho da sequência N. Compare com MHA e MQA.

2. Analise teoricamente o impacto da redução no número de cabeças de chave e valor na capacidade representacional do modelo. Como isso afeta a habilidade do modelo de capturar diferentes tipos de dependências nos dados?

3. Proponha um método teórico para determinar o número ótimo de grupos GQA dado um tamanho de modelo e restrições de latência específicas.

### Perguntas Teóricas Avançadas

1. Desenvolva uma prova formal demonstrando que, sob certas condições, GQA com um número apropriado de grupos pode aproximar arbitrariamente bem o desempenho de MHA. Quais seriam essas condições?

2. Analise o impacto teórico de GQA na estabilidade numérica e na convergência durante o treinamento. Como a redução no número de cabeças de chave e valor afeta a propagação do gradiente?

3. Proponha uma extensão teórica de GQA que permita um número variável de grupos por camada. Derive as equações para o cálculo da atenção neste cenário e discuta os potenciais benefícios e desafios desta abordagem.

4. Desenvolva um modelo teórico para prever o desempenho de GQA em função do tamanho do modelo, número de grupos, e características da tarefa. Como este modelo poderia ser usado para otimizar a arquitetura do Transformer para uma dada aplicação?

5. Analise as implicações teóricas de GQA para a interpretabilidade do modelo. Como a estrutura de grupos afeta nossa capacidade de compreender e visualizar os padrões de atenção aprendidos pelo modelo?

### Referências

[1] "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (Título do Artigo)

[2] "Multi-query attention (MQA), which only uses a single key-value head, drastically speeds up decoder inference. However, MQA can lead to quality degradation" (Resumo)

[3] "We (1) propose a recipe for uptraining existing multi-head language model checkpoints into models with MQA using 5% of original pre-training compute"