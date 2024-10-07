# Embeddings Estáticos: Representações Vetoriais Fixas de Palavras

<imagem: Uma visualização 2D de embeddings de palavras, mostrando clusters de palavras semanticamente relacionadas em diferentes regiões do espaço vetorial>

### Introdução

Os embeddings estáticos representam um avanço significativo na representação computacional do significado das palavras, constituindo um pilar fundamental no processamento de linguagem natural moderno. Esses embeddings são vetores densos de baixa dimensionalidade que capturam aspectos semânticos e sintáticos das palavras, permitindo operações matemáticas que refletem relações linguísticas [1]. Diferentemente de representações esparsas anteriores, como one-hot encoding ou tf-idf, os embeddings estáticos oferecem uma representação mais compacta e rica em informações, onde a similaridade entre palavras pode ser quantificada através de operações vetoriais simples [2].

O conceito de embeddings estáticos surgiu da convergência de ideias em linguística, psicologia e ciência da computação nos anos 1950, culminando em modelos computacionais que representam palavras como pontos em um espaço semântico multidimensional [3]. Esta abordagem se baseia na hipótese distribucional, que postula que palavras que ocorrem em contextos similares tendem a ter significados similares [4].

> ⚠️ **Nota Importante**: Os embeddings estáticos são chamados assim porque, uma vez treinados, permanecem fixos, atribuindo o mesmo vetor a uma palavra independentemente do contexto em que ela aparece [5].

### Conceitos Fundamentais

| Conceito                     | Explicação                                                   |
| ---------------------------- | ------------------------------------------------------------ |
| **Vetor Semântico**          | Representação de uma palavra como um ponto em um espaço multidimensional, geralmente com 50 a 300 dimensões. Cada dimensão potencialmente captura um aspecto semântico ou sintático da palavra [6]. |
| **Similaridade Cossenoidal** | Medida padrão de similaridade entre embeddings, calculada como o cosseno do ângulo entre dois vetores. Valores próximos a 1 indicam alta similaridade [7]. |
| **Janela de Contexto**       | Número de palavras ao redor de uma palavra-alvo consideradas durante o treinamento do embedding. Influencia o tipo de relações semânticas capturadas [8]. |

> ❗ **Ponto de Atenção**: A escolha da dimensionalidade e do tamanho da janela de contexto são hiperparâmetros críticos que afetam significativamente a qualidade e as propriedades dos embeddings resultantes [9].

### Modelos de Embeddings Estáticos

<imagem: Diagrama comparativo mostrando a arquitetura neural do Word2Vec (Skip-gram e CBOW) e GloVe>

#### Word2Vec

O Word2Vec, introduzido por Mikolov et al. (2013), é um dos modelos mais influentes para a criação de embeddings estáticos [10]. Ele utiliza redes neurais rasas para aprender representações vetoriais de palavras a partir de grandes corpora de texto. O Word2Vec possui duas variantes principais:

1. **Skip-gram**: Prediz palavras de contexto dado uma palavra-alvo.
2. **Continuous Bag of Words (CBOW)**: Prediz uma palavra-alvo dados seus contextos.

A função objetivo do Skip-gram com amostragem negativa (SGNS) é definida como:

$$
\mathcal{L} = -\left[\log \sigma(c_{pos} \cdot w) + \sum_{i=1}^{k} \log \sigma(-c_{neg_i} \cdot w)\right]
$$

Onde:
- $\sigma$ é a função sigmoide
- $c_{pos}$ é o embedding do contexto positivo
- $w$ é o embedding da palavra-alvo
- $c_{neg_i}$ são os embeddings dos contextos negativos amostrados
- $k$ é o número de amostras negativas

O treinamento envolve a otimização desta função através de descida de gradiente estocástica [11].

#### GloVe (Global Vectors)

O GloVe, proposto por Pennington et al. (2014), é outro modelo popular de embedding estático que combina as vantagens de modelos baseados em contagem (como LSA) e predição (como Word2Vec) [12]. O GloVe se baseia na ideia de que as razões das probabilidades de co-ocorrência de palavras carregam informações semânticas significativas.

A função objetivo do GloVe é:

$$
J = \sum_{i,j=1}^{V} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2
$$

Onde:
- $X_{ij}$ é a contagem de co-ocorrência das palavras i e j
- $w_i$ e $\tilde{w}_j$ são vetores de palavra e contexto, respectivamente
- $b_i$ e $\tilde{b}_j$ são termos de viés
- $f(X_{ij})$ é uma função de ponderação para lidar com palavras raras

> ✔️ **Destaque**: Tanto Word2Vec quanto GloVe produzem embeddings que exibem propriedades algébricas interessantes, como a capacidade de capturar relações analógicas (e.g., "rei" - "homem" + "mulher" ≈ "rainha") [13].

#### Perguntas Teóricas

1. Derive a atualização do gradiente para os vetores de palavra e contexto no modelo Skip-gram com amostragem negativa. Como essa formulação difere da formulação original do Word2Vec sem amostragem negativa?

2. Analise teoricamente como a escolha do número de dimensões dos embeddings afeta o trade-off entre a capacidade de capturar informações semânticas e o overfitting. Como isso se relaciona com o conceito de "maldição da dimensionalidade"?

3. Demonstre matematicamente por que a similaridade cossenoidal é preferível à distância euclidiana para medir a similaridade entre embeddings de palavras. Quais propriedades da similaridade cossenoidal a tornam mais adequada para este propósito?

### Propriedades e Limitações dos Embeddings Estáticos

#### Propriedades Semânticas

Os embeddings estáticos são capazes de capturar uma variedade de relações semânticas e sintáticas entre palavras [14]:

1. **Similaridade**: Palavras com significados similares tendem a ter embeddings próximos no espaço vetorial.
2. **Analogia**: Relações semânticas podem ser modeladas através de operações vetoriais (e.g., "Berlim" - "Alemanha" + "França" ≈ "Paris").
3. **Clustering**: Palavras relacionadas formam clusters naturais no espaço de embeddings.

> 💡 **Insight**: A capacidade dos embeddings de capturar relações analógicas sugere que eles codificam informações sobre as relações semânticas em suas estruturas geométricas [15].

#### Limitações

1. **Polissemia**: Embeddings estáticos atribuem um único vetor para cada palavra, não capturando diferentes sentidos de palavras polissêmicas [16].
2. **Dependência de Contexto**: O significado de uma palavra pode variar significativamente dependendo do contexto, algo que embeddings estáticos não podem capturar [17].
3. **Viés**: Embeddings treinados em dados do mundo real podem perpetuar e amplificar vieses sociais presentes nos dados de treinamento [18].

#### [Nova Seção Adicional: Provas e Demonstrações]

Demonstração da propriedade de analogia em embeddings:

Seja $a:b::c:d$ uma analogia (e.g., "homem:rei::mulher:rainha"). A propriedade de analogia em embeddings postula que:

$$
\vec{b} - \vec{a} \approx \vec{d} - \vec{c}
$$

Prova:
1. Assuma que a relação $a:b$ é representada por um vetor $\vec{r} = \vec{b} - \vec{a}$
2. Se a mesma relação se aplica a $c:d$, então $\vec{d} - \vec{c} \approx \vec{r}$
3. Portanto, $\vec{d} \approx \vec{c} + \vec{r} = \vec{c} + (\vec{b} - \vec{a})$
4. Rearranjando, obtemos: $\vec{d} \approx \vec{b} - \vec{a} + \vec{c}$

Esta demonstração fundamenta o método do paralelogramo para resolver problemas de analogia em espaços de embeddings [19].

#### Perguntas Teóricas

1. Considerando a limitação dos embeddings estáticos em relação à polissemia, proponha e analise matematicamente uma extensão do modelo Word2Vec que poderia abordar esse problema. Como essa extensão afetaria a complexidade computacional do treinamento e da inferência?

2. Desenvolva uma prova formal demonstrando que, sob certas condições ideais, a matriz de co-ocorrência fatorada pelo GloVe é equivalente à matriz de embeddings produzida pelo Word2Vec. Quais são as implicações teóricas dessa equivalência?

3. Analise teoricamente como o fenômeno de "hubness" (tendência de certos pontos se tornarem vizinhos mais próximos de muitos outros pontos em espaços de alta dimensão) afeta a qualidade dos embeddings estáticos. Como esse fenômeno se relaciona com a escolha da dimensionalidade dos embeddings?

### Aplicações e Avaliação de Embeddings Estáticos

#### Aplicações

1. **Classificação de Texto**: Embeddings são usados como features de entrada para modelos de classificação [20].
2. **Sistemas de Recomendação**: Representação de itens e usuários para cálculo de similaridade [21].
3. **Análise de Sentimento**: Captura de nuances semânticas para melhorar a detecção de sentimento [22].
4. **Tradução Automática**: Como inicialização para modelos de tradução neural [23].

#### Métodos de Avaliação

1. **Similaridade de Palavras**: Correlação entre similaridade cossenoidal de embeddings e julgamentos humanos de similaridade (e.g., WordSim-353, SimLex-999) [24].
2. **Analogias**: Precisão em tarefas de analogia (e.g., "homem está para rei assim como mulher está para ?") [25].
3. **Tarefas Downstream**: Desempenho em aplicações práticas como classificação de texto ou NER [26].

> ⚠️ **Nota Importante**: A avaliação intrínseca (similaridade, analogias) nem sempre se correlaciona com o desempenho em tarefas downstream, destacando a importância de avaliações específicas de tarefas [27].

#### [Nova Seção Adicional: Discussão Crítica]

Apesar do sucesso dos embeddings estáticos, várias questões críticas emergem:

1. **Interpretabilidade**: As dimensões dos embeddings não têm interpretação semântica clara, dificultando a análise linguística [28].
2. **Estabilidade**: Pequenas mudanças nos dados de treinamento podem levar a grandes mudanças nos embeddings, questionando sua robustez [29].
3. **Viés e Ética**: A amplificação de vieses sociais em embeddings levanta questões éticas sobre seu uso em sistemas de tomada de decisão [30].

Futuros desenvolvimentos podem focar em:
- Embeddings interpretáveis com dimensões semanticamente significativas
- Métodos robustos de debiasing que preservam informações linguísticas úteis
- Integração de conhecimento simbólico para melhorar a qualidade semântica dos embeddings

### Conclusão

Os embeddings estáticos representam um avanço fundamental na representação computacional do significado das palavras, oferecendo uma ponte entre a linguística teórica e aplicações práticas de NLP [31]. Sua capacidade de capturar relações semânticas complexas em um formato computacionalmente eficiente os tornou uma ferramenta indispensável em diversas aplicações de processamento de linguagem natural [32].

No entanto, as limitações dos embeddings estáticos, como a incapacidade de lidar com polissemia e a falta de sensibilidade ao contexto, abriram caminho para desenvolvimentos mais recentes em representações contextuais dinâmicas, como BERT e GPT [33]. Apesar disso, os princípios fundamentais estabelecidos pelos modelos de embeddings estáticos continuam a influenciar o design de arquiteturas mais avançadas de NLP [34].

A pesquisa contínua nesta área promete não apenas melhorar nossas ferramentas computacionais, mas também aprofundar nossa compreensão teórica da semântica linguística e da representação do conhecimento [35].

### Perguntas Teóricas Avançadas

1. Desenvolva uma prova matemática mostrando que, sob certas condições, a matriz de embeddings aprendida pelo Word2Vec (Skip-gram) converge para uma fatoração da matriz de Informação Mútua Pontual (PMI) entre palavras e contextos. Quais são as implicações teóricas desta equivalência para nossa compreensão dos embeddings de palavras?

2. Analise teoricamente o impacto da dimensionalidade dos embeddings na capacidade do modelo de capturar relações semânticas. Derive uma expressão que relacione o número de dimensões, o tamanho do vocabulário e a quantidade de informação semântica preservada. Como essa relação se compara com os limites teóricos da compressão de informação?

3. Proponha e analise matematicamente um método para combinar embeddings estáticos com informações contextuais dinâmicas. Como esse método poderia superar as limitações dos embeddings estáticos em relação à polissemia e à sensibilidade ao contexto? Quais seriam os trade-offs computacionais e de desempenho?

4. Demonstre formalmente como o problema de "hubness" em espaços de alta dimensão afeta a distribuição de similaridades cossenoidais entre embeddings de palavras. Como esse fenômeno influencia a confiabilidade de tarefas baseadas em vizinhança, como busca de palavras similares? Proponha e analise teoricamente uma métrica de similaridade alternativa que poderia mitigar esse problema.

5. Desenvolva um framework teórico para quantificar e mitigar o viés em embeddings de palavras. Como podemos formalizar matematicamente o conceito de "viés" em um espaço vetorial de embeddings? Analise as implicações éticas e práticas de diferentes abordagens de debiasing, considerando o trade-off entre redução de viés e preservação de informação semântica útil.

### Referências

[1] "Vector semantics is the standard way to represent word meaning in NLP, helping us model many of the aspects of word meaning we saw in the previous section." *(Trecho de Vector Semantics and Embeddings.pdf.md)*

[2] "The idea of vector semantics is to represent a word as a point in a multidimensional semantic space that is derived (in ways we'll see) from the distributions of word neighbors." *(Trecho de Vector Semantics and Embeddings.pdf.md)*

[3] "The idea that meaning is related to the distribution of words in context was widespread in linguistic theory of the 1950s, among distributionalists like Zellig Harris, Martin Joos, and J. R. Firth, and semioticians like Thomas Sebeok." *(Trecho de Vector Semantics and Embeddings.pdf.md)*

[4] "The