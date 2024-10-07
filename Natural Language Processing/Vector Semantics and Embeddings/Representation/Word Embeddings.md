# Word Embeddings: Representação de Palavras em Espaços Semânticos Multidimensionais

<imagem: Uma visualização tridimensional de um espaço vetorial semântico, mostrando palavras como pontos coloridos em diferentes clusters, com setas indicando relações entre palavras semanticamente próximas.>

## Introdução

Os **word embeddings** representam uma revolução na forma como processamos e entendemos a linguagem natural computacionalmente. ==Essas representações vetoriais densas de palavras em espaços semânticos multidimensionais são fundamentais para muitas aplicações modernas de **Processamento de Linguagem Natural (NLP)** e aprendizado de máquina [1]==. O conceito central de **semântica vetorial** baseia-se na ideia de que ==o significado de uma palavra pode ser capturado por sua distribuição em um corpus de texto==, uma noção que tem raízes na **linguística distribucional** dos anos 1950 [2].

> ✔️ **Destaque**: A semântica vetorial modela cada palavra como um vetor — um ponto em um espaço de alta dimensão, também chamado de **embedding** [3].

Essa abordagem permite que modelos computacionais capturem nuances semânticas e relações entre palavras de forma muito mais eficaz do que métodos tradicionais baseados em dicionários ou regras explícitas. Ao transformar palavras em vetores numéricos, torna-se possível aplicar técnicas matemáticas e algorítmicas avançadas para analisar a linguagem natural.

## Conceitos Fundamentais

| Conceito                        | Explicação                                                   |
| ------------------------------- | ------------------------------------------------------------ |
| **Embedding**                   | Representação de uma palavra como um vetor denso em um espaço multidimensional. Tipicamente, esses vetores têm entre 50 e 1000 dimensões [4]. |
| **Espaço Semântico**            | O espaço multidimensional no qual as palavras são representadas como pontos. A proximidade nesse espaço indica similaridade semântica [5]. |
| **Vetores Esparsos vs. Densos** | ==Vetores esparsos têm muitos zeros e poucas dimensões não nulas, enquanto vetores densos têm valores em todas as dimensões==. Embeddings modernos são geralmente densos [6]. |

### Hipótese Distribucional

A base teórica para os word embeddings é a **hipótese distribucional**, formulada por linguistas como Joos (1950), Harris (1954) e Firth (1957) [7]. Essa hipótese afirma que:

> ❗ **Ponto de Atenção**: *"Você conhecerá uma palavra pela companhia que ela mantém."* — J.R. Firth [8].

Em termos práticos, isso significa que palavras que ocorrem em contextos similares tendem a ter significados similares. Essa intuição é formalizada matematicamente nos modelos de embedding, onde a similaridade entre vetores de palavras (geralmente medida pelo cosseno do ângulo entre eles) corresponde à similaridade semântica entre as palavras.

### Motivação para Word Embeddings

Tradicionalmente, a representação de palavras em NLP era feita através de vetores de características esparsos, como o **one-hot encoding**, onde cada palavra é representada por um vetor binário com dimensão igual ao tamanho do vocabulário. Essa abordagem tem várias limitações:

- **Alta Dimensionalidade**: O tamanho do vocabulário pode ser muito grande, resultando em vetores de alta dimensionalidade, o que é computacionalmente ineficiente.
- **Esparsidade**: A maioria dos valores nos vetores é zero, dificultando a captura de relações semânticas entre palavras.
- **Falta de Generalização**: Não captura similaridades entre palavras diferentes.

Os word embeddings superam essas limitações ao representar palavras em vetores densos de dimensões mais baixas, onde a posição de uma palavra no espaço vetorial reflete seu significado semântico.

### Tipos de Modelos de Embedding

1. **Modelos Baseados em Contagem**:
   - Utilizam estatísticas de coocorrência de palavras em um corpus.
   - Exemplos incluem **tf-idf** e **PPMI** (Positive Pointwise Mutual Information) [9].
   - **Vantagens**: Intuitivos, capturam informações globais do corpus.
   - **Desvantagens**: Resultam em vetores esparsos e de alta dimensionalidade, menos eficientes.

2. **Modelos Preditivos**:
   - Aprendem embeddings através de tarefas de previsão em redes neurais.
   - Exemplos incluem **Word2Vec**, **GloVe** e **FastText** [10].
   - **Vantagens**: Produzem vetores densos e capturam relações semânticas complexas.
   - **Desvantagens**: Requerem mais recursos computacionais para treinamento.

## Matriz Termo-Documento e Termo-Termo

==As representações vetoriais de palavras são frequentemente baseadas em dois tipos principais de matrizes que refletem diferentes abordagens para capturar informações semânticas.==

### Matriz Termo-Documento

$$
M_{TD} = \begin{bmatrix}
w_{11} & w_{12} & \cdots & w_{1D} \\
w_{21} & w_{22} & \cdots & w_{2D} \\
\vdots & \vdots & \ddots & \vdots \\
w_{V1} & w_{V2} & \cdots & w_{VD}
\end{bmatrix}
$$

Onde:

- $w_{ij}$ é o peso da palavra $i$ no documento $j$, que pode ser a frequência ou uma medida ponderada.
- $V$ é o tamanho do vocabulário.
- $D$ é o número de documentos.

Esta matriz captura a frequência (ou uma função ponderada dela) de cada palavra em cada documento do corpus [11]. É amplamente utilizada em tarefas de recuperação de informação.

### Matriz Termo-Termo

$$
M_{TT} = \begin{bmatrix}
c_{11} & c_{12} & \cdots & c_{1V} \\
c_{21} & c_{22} & \cdots & c_{2V} \\
\vdots & \vdots & \ddots & \vdots \\
c_{V1} & c_{V2} & \cdots & c_{VV}
\end{bmatrix}
$$

Onde:

- ==$c_{ij}$ é uma medida de coocorrência entre as palavras $i$ e $j$, como a frequência com que aparecem próximas no texto.==
- $V$ é o tamanho do vocabulário.

Esta matriz captura as relações entre palavras baseadas em suas coocorrências em contextos similares [12]. É fundamental para modelos baseados em contagem.

> 💡 **Insight**: A matriz termo-termo permite capturar relações semânticas mais refinadas entre palavras, incluindo similaridades de segunda ordem e estruturas latentes no corpus [13].

### Perguntas Teóricas

1. **Derivação Matemática da Similaridade do Cosseno**:
   - Derive como a similaridade do cosseno entre dois vetores de palavras na matriz termo-documento se relaciona com a frequência relativa dessas palavras nos mesmos documentos.
   - *Dica*: Considere a normalização dos vetores de frequência e como o produto escalar reflete a coocorrência.

2. **Autovetores e Espaço de Embeddings**:
   - Prove que, para uma matriz termo-termo simétrica normalizada, os autovetores da matriz fornecem uma base ortogonal para o espaço de embeddings.
   - *Dica*: Utilize a decomposição espectral e relacione com técnicas de redução de dimensionalidade como **SVD** (Singular Value Decomposition).

3. **Medidas de Associação Robusta**:
   - Considerando a hipótese distribucional, proponha e justifique matematicamente uma medida de associação entre palavras que seja mais robusta a variações na frequência das palavras do que a contagem de coocorrência simples.
   - *Dica*: Explore medidas como **PMI** (Pointwise Mutual Information) e suas variantes suavizadas.

## Técnicas de Ponderação: tf-idf e PPMI

Para melhorar a qualidade das representações, são aplicadas técnicas de ponderação que ajustam os valores das matrizes para refletir melhor a importância das palavras.

### tf-idf (Term Frequency-Inverse Document Frequency)

O **tf-idf** é uma técnica de ponderação utilizada principalmente em matrizes termo-documento. Ela combina duas métricas:

1. **Term Frequency (tf)**: Frequência da palavra no documento, refletindo sua importância local.
2. **Inverse Document Frequency (idf)**: Logaritmo inverso da frequência de documentos que contêm a palavra, refletindo sua raridade global.

Matematicamente, o tf-idf é definido como [14]:

$$
\text{tf-idf}_{t,d} = \text{tf}_{t,d} \times \text{idf}_t
$$

Onde:

$$
\text{tf}_{t,d} = \begin{cases}
1 + \log_{10}(\text{count}(t,d)), & \text{se } \text{count}(t,d) > 0 \\
0, & \text{caso contrário}
\end{cases}
$$

$$
\text{idf}_t = \log_{10}\left(\frac{N}{\text{df}_t}\right)
$$

- $\text{count}(t,d)$ é o número de ocorrências do termo $t$ no documento $d$.
- $N$ é o número total de documentos.
- $\text{df}_t$ é o número de documentos que contêm o termo $t$.

> ⚠️ **Nota Importante**: O tf-idf dá maior peso a palavras que são frequentes em um documento específico, mas raras no corpus como um todo [15].

### PPMI (Positive Pointwise Mutual Information)

**PPMI** é uma medida de associação entre palavras usada principalmente em matrizes termo-termo. É definida como [16]:

$$
\text{PPMI}(w,c) = \max\left(0, \log_2\left(\frac{P(w,c)}{P(w)P(c)}\right)\right)
$$

Onde:

- $P(w,c)$ é a probabilidade de coocorrência das palavras $w$ e $c$.
- $P(w)$ e $P(c)$ são as probabilidades marginais de $w$ e $c$, respectivamente.

A PPMI ajusta a **Pointwise Mutual Information (PMI)** para zero nos casos em que a PMI é negativa.

> 💡 **Insight**: PPMI captura associações positivas entre palavras, ignorando associações negativas que são geralmente menos confiáveis em corpora pequenos [17].

### Comparação entre tf-idf e PPMI

| Característica                 | tf-idf                                | PPMI                                          |
| ------------------------------ | ------------------------------------- | --------------------------------------------- |
| Tipo de Matriz                 | Termo-Documento                       | Termo-Termo                                   |
| Captura                        | Importância de palavras em documentos | Associações entre palavras                    |
| Valores                        | Sempre não negativos                  | Não negativos (valores negativos são zerados) |
| Sensibilidade a palavras raras | Alta (através do componente idf)      | Alta (tende a superestimar associações raras) |

## Word2Vec: Um Modelo Preditivo de Embedding

**Word2Vec**, introduzido por Mikolov et al. (2013), é um modelo neural que aprende embeddings de palavras através de uma tarefa de previsão [18]. Existem duas variantes principais:

1. **Continuous Bag-of-Words (CBOW)**: Prevê a palavra alvo dado o contexto.
2. **Skip-gram**: Prevê o contexto dado uma palavra alvo.

Focaremos no modelo **Skip-gram**, que é mais eficaz para palavras raras e captura melhor as relações semânticas.

### Arquitetura Skip-gram

O modelo Skip-gram treina uma rede neural rasa para maximizar a probabilidade de palavras de contexto dado uma palavra alvo. A função objetivo é:

$$
\arg\max_\theta \sum_{t=1}^T \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j}|w_t; \theta)
$$

Onde:

- $w_t$ é a palavra alvo no tempo $t$.
- $c$ é o tamanho da janela de contexto.
- $\theta$ são os parâmetros do modelo (os embeddings).

A probabilidade $P(w_{t+j}|w_t)$ é modelada usando softmax:

$$
P(w_O|w_I) = \frac{\exp(v_{w_O}^\top v_{w_I})}{\sum_{w \in V} \exp(v_w^\top v_{w_I})}
$$

Onde:

- $v_{w_I}$ é o vetor de embedding da palavra de entrada (input word).
- $v_{w_O}$ é o vetor de embedding da palavra de saída (output word).
- $V$ é o tamanho do vocabulário [19].

O cálculo direto dessa expressão é computacionalmente custoso devido à soma sobre todo o vocabulário.

### Negative Sampling

Para tornar o treinamento computacionalmente viável, o Word2Vec utiliza o **Negative Sampling**, que simplifica a função objetivo ao considerar apenas um pequeno número de palavras negativas em cada atualização.

A função de perda simplificada é:

$$
\log \sigma(v_{w_O}^\top v_{w_I}) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)}\left[ \log \sigma(-v_{w_i}^\top v_{w_I}) \right]
$$

Onde:

- $\sigma(x) = \frac{1}{1 + \exp(-x)}$ é a função sigmoide.
- $k$ é o número de amostras negativas.
- $P_n(w)$ é a distribuição de ruído utilizada para amostrar palavras negativas [20].

> ✔️ **Destaque**: Negative Sampling permite o treinamento eficiente de Word2Vec em corpora de grande escala, mantendo a qualidade dos embeddings [21].

### Relação com Informação Mútua

O Word2Vec, especialmente com Negative Sampling, pode ser interpretado como uma forma de fatoração implícita de uma matriz de coocorrência ponderada, relacionando-se com medidas como a PMI. Essa interpretação ajuda a conectar modelos preditivos com modelos baseados em contagem.

### Perguntas Teóricas

1. **Derivação da Função de Perda com Negative Sampling**:
   - Derive a função de perda para o modelo Skip-gram com Negative Sampling e mostre como ela se relaciona com a maximização da informação mútua entre palavras e seus contextos.
   - *Dica*: Analise a função de perda como uma aproximação da maximização da probabilidade de dados observados.

2. **Efeito do Tamanho da Janela de Contexto**:
   - Analise teoricamente como o tamanho da janela de contexto no Skip-gram afeta as propriedades semânticas dos embeddings resultantes.
   - *Dica*: Janelas maiores capturam relações semânticas mais globais, enquanto janelas menores capturam relações sintáticas locais.

3. **Melhoria para Palavras Raras**:
   - Proponha e justifique matematicamente uma modificação no algoritmo de Negative Sampling que poderia melhorar a qualidade dos embeddings para palavras raras sem aumentar significativamente o custo computacional.
   - *Dica*: Considere ajustar a distribuição de ruído $P_n(w)$ para dar mais ênfase a palavras raras.

## Propriedades e Aplicações de Word Embeddings

### Captura de Relações Semânticas

Os word embeddings são capazes de capturar várias relações semânticas e sintáticas entre palavras, que podem ser reveladas através de operações vetoriais [22]:

1. **Similaridade**: Medida pelo cosseno entre vetores.

   $$
   \text{similaridade}(w_1, w_2) = \frac{v_{w_1} \cdot v_{w_2}}{\|v_{w_1}\| \|v_{w_2}\|}
   $$

   Onde $\|v_{w}\|$ é a norma do vetor $v_{w}$.

2. **Analogias Semânticas**: Capturadas por operações vetoriais aritméticas.

   $$
   v_{\text{rei}} - v_{\text{homem}} + v_{\text{mulher}} \approx v_{\text{rainha}}
   $$

   Essa propriedade permite resolver tarefas de analogias semânticas e sintáticas.

3. **Relações Hierárquicas e Clusters**: Palavras com significados relacionados tendem a formar clusters no espaço vetorial.

> 💡 **Insight**: Essas propriedades emergem do treinamento não supervisionado dos embeddings, demonstrando a capacidade dos modelos em capturar estruturas linguísticas complexas [23].

### Aplicações

Os word embeddings são aplicados em uma variedade de tarefas em NLP:

1. **Análise de Sentimento**: Embeddings capturam nuances emocionais, melhorando classificadores.
2. **Tradução Automática**: Embeddings multilingues permitem o alinhamento semântico entre línguas.
3. **Sistemas de Recomendação**: Representação de itens e usuários em um espaço comum facilita a recomendação personalizada.
4. **Análise Temporal de Linguagem**: Rastreamento de mudanças semânticas ao longo do tempo usando embeddings históricos [24].
5. **Reconhecimento de Entidades Nomeadas**: Melhora a identificação de entidades como nomes próprios, lugares, etc.

## Desafios e Limitações

Apesar dos avanços, os word embeddings enfrentam desafios importantes:

1. **Palavras Polissêmicas**: Embeddings estáticos não distinguem entre diferentes sentidos de uma mesma palavra (e.g., "banco" como instituição financeira ou assento).
2. **Viés**: Embeddings podem perpetuar e amplificar vieses presentes nos dados de treinamento, como estereótipos de gênero ou raça [25].
3. **Dependência de Contexto**: Não capturam variações contextuais no significado das palavras.

### Abordagens para Superação

- **Embeddings Contextuais**: Modelos como **ELMo**, **BERT** e **GPT** geram embeddings que dependem do contexto específico em que a palavra aparece.
- **Técnicas de Debiasing**: Métodos para identificar e remover componentes de viés nos embeddings.
- **Representações Multi-sentido**: Modelos que aprendem diferentes vetores para diferentes sentidos de uma palavra.

## Conclusão

Os word embeddings representam um avanço fundamental na representação computacional do significado das palavras. Ao mapear palavras para espaços vetoriais densos, eles capturam relações semânticas complexas e permitem uma variedade de aplicações em NLP e áreas correlatas. Modelos como **Word2Vec** e **GloVe** estabeleceram o padrão para embeddings estáticos, mas a área continua a evoluir com o desenvolvimento de representações cada vez mais sofisticadas e contextuais [26].

Uma compreensão profunda dos fundamentos matemáticos e estatísticos por trás dos word embeddings é crucial para avançar o estado da arte em NLP. Além disso, é importante abordar os desafios e limitações atuais para desenvolver aplicações que fazem uso efetivo dessas representações, de forma ética e responsável.

## Perguntas Teóricas Avançadas

1. **Word2Vec e Informação Mútua**:
   - Considerando a teoria da informação, demonstre como a otimização da função objetivo do Word2Vec se relaciona com a maximização da informação mútua entre palavras e seus contextos.
   - *Dica*: Relacione a função de perda do Negative Sampling com a maximização da PMI entre palavras e contextos.

2. **Comparação de Arquiteturas de Embedding**:
   - Proponha um framework teórico para quantificar e comparar a capacidade de diferentes arquiteturas de embedding (e.g., Skip-gram, GloVe, FastText) em capturar diferentes tipos de relações semânticas (sinonímia, antonímia, hiperonímia).
   - *Dica*: Considere métricas baseadas em tarefas específicas e medidas intrínsecas de qualidade.

3. **Dimensionalidade e Trade-off**:
   - Analise matematicamente como a dimensionalidade do espaço de embedding afeta o trade-off entre a capacidade de representação e a generalização.
   - *Dica*: Explore conceitos de teoria da informação e complexidade do modelo.

4. **Embeddings Dinâmicos e Mudanças Semânticas**:
   - Desenvolva um modelo teórico para embeddings dinâmicos que possam capturar mudanças semânticas ao longo do tempo ou em diferentes domínios.
   - *Dica*: Considere incorporar a dimensão temporal nos embeddings ou utilizar técnicas de modelagem de tópicos dinâmicos.

## Referências

[1] Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv preprint arXiv:1301.3781.

[2] Harris, Z. S. (1954). *Distributional Structure*. Word, 10(2-3), 146-162.

[3] Bengio, Y., Ducharme, R., Vincent, P., & Janvin, C. (2003). *A Neural Probabilistic Language Model*. Journal of Machine Learning Research, 3(Feb), 1137-1155.

[4] Turian, J., Ratinov, L., & Bengio, Y. (2010). *Word Representations: A Simple and General Method for Semi-Supervised Learning*. In Proceedings of the 48th annual meeting of the association for computational linguistics.

[5] Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation*. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP).

[6] Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT press.

[7] Firth, J. R. (1957). *A synopsis of linguistic theory 1930–1955*. Studies in linguistic analysis.

[8] Firth, J. R. (1957). *Papers in Linguistics 1934-1951*. Oxford University Press.

[9] Church, K. W., & Hanks, P. (1990). *Word Association Norms, Mutual Information, and Lexicography*. Computational Linguistics, 16(1), 22-29.

[10] Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). *Enriching Word Vectors with Subword Information*. Transactions of the Association for Computational Linguistics, 5, 135-146.

[11] Salton, G., & Buckley, C. (1988). *Term-weighting Approaches in Automatic Text Retrieval*. Information Processing & Management, 24(5), 513-523.

[12] Levy, O., & Goldberg, Y. (2014). *Neural Word Embedding as Implicit Matrix Factorization*. In Advances in neural information processing systems.

[13] Turney, P. D., & Pantel, P. (2010). *From Frequency to Meaning: Vector Space Models of Semantics*. Journal of artificial intelligence research, 37, 141-188.

[14] Ramos, J. (2003). *Using tf-idf to Determine Word Relevance in Document Queries*. In Proceedings of the first instructional conference on machine learning.

[15] Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

[16] Bullinaria, J. A., & Levy, J. P. (2007). *Extracting Semantic Representations from Word Co-occurrence Statistics: A Computational Study*. Behavior Research Methods, 39(3), 510-526.

[17] Levy, O., & Goldberg, Y. (2014). *Linguistic Regularities in Sparse and Explicit Word Representations*. In Proceedings of the eighteenth conference on computational natural language learning.

[18] Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). *Distributed Representations of Words and Phrases and Their Compositionality*. In Advances in neural information processing systems.

[19] Goldberg, Y., & Levy, O. (2014). *Word2Vec Explained: Deriving Mikolov et al.'s Negative-Sampling Word-Embedding Method*. arXiv preprint arXiv:1402.3722.

[20] Mnih, A., & Kavukcuoglu, K. (2013). *Learning Word Embeddings Efficiently with Noise-contrastive Estimation*. In Advances in neural information processing systems.

[21] Rong, X. (2014). *Word2Vec Parameter Learning Explained*. arXiv preprint arXiv:1411.2738.

[22] Mikolov, T., Yih, W. T., & Zweig, G. (2013). *Linguistic Regularities in Continuous Space Word Representations*. In Proceedings of the 2013 conference of the north american chapter of the association for computational linguistics: Human language technologies.

[23] Baroni, M., Dinu, G., & Kruszewski, G. (2014). *Don't Count, Predict! A Systematic Comparison of Context-counting vs. Context-predicting Semantic Vectors*. In Proceedings of the 52nd annual meeting of the association for computational linguistics.

[24] Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016). *Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change*. arXiv preprint arXiv:1605.09096.

[25] Bolukbasi, T., Chang, K. W., Zou, J. Y., Saligrama, V., & Kalai, A. (2016). *Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings*. In Advances in neural information processing systems.

[26] Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L. (2018). *Deep Contextualized Word Representations*. In Proceedings of the 2018 conference of the north american chapter of the association for computational linguistics.